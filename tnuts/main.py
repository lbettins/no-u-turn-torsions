#!/usr/bin/env python3
import arviz as az
import dill as pickle
import numpy as np
import pymc3 as pm
import rmgpy.constants as constants
import os
import warnings
warnings.filterwarnings("ignore")
import theano
import theano.tensor as tt
from scipy.integrate import quad
from scipy import stats
from pymc3.distributions import Interpolated
from tnuts.mc.ops import Energy, LogPrior
from tnuts.mc.metrics import get_step_for_trace, get_initial_mass_matrix,\
        get_initial_values, get_priors_dict
from tnuts.molconfig import get_energy_at, get_grad_at
from tnuts.mode import dicts_to_NModes
from tnuts.geometry import Geometry
from tnuts.thermo.thermo import MCThermoJob
from ape.sampling import SamplingJob
from tnuts.mc.dist import MyDist, MyPeriodic
from pymc3.distributions.dist_math import bound
from pymc3.distributions.transforms import TransformedDistribution
import pymc3_ext as pmx
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

print('Running on Theano v{}'.format(theano.__version__))
print('Running on PyMC3 v{}'.format(pm.__version__))

class MCMCTorsions:
    def __init__(self, samp_obj, T, mckwargs_dict, hpc=1, ncpus=4):
        self.samp_obj = samp_obj
        self.T = T
        self.beta = 1/(constants.kB*T)*constants.E_h

        # Set up sampling information so that csv is returned for Thermo
        # (important)!
        self.samp_obj.csv_path = os.path.join(samp_obj.output_directory,
                '{}_sampling_result.csv'.format(samp_obj.label))
        #if os.path.exists(samp_obj.csv_path):
        #    os.remove(samp_obj.csv_path)
        xyz_dict, energy_dict, mode_dict = self.samp_obj.sampling()
        self.tmodes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=self.samp_obj, just_tors=True)
        self.syms = np.array([mode.get_symmetry_number()\
                for mode in self.tmodes])
        self.geom = Geometry(self.samp_obj,
                self.samp_obj.torsion_internal, self.syms)
        self.energy_fn = Energy(self.geom, self.beta)

        # Model parameters
        self.n_d = len(self.tmodes)
        self.priors_dict = get_priors_dict(self.tmodes, self.T)

        self.mckwargs = mckwargs_dict
        self.hpc = hpc
        self.ncpus = ncpus

        self.test_training_path = os.path.join(self.samp_obj.output_directory,
                    'test-results', self.samp_obj.label)
        self.training_path = os.path.join(self.samp_obj.output_directory,
                    'results', self.samp_obj.label)
        if not os.path.exists(self.test_training_path):
            os.makedirs(self.test_training_path)
        if not os.path.exists(self.training_path):
            os.makedirs(self.training_path)
        self.cov_path = os.path.join(self.training_path, 'cov-{}-{}')
        if not self.hpc:
            self.cov_path = os.path.join(self.test_training_path, 'cov-{}-{}')
        
    def get_model(self, prior, test=False):
        logpE = lambda E: -E    # for dimensionless E
        #logp, Z = self.generate_logprior(prior)
        logp = self.priors_dict[prior]['logp']
        if not self.hpc or test:
            energy_fn = lambda phi: -self.priors_dict['umvt']['logp'](phi)
        else:
            energy_fn = Energy(self.geom, self.beta)
        with pm.Model() as model:
            x = pmx.Periodic('x', shape=self.n_d,
                    lower=-np.pi/self.syms,
                    upper=np.pi/self.syms)
            phi = pm.DensityDist('phi', logp, observed=x)
            bE = pm.Deterministic('bE', energy_fn(phi))
            DeltaE = bE + logp(phi)
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            Pot = pm.Potential('pot', logpE(DeltaE))
        return model

    def sample(self, prior='umvt', method='NUTS',
            res=22.0, retrain=False, train=True, **step_kwargs):
        model = self.get_model(prior)
        if retrain:
            self.train_model()
        variances = self.read_cov(method, prior)
        #step_scale = res*(np.pi/180) / (self.n_d)**(0.25)
        step_scale = res*np.pi/180 / (1/self.n_d)**0.25
        if method == 'NUTS':
            with model:
                step = pm.NUTS(scaling=variances, is_cov=True,
                        step_scale=step_scale, adapt_step_size=False,
                        **step_kwargs)
        elif method == 'HMC':
            scaling = variances
            with model:
                step = pm.HamiltonianMC(step_scale=step_scale,
                        adapt_step_size=False,
                        **step_kwargs)
        elif method == 'MH':
            S = variances
            with model:
                step = pm.Metropolis()
        with model:
            trace = pm.sample(step=step, cores=1, **self.mckwargs)
        self.pickle(trace, prior, method, model, dict(**step_kwargs))
        myfile='corner_{ns}_{method}_{prior}.png'.format(
                ns=self.mckwargs['draws'],
                method=method, prior=prior)
        results = 'results' if self.hpc else 'test-results'
        mypath = os.path.join(self.samp_obj.output_directory,
                    results, self.samp_obj.label, method, prior, myfile)
        save_plot(trace, self.tmodes, self.T, mypath)
        if not self.hpc:
            plot_MC_torsion_result(trace, self.tmodes, self.T)
            #full_cov = self.priors_dict[prior]['full_cov']
            def flat_t(var):
                x = trace[str(var)]
                return x.reshape((x.shape[0], np.prod(x.shape[1:], dtype=int)))
            #print("Predicted full cov:", np.diag(
            #    self.priors_dict[prior]['full_cov']))
            #print("Actual full cov:", pm.trace_cov(trace, model=model))
            #print("Actual full cov:", np.cov(trace.x.T))

        return trace

    def write_(self, cov, method, prior):
        np.save(self.cov_path.format(method,prior), cov)

    def read_cov(self, method, prior):
        cov_path = self.cov_path.format(method,prior) + '.npy'
        if not os.path.exists(cov_path):
            print(cov_path,
                    "doesn't exist! Auto-training 500 iters")
            self.train_model(method, prior)
        with open(cov_path, 'rb') as f:
            cov = np.load(f)
        return cov

    def train_model(self, method, prior, res=22.):
        test_model = self.get_model(prior, test=True)
        with test_model:
            # True step size is scaled down
            # by (1/n)^(1/4), so that a res of 20º -> 15º sampling in 3-d
            # therefore, we scale UP by (1/n)^(1/4) so when we specify
            # 20º, we get 20º!
            step_scale = res*np.pi/180 / (1/self.n_d)**0.25
            if method == 'MH':
                step = pm.Metropolis()
                test_trace = pm.sample(step=step, cores=1,
                        chains=1, draws=1, tune=8000,
                        discard_tuned_samples=False)
                cov = pm.trace_cov(test_trace)
                self.write_(cov, method, prior)
                return
            elif method == 'HMC':
                step = pm.HamiltonianMC(step_scale=step_scale) 
                test_trace = pm.sample(step=step, cores=1,
                        chains=1, draws=1, tune=8000,
                        discard_tuned_samples=False)
                cov = pm.trace_cov(test_trace)
                self.write_(cov, method, prior)
                return
            elif method == 'NUTS':
                step = pm.NUTS(step_scale=step_scale, adapt_step_size=False,
                        target_accept=0.8)
                test_trace = pm.sample(step=step, cores=1,
                        chains=1, draws=1, tune=8000,
                        discard_tuned_samples=False)
                cov = pm.trace_cov(test_trace)
                accept = test_trace.get_sampler_stats("mean_tree_accept")
                print("Mean acceptance for method",method,
                        "and prior",prior,"is",accept.mean())
                print("Step size is",test_trace.get_sampler_stats("step_size_bar"))
                t_a = accept.mean() if accept.mean() > 0.5 else 0.8
        training_model = self.get_model(prior)
        # Method is NUTS
        # NUTS tuning will be a little more intense:
        with training_model:
            step = pm.NUTS(scaling=cov, is_cov=True,
                    step_scale=step_scale, adapt_step_size=False,
                    target_accept=t_a)
            train_trace = pm.sample(step=step, cores=1,
                    chains=1, draws=1, tune=500,
                    discard_tuned_samples=False)
            cov = pm.trace_cov(train_trace)
            #print("Step size is",train_trace.get_sampler_stats("step_size_bar"))
        self.write_(cov, method, prior)

    def compute_thermo(self, trace, T):
        thermo_obj = MCThermoJob(trace, T,
                sampT=self.T, samp_obj=self.samp_obj)
        a,b = thermo_obj.execute()
    
    def pickle(self, trace, prior, method, model, step_kwargs_dict):
        model_dict = dict(trace=trace, model=model,
                Z=self.priors_dict[prior]['Z'],
                samp_obj=self.samp_obj,
                geom=self.geom,
                tmodes=self.tmodes,
                prior=prior, method=method,
                step_kwargs_dict=step_kwargs_dict,
                **self.mckwargs)
        trace_file = '{label}_{method}_{prior}p_{T}K_{ns}_trace{n}.p'
        if not self.hpc:
            pickle_path = os.path.join(self.samp_obj.output_directory,
                    'test-results', self.samp_obj.label, method, prior)
        else:
            pickle_path = os.path.join(self.samp_obj.output_directory,
                    'results', self.samp_obj.label, method, prior)
        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        n = 0
        pkl_kwargs = dict(label=self.samp_obj.label,
                method=method, prior=prior,
                T=self.T, ns=model_dict['draws'], n=n)
        print("File to be written to: ", pickle_path)
        while os.path.exists(os.path.join(pickle_path,
            trace_file.format(**pkl_kwargs))):
            n += 1
            pkl_kwargs['n'] = n
        print("File name will be", trace_file.format(**pkl_kwargs))
        # SAVE NP ARRAYS FOR EASE
        Efile = 'E{}.npy'.format(n)
        phifile = 'phi{}.npy'.format(n)
        afile = 'a{}.npy'.format(n)
        Epath = os.path.join(pickle_path, Efile)
        phipath = os.path.join(pickle_path, phifile)
        apath = os.path.join(pickle_path, afile)
        np.save(Epath, trace.bE)
        np.save(phipath, trace.x)
        np.save(apath, trace.a)
        with open(os.path.join(pickle_path,
            trace_file.format(**pkl_kwargs)), 'wb') as f:
            pickle.dump(model_dict, f, protocol=4)
        print("File written.")



def NUTS_run(samp_obj,T,
        nsamples=1000, tune=200, nchains=1, ncpus=4, hpc=False,
        method='NUTS', prior='umvt'):
    """
    Run the sampling job according to the given protocol (default: NUTS)
    """
    # Define the model and model parameters
    beta = 1/(constants.kB*T)*constants.E_h
    logpE = lambda E: -E    # E must be dimensionless
    logp, Z, tmodes = generate_logprior(samp_obj, T, prior)
    syms = np.array([mode.get_symmetry_number() for mode in tmodes])
    Ks = np.array([beta*mode.get_spline_fn()(0,2) for mode in tmodes])
    variances = get_initial_mass_matrix(tmodes, T)
    #variances = 1/Ks
    geom = Geometry(samp_obj, samp_obj.torsion_internal, syms)
    energy_fn = Energy(geom, beta)
    n_d = len(tmodes)
    resolution = 10.0  #degrees
    step_scale = resolution*(np.pi/180) / (1/n_d)**(0.25)
    L = 2*np.pi/syms
    center_mod = lambda x: \
        ((x.transpose() % L) - L*((x.transpose() % L) // (L/2))).transpose()
    if not hpc:
        with pm.Model() as model:
            xy = pmx.UnitDisk('xy', shape=(2,n_d))
            x = pm.Deterministic('x', tt.arctan2(xy[1], xy[0]))
            phi = pm.DensityDist('phi', logp, observed=x)
            bE = pm.Deterministic('bE', -logp(phi)+\
                    (np.random.rand()-0.5)/10000)
            DeltaE = bE - (-logp(phi))
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.Potential('E_obs', logpE(DeltaE))
            #E_obs = pm.DensityDist('E_obs', lambda E: logpE(E),
            #        observed={'E':DeltaE})
        with model:
            print("Method is", method)
            if method == 'NUTS':
                pass
                #nuts_kwargs = dict(target_accept=0.5,
                #        step_scale=step_scale, early_max_treedepth=5,
                #        max_treedepth=6, adapt_step_size=False)
                #step = pm.NUTS(scaling=variances, is_cov=True,
                #        **nuts_kwargs)
                step = pm.NUTS(target_accept=0.6,
                        step_scale=step_scale, adapt_step_size=False)
                trace = pm.sample(1000, cores=1, target_accept=0.6)
            elif method == 'HMC':
                return
                hmc_kwargs = dict(target_accept=0.5,
                        step_scale=step_scale)
                step = pm.HamiltonianMC(scaling=variances,
                        is_cov=True, **hmc_kwargs)
                trace = pm.sample(nsamples, tune=tune, step=step,
                    chains=nchains, cores=1, discard_tuned_samples=False)
            elif method == 'MH':
                mh_kwargs = dict(S=variances)
                step = pm.Metropolis(**mh_kwargs)
            #step = pm.NUTS(
            #        target_accept=0.65, scaling=variances, is_cov=True,
            #        step_scale=step_scale, early_max_treedepth=6,
            #        max_treedepth=6, adapt_step_size=False)
                trace = pm.sample(nsamples, tune=tune, step=step,
                    chains=nchains, cores=1, discard_tuned_samples=False)
    else:
        with pm.Model() as model:
            #x = pm.DensityDist('x', logp, shape=n_d, testval=np.random.rand(n_d)*2*np.pi)
            x = pm.DensityDist('x', logp, shape=n_d,
                    testval=get_initial_values(tmodes, T))
            xmod = pm.Deterministic('xmod', center_mod(x))
            bE = pm.Deterministic('bE', energy_fn(x))
            DeltaE = (bE)-(-logp(x))
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.DensityDist('E_obs',
                    lambda E: logpE(E), observed={'E':DeltaE})
        with model:
            if method == 'NUTS':
                nuts_kwargs = dict(target_accept=0.5,
                        step_scale=step_scale, early_max_treedepth=5,
                        max_treedepth=6, adapt_step_size=False)
                step = pm.NUTS(scaling=variances, is_cov=True,
                        **nuts_kwargs)
            elif method == 'HMC':
                step = pm.HamiltonianMC()
                pass
            elif method == 'MH':
                step = pm.Metropolis()
                pass
            trace = pm.sample(nsamples, tune=tune, step=step,
                    chains=nchains, cores=1)
            #<-  Indent the following lines to here after debugging:
    pickle_the_model(trace, model, Z, samp_obj, geom, tmodes,
            ncpus, nchains, tune, nsamples, T,
            prior=prior, method=method)
    #thermo_obj = MCThermoJob(trace, T, sampT=T, samp_obj=samp_obj, model=model)
    #a,b = thermo_obj.execute()
    if not hpc:
        plot_MC_torsion_result(trace,tmodes,T)
        
def pickle_the_model(trace, model, Z, samp_obj, geom, tmodes,
        ncpus, nchains, tune, nsamples, T,
        prior, method,
        step_kwargs_dict=None, **samp_kwargs):
    Q = Z*np.mean(trace.a)
    model_dict = dict(trace=trace, model=model, Z=Z, samp_obj=samp_obj,
            geom=geom, tmodes=tmodes, prior=prior, method=method,
            step_kwargs_dict=step_kwargs_dict, **samp_kwargs)
    model_dict = {'model' : model, 'trace' : trace,\
            'n' : nsamples, 'chains' : nchains, 'cores' : ncpus,\
            'tune' : tune, 'Q' : Q, 'Z' : Z, 'T' : T, 'sampT' : T,\
            'samp_obj' : samp_obj,\
            'geom_obj' : geom, 'modes' : tmodes}
    trace_file = '{label}_{method}_{prior}-prior_{T}K_{nc}_{nburn}_{ns}_{n}_trace.p'
    n = 0
    pkl_kwargs = dict(label=samp_obj.label, nc=nchains,
            nburn=tune, ns=nsamples, T=T,
            n=n, prior=prior, method=method)
    pickle_path = os.path.join(samp_obj.output_directory,
            'nuts_out', samp_obj.label)
    print("File to be written to: ", pickle_path)
    while os.path.exists(os.path.join(pickle_path,
        trace_file.format(**pkl_kwargs))):
        n += 1
        pkl_kwargs['n'] = n
    print("File name will be", trace_file.format(**pkl_kwargs))
    with open(os.path.join(pickle_path,
        trace_file.format(**pkl_kwargs)), 'wb') as f:
        pickle.dump(model_dict, f, protocol=4)
    print("File written.")

def generate_logprior(samp_obj, T, prior):
    # Get the torsions from the APE object
    # With APE updates, should edit APE sampling.py to [only] sample torsions
    if prior=='umvt':
        samp_obj.csv_path = os.path.join(samp_obj.output_directory,
                '{}_sampling_result.csv'.format(samp_obj.label))
        #if os.path.exists(samp_obj.csv_path):
        #    os.remove(samp_obj.csv_path)
        xyz_dict, energy_dict, mode_dict = samp_obj.sampling()
        modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=samp_obj, just_tors=True)
        energy_array = []
        sym_nums = []
        Z = 1.
        for mode in modes:
            if not mode.is_tors():
                continue
            # Get the 1D Potential Energy functions
            energy_array.append(mode.get_spline_fn())
            # Get the symmetry numbers for each 1D torsion
            sym_nums.append(mode.get_symmetry_number())
            Z *= mode.get_classical_partition_fn(T)
        # Return theano op LogPrior and partition function
        return LogPrior(np.array(energy_array), T, np.array(sym_nums)),\
                Z, modes
    elif prior=='ho':
        pass
    elif prior=='uniform':
        pass


def save_plot(trace, NModes, T, abs_path_to_figure):
    import matplotlib.pyplot as plt
    beta = 1/(constants.kB*T)*constants.E_h
    if NModes is not None:
        n_d = len(NModes)
        logpriors = [mode.get_spline_fn() for mode in NModes]
        syms = np.array([mode.get_symmetry_number() for mode in NModes])
        xvals = [np.linspace(-np.pi/sig, np.pi/sig, 500) for sig in syms]
        yvals = [np.exp(-beta*V(xvals[i]))/quad(lambda x: np.exp(-beta*V(x)),
            -np.pi/sig,np.pi/sig)[0]\
                for i,V,sig in zip(range(n_d),logpriors,syms)]
        for i in range(n_d):
            print("Symmetry values for mode",i,"is",syms[i])
            plt.figure(2)
            fullx = np.linspace(-np.pi,np.pi,500)
            plt.plot(fullx, logpriors[i](fullx))
    else:
        n_d = len(trace.x[0])
        logpriors = None
        syms = np.ones(n_d)
    if n_d >= 2:
        import corner
        hist_kwargs = dict(density=True)
        xsamples = [trace['x']%(2*np.pi/syms)]
        xsamples = np.array([np.array([xi if xi < np.pi/sym \
                                else xi-2*np.pi/sym \
                                for sym,xi in zip(syms,x)]) for x in xsamples[0]])
        #print(xsamples)
        samples=np.vstack(xsamples)
        figure = corner.corner(samples,
                labels=["$x_{{{0}}}$".format(i) for i in range(1, n_d+1)],
                hist_kwargs=hist_kwargs)

        # Extract the axes
        axes = np.array(figure.axes).reshape((n_d, n_d))
        # Loop over the diagonal
        if NModes is not None:
            for i in range(n_d):
                ax = axes[i, i]
                ax.plot(xvals[i],yvals[i], color="b")
    plt.savefig(abs_path_to_figure, format='png')

def plot_MC_torsion_result(trace, NModes=None, T=300):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('MacOSX')
    beta = 1/(constants.kB*T)*constants.E_h
    if NModes is not None:
        n_d = len(NModes)
        logpriors = [mode.get_spline_fn() for mode in NModes]
        syms = np.array([mode.get_symmetry_number() for mode in NModes])
        xvals = [np.linspace(-np.pi/sig, np.pi/sig, 500) for sig in syms]
        yvals = [np.exp(-beta*V(xvals[i]))/quad(lambda x: np.exp(-beta*V(x)),
            -np.pi/sig,np.pi/sig)[0]\
                for i,V,sig in zip(range(n_d),logpriors,syms)]
        for i in range(n_d):
            print("Symmetry values for mode",i,"is",syms[i])
            plt.figure(2)
            fullx = np.linspace(-np.pi,np.pi,500)
            plt.plot(fullx, logpriors[i](fullx))
    else:
        n_d = len(trace.x[0])
        logpriors = None
        syms = np.ones(n_d)
    if n_d >= 2:
        import corner
        hist_kwargs = dict(density=True)
        xsamples = [trace['x']%(2*np.pi/syms)]
        xsamples = np.array([np.array([xi if xi < np.pi/sym \
                                else xi-2*np.pi/sym \
                                for sym,xi in zip(syms,x)]) for x in xsamples[0]])
        #print(xsamples)
        samples=np.vstack(xsamples)
        figure = corner.corner(samples,
                labels=["$x_{{{0}}}$".format(i) for i in range(1, n_d+1)],
                hist_kwargs=hist_kwargs)

        # Extract the axes
        axes = np.array(figure.axes).reshape((n_d, n_d))
        # Loop over the diagonal
        if NModes is not None:
            for i in range(n_d):
                ax = axes[i, i]
                ax.plot(xvals[i],yvals[i], color="b")
    plt.show()

if __name__=='__main__':
    T = 300

#print("Initial cov:\n", np.diag(variances))
    #print("Trace cov:\n", np.atleast_1d(pm.trace_cov(trace,
    #    model=model)))
    #print("Trace summary:\n", np.cov(trace.x.transpose()),
    #        np.mean(trace.x, axis=0)) 
    #print("Trace summary: modx\n", np.cov(trace.xmod.transpose()),
    #        np.mean(trace.xmod, axis=0))
    #print("Best step size:", trace.get_sampler_stats("step_size_bar")[:tune+1])
    #print("Step size:", trace.get_sampler_stats("step_size")[:tune+1])
#print("Prior partition function:\t", Z)
#        print("Posterior partition function:\t", np.mean(trace.a)*Z)
#        print("Expected likelihood:\t", np.mean(trace.a))
#        print(model_dict)
#
#print(a.loc[:,['mode', 'q', 'e', 's', 'protocol', 'sb_protocol']])
        #Bound = pm.Bound(MyDist, lower=-np.pi/syms,
            #        upper=np.pi/syms)
            #x = Bound('x', logp, tmodes, T,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))
            #x = MyDist('x', logp, tmodes, T,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))
            #x = MyPeriodic('x', logp, tmodes,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))

