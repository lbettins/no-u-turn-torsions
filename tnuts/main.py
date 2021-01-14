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
        get_initial_values
from tnuts.molconfig import get_energy_at, get_grad_at
from tnuts.mode import dicts_to_NModes
from tnuts.geometry import Geometry
from tnuts.thermo.thermo import MCThermoJob
from ape.sampling import SamplingJob
from tnuts.mc.dist import MyDist, MyPeriodic
from pymc3.distributions.dist_math import bound
from pymc3.distributions.transforms import TransformedDistribution
from pymc3_ext.distributions import transforms as tr

print('Running on PyMC3 v{}'.format(pm.__version__))

def NUTS_run(samp_obj,T,
        nsamples=1000, tune=200, nchains=10, ncpus=4, hpc=False,
        protocol='NUTS'):
    """
    Run the sampling job according to the given protocol (default: NUTS)
    """
    # Define the model and model parameters
    beta = 1/(constants.kB*T)*constants.E_h
    logpE = lambda E: -E    # E must be dimensionless
    logp, Z, tmodes = generate_umvt_logprior(samp_obj, T)
    syms = np.array([mode.get_symmetry_number() for mode in tmodes])
    Is = np.array([mode.get_I() for mode in tmodes])
    #Ks = np.array([beta*mode.get_spline_fn()(0,2) for mode in modes])
    variances = get_initial_mass_matrix(tmodes, T)
    geom = Geometry(samp_obj, samp_obj.torsion_internal, syms)
    energy_fn = Energy(geom, beta)
    lower = np.array([-np.pi/s for s in syms])
    upper = np.array([np.pi/s for s in syms])
    transform = tr.PeriodicTransform(lower=lower, upper=upper)
    n_d = len(tmodes)
    resolution = 10.0  #degrees
    step_scale = resolution*(np.pi/180) / (1/n_d)**(0.25)
    L = 2*np.pi/syms
    center_mod = lambda x: \
        ((x.transpose() % L) \
        - L*((x.transpose() % L)\
        // (L/2))).transpose()
    if not hpc:
        with pm.Model() as model:
            #Bound = pm.Bound(MyDist, lower=-np.pi/syms,
            #        upper=np.pi/syms)
            #x = Bound('x', logp, tmodes, T,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))
            x = pm.DensityDist('x', logp, shape=n_d,
                    testval=get_initial_values(tmodes, T))
            #x = MyDist('x', logp, tmodes, T,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))
            #x = MyPeriodic('x', logp, tmodes,
            #        shape=n_d,
            #        testval=get_initial_values(tmodes, T))
            #x = MyDist('x', logp, tmodes, T,
            #        shape=(n_d, 1),
            #        testval=get_initial_values(tmodes,T),
            #        transform=transform)
            xmod = pm.Deterministic('xmod', center_mod(x))
            bE = pm.Deterministic('bE', -logp(xmod)+\
                    (np.random.rand()-0.5)/10000)
            DeltaE = bE - (-logp(xmod))
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            #E_obs = pm.Potential('E_obs', logpE(DeltaE))
            E_obs = pm.DensityDist('E_obs', lambda E: logpE(E),
                    observed={'E':DeltaE})
        with model:
            step = pm.NUTS(
                    target_accept=0.65, scaling=variances, is_cov=True,
                    step_scale=step_scale, early_max_treedepth=6,
                    max_treedepth=6, adapt_step_size=False)
            trace = pm.sample(nsamples, tune=tune, step=step,
                    chains=nchains, cores=ncpus, discard_tuned_samples=False)
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
            if protocol == 'NUTS':
                nuts_kwargs = dict(target_accept=0.5,
                        step_scale=step_scale, early_max_treedepth=5,
                        max_treedepth=6, adapt_step_size=True)
                step = pm.NUTS(scaling=variances, is_cov=True,
                        **nuts_kwargs)
            trace = pm.sample(nsamples, tune=0, step=step,
                    chains=nchains, cores=1, start=start)
    print("Initial cov:\n", np.diag(variances))
    print("Trace cov:\n", np.atleast_1d(pm.trace_cov(trace,
        model=model)))
    print("Trace summary:\n", np.cov(trace.x.transpose()),
            np.mean(trace.x, axis=0)) 
    print("Trace summary: modx\n", np.cov(trace.xmod.transpose()),
            np.mean(trace.xmod, axis=0))
    Q = Z*np.mean(trace.a)
    model_dict = {'model' : model, 'trace' : trace,\
            'n' : nsamples, 'chains' : nchains, 'cores' : ncpus,\
            'tune' : tune, 'Q' : Q, 'Z' : Z, 'T' : T, 'samp_obj' : samp_obj,\
            'geom_obj' : geom, 'modes' : tmodes}
    thermo_obj = MCThermoJob(trace, T, samp_obj=samp_obj, model=model)
    a,b = thermo_obj.execute()
    print(a.loc[:,['mode', 'q', 'e', 's', 'protocol', 'sb_protocol']])
    pkl_file = '{label}_{nc}_{nburn}_{ns}_{T}K_{t_a}_{n}.p'
    trace_file = '{label}_{nc}_{nburn}_{ns}_{T}K_{t_a}_{n}_trace.p'
    n = 0
    pkl_kwargs = dict(label=samp_obj.label, nc=nchains,
            nburn=tune, ns=nsamples, T=T,
            t_a=0.5, n=n)
    while os.path.exists(os.path.join(samp_obj.output_directory, pkl_file.format(**pkl_kwargs))):
        n += 1
        pkl_kwargs['n'] = n
    #pickle.dump(model_dict,
    #        open(os.path.join(samp_obj.output_directory,pkl_file.format(**pkl_kwargs)),'wb'),
    #        protocol=4)
    pickle.dump(model_dict,
            open(os.path.join(samp_obj.output_directory,trace_file.format(**pkl_kwargs)),'wb'),
            protocol=4)
    if not hpc:
        plot_MC_torsion_result(trace,tmodes,T)
        print("Prior partition function:\t", Z)
        print("Posterior partition function:\t", np.mean(trace.a)*Z)
        print("Expected likelihood:\t", np.mean(trace.a))
        #print("Best step size:", trace.get_sampler_stats("step_size_bar")[:tune+1])
        #print("Step size:", trace.get_sampler_stats("step_size")[:tune+1])
        print(model_dict)

def generate_umvt_logprior(samp_obj, T):
    # Get the torsions from the APE object
    # With APE updates, should edit APE sampling.py to [only] sample torsions
    samp_obj.csv_path = os.path.join(samp_obj.output_directory,
            '{}_sampling_result.csv'.format(samp_obj.label))
    if os.path.exists(samp_obj.csv_path):
        os.remove(samp_obj.csv_path)
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
    return LogPrior(np.array(energy_array), T, np.array(sym_nums)), Z, modes

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
