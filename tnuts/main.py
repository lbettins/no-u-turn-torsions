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
from tnuts.ops import Energy
from tnuts.molconfig import get_energy_at, get_grad_at
from tnuts.mode import dicts_to_NModes
from tnuts.geometry import Geometry
from ape.sampling import SamplingJob

print('Running on PyMC3 v{}'.format(pm.__version__))

def NUTS_run(samp_obj,T,
        nsamples=1000, tune=200, nchains=10, ncpus=4, hpc=False):
    """
    """
    # Define the model and model parameters
    beta = 1/(constants.kB*T)*constants.E_h
    logpE = lambda E: -E    # E must be dimensionless
    logp, Z, modes = generate_umvt_logprior(samp_obj, T)
    syms = np.array([mode.get_symmetry_number() for mode in modes])
    Is = np.array([mode.get_I() for mode in modes])
    #Ks = np.array([beta*mode.get_spline_fn()(0,2) for mode in modes])
    variances = get_initial_mass_matrix(modes, T)
    geom = Geometry(samp_obj, samp_obj.torsion_internal, syms)
    energy_fn = Energy(geom, beta)
    n_d = len(modes)
    resolution = 5.0  #degrees
    step_scale = resolution*(np.pi/180) / (1/n_d)**(0.25)
    if not hpc:
        with pm.Model() as model:
            #x = pm.DensityDist('x', logp, shape=n_d, testval=np.random.uniform(-1,1,n_d)*np.pi)
            x = pm.DensityDist('x', logp, shape=n_d)
            DeltaE = (-logp(x)+(np.random.rand()-0.5)/500) -\
                    (-logp(x))
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
        with model:
            step = pm.NUTS(target_accept=0.9, step_scale=step_scale, early_max_treedepth=6,
                    max_treedepth=7, adapt_step_size=False)
            step = pm.NUTS(scaling=1/Ks, is_cov=True,
                    step_scale=step_scale, early_max_treedepth=6,
                    max_treedepth=6, adapt_step_size=True)
            step = pm.NUTS(target_accept=0.65, scaling=variances, is_cov=True,
                    step_scale=step_scale, early_max_treedepth=6,
                    max_treedepth=6, adapt_step_size=False)
            trace = pm.sample(nsamples, tune=tune, step=step, 
                    chains=nchains, cores=ncpus, discard_tuned_samples=False)
    else:
        with pm.Model() as model:
            #x = pm.DensityDist('x', logp, shape=n_d, testval=np.random.rand(n_d)*2*np.pi)
            x = pm.DensityDist('x', logp, shape=n_d, testval=np.zeros(n_d))
            bE = pm.Deterministic('bE', energy_fn(x))
            DeltaE = (bE)-(-logp(x))
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
        with model:
            #step = pm.NUTS(target_accept=0.7, scaling=1/Ks, is_cov=True,
            #        step_scale=step_scale, early_max_treedepth=6,
            #        max_treedepth=6, adapt_step_size=False)
            step = pm.NUTS(target_accept=0.5, scaling=variances, is_cov=True,
                    step_scale=step_scale, early_max_treedepth=4,
                    max_treedepth=5, adapt_step_size=True)
            trace = pm.sample(nsamples, tune=tune, step=step, 
                    chains=nchains, cores=1, discard_tuned_samples=False)
    Q = Z*np.mean(trace.a)
    model_dict = {'model' : model, 'trace' : trace,\
            'n' : nsamples, 'chains' : nchains, 'cores' : ncpus,\
            'tune' : tune, 'Q' : Q, 'Z' : Z, 'T' : T, 'samp_obj' : samp_obj,\
            'geom_obj' : geom}
    pkl_file = '{label}_{nc}_{nburn}_{ns}_{T}K_{t_a}_{n}.p'
    n = 0
    pkl_kwargs = dict(label=samp_obj.label, nc=nchains,
            nburn=tune, ns=nsamples, T=T,
            t_a=0.5, n=n)
    while os.path.exists(os.path.join(samp_obj.output_directory, pkl_file.format(**pkl_kwargs))):
        n += 1
        pkl_kwargs['n'] = n
    pickle.dump(model_dict,
            open(os.path.join(samp_obj.output_directory,pkl_file.format(**pkl_kwargs)),'wb'))
    if not hpc:
        plot_MC_torsion_result(trace,modes,T)
        print("Prior partition function:\t", Z)
        print("Posterior partition function:\t", np.mean(trace.a)*Z)
        print("Expected likelihood:\t", np.mean(trace.a))
        print("Best step size:", trace.get_sampler_stats("step_size_bar")[:tune+1])
        print("Step size:", trace.get_sampler_stats("step_size")[:tune+1])

def get_initial_mass_matrix(modes, T):
    from scipy.integrate import quad
    import rmgpy.constants as constants
    fns = np.array([mode.get_spline_fn() for mode in modes])
    sym_nums = np.array([mode.get_symmetry_number() for mode in modes])
    beta = 1/(constants.kB*T)*constants.E_h
    variances = []
    Zs = []
    for fn,s in zip(fns,sym_nums):
        Z = quad(lambda x: np.exp(-beta*fn(x)), -np.pi/s, np.pi/s)[0]
        var = np.power(Z,-1)*quad(lambda y: np.power(y,2)*np.exp(-beta*fn(y)), -np.pi/s, np.pi/s)[0] -\
                np.power(1/Z*quad(lambda y: y*np.exp(-beta*fn(y)), -np.pi/s, np.pi/s)[0],2)
        Zs.append(Z)
        variances.append(var)
    return np.array(variances)

def generate_umvt_logprior(samp_obj, T):
    from tnuts.ops import LogPrior
    # Get the torsions from the APE object
    # With APE updates, should edit APE sampling.py to [only] sample torsions
    xyz_dict, energy_dict, mode_dict = samp_obj.sampling()
    modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=samp_obj)
    # Get the 1D Potential Energy functions
    energy_array = [mode.get_spline_fn() for mode in modes]
    # Get the symmetry numbers for each 1D torsion
    sym_nums = np.array([mode.get_symmetry_number() for mode in modes])
    beta = 1/(constants.kB*T)*constants.E_h
    Z = np.array([quad(lambda x: np.exp(-beta*fn(x)), -np.pi/s, np.pi/s)[0]\
            for fn,s in zip(energy_array,sym_nums)]).prod()

    # Return theano op LogPrior and partition function
    return LogPrior(energy_array, T, sym_nums), Z, modes
    
def plot_MC_torsion_result(trace, NModes, T=300):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('MacOSX')
    beta = 1/(constants.kB*T)*constants.E_h
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
        for i in range(n_d):
            ax = axes[i, i]
            ax.plot(xvals[i],yvals[i], color="b")
    plt.show()

if __name__=='__main__':
    T = 300
