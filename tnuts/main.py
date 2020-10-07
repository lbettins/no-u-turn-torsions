#!/usr/bin/env python3
import arviz as az
import dill as pickle
import numpy as np
import pymc3 as pm
import rmgpy.constants as constants
import os
import warnings
warnings.filterwarnings("ignore")
import theano.tensor as tt
from scipy.integrate import quad
from scipy import stats
from pymc3.distributions import Interpolated
from tnuts.ops import Energy
from tnuts.molconfig import get_energy_at, get_grad_at
from tnuts.mode import dicts_to_NModes
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
    energy_fn = Energy(get_energy_at, samp_obj, grad_fn=get_grad_at)
    #energy_fn = Energy(get_energy_at, samp_obj)
    n_d = len(modes)
    if not hpc:
        with pm.Model() as model:
            xi = pm.DensityDist('xi', logp, shape=n_d)
            x = pm.DensityDist('x', logp, shape=n_d, testval=xi)
            Etrial = pm.Deterministic('Etrial', -logp(x)+\
                    (np.random.rand()-0.5)/50)   # For computational ease
            Eprior = pm.Deterministic('Eprior', -logp(x))
            DeltaE = pm.Deterministic('DeltaE', Etrial-Eprior)
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
    else:
        with pm.Model() as model:
            xi = pm.DensityDist('xi', logp, shape=n_d)
            x = pm.DensityDist('x', logp, shape=n_d, testval=xi)
            Etrial = pm.Deterministic('Etrial', beta*energy_fn(x))
            Eprior = pm.Deterministic('Eprior', -logp(x))
            DeltaE = pm.Deterministic('DeltaE', Etrial-Eprior)
            alpha = pm.Deterministic('a', np.exp(-DeltaE))
            E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
    with model:
        step = [pm.NUTS(x), pm.Metropolis(xi)]
        trace = pm.sample(nsamples, tune=tune, step=step, 
                chains=nchains, cores=1, discard_tuned_samples=True)
    Q = Z*np.mean(trace.a)
    model_dict = {'model' : model, 'trace' : trace,\
            'n' : nsamples, 'chains' : nchains, 'cores' : ncpus,\
            'tune' : tune, 'Q' : Q}
    pickle.dump(model_dict,
            open(os.path.join(samp_obj.output_directory,
                '{}_trace.p'.format(samp_obj.label)),'wb'))
    if not hpc:
        plot_MC_torsion_result(trace,modes,T)
        print("Prior partition function:\t", Z)
        print("Posterior partition function:\t", np.mean(trace.a)*Z)
        print("Expected likelihood:\t", np.mean(trace.a))

def generate_umvt_logprior(samp_obj, T):
    from tnuts.ops import LogPrior
    # Get the torsions from the APE object
    samp_obj.parse()
    # With APE updates, should edit APE sampling.py to [only] sample torsions
    xyz_dict, energy_dict, mode_dict = samp_obj.sampling()
    modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=samp_obj)
    # Get the 1D Potential Energy functions
    energy_array = [mode.get_spline_fn() for mode in modes]
    # Get the symmetry numbers for each 1D torsion
    sym_nums = np.array([mode.get_symmetry_number() for mode in modes])
    Z = np.array([quad(lambda x: fn(x), 0, 2*np.pi/s)[0]\
            for fn,s in zip(energy_array,sym_nums)]).sum()

    # Return theano op LogPrior and partition function
    return LogPrior(energy_array, T, sym_nums), Z, modes
    
def plot_MC_torsion_result(trace, NModes, T=300):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('MacOSX')
    beta = 1/(constants.kB*T)*constants.E_h
    n_d = len(NModes)
    logpriors = [mode.get_spline_fn() for mode in NModes]
    syms = [mode.get_symmetry_number() for mode in NModes]
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
        samples=np.vstack(trace['x'])
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
