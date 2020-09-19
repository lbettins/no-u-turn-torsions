#!/usr/bin/env python3
import arviz as az
import dill as pickle
import numpy as np
import pymc3 as pm
import rmgpy.constants as constants
import os
import warnings
#warnings.filterwarnings("ignore")

import pandas as pd
#import seaborn as sns

import theano.tensor as tt
from scipy.integrate import quad
from scipy import stats
from pymc3.distributions import Interpolated

from tnuts.ops import Energy, LogPrior
from tnuts.molconfig import get_energy_at
from tnuts.mode import dicts_to_NModes
from ape.sampling import SamplingJob

#sns.set_context('notebook')
#plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

def run_loglike(samp_obj,T,
        nsamples=1000, tune=200, nchains=10, ncpus=4):
    """
    """
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
    # Get the number of torsions (in this case the dimensionality)
    n_d = samp_obj.n_rotors

    # Define the model and model parameters
    beta = 1/(constants.kB*T)*constants.E_h
    logpE = lambda E: -E    # E must be dimensionless
    logp = LogPrior(energy_array, T, sym_nums)
    energy_fn = Energy(get_energy_at, samp_obj)
    with pm.Model() as model:
        xi = pm.DensityDist('xi', logp, shape=n_d)
        x = pm.DensityDist('x', logp, shape=n_d, testval=xi)
        v = x
        Etrial = pm.Deterministic('Etrial', beta*energy_fn(v))
        #Etrial = pm.Deterministic('Etrial', -logp(v))
        Eprior = pm.Deterministic('Eprior', -logp(v))
        DeltaE = pm.Deterministic('DeltaE', Etrial-Eprior)
        E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
    with model:
        step = [pm.NUTS(x), pm.Metropolis(xi)]
        trace = pm.sample(nsamples, tune=tune, step=step, 
                chains=nchains, cores=ncpus, discard_tuned_samples=True)
        #ppc = pm.sample_posterior_predictive(trace, var_names=['x','xi','E_obs'])
    #plot_MC_torsion_result(trace,modes,T)
    model_dict = {'model' : model, 'trace' : trace,\
            'n' : nsamples, 'chains' : nchains, 'cores' : ncpus,\
            'tune' : tune}
    pickle.dump(model_dict,
            open(os.path.join(samp_obj.output_directory,
                '{}_trace.p'.format(samp_obj.label)),'wb'))

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
