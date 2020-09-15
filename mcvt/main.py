import arviz as az
import numpy as np
import pymc3 as pm
import rmgpy.constants as constants
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import theano.tensor as tt
from scipy.integrate import quad
from scipy import stats
from pymc3.distributions import Interpolated

from ape.MC.ops import Energy, LogPrior
from ape.MC.configuration import get_energy, get_energy_at
from ape.main import APE, dicts_to_NModes
from ape.data_parser import csv_to_NModes

import theano.tests.unittest_tools as utt

sns.set_context('notebook')
plt.style.use('seaborn-darkgrid')
print('Running on PyMC3 v{}'.format(pm.__version__))

def run_loglike(ape_obj,T):
    """
    """
    # Get the torsions from the APE object
    xyz_dict, energy_dict, mode_dict = ape_obj.sampling()
    modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                ape_obj=ape_obj)

    # Get the 1D Potential Energy functions
    energy_array = [mode.get_spline_fn() for mode in modes]
    # Get the symmetry numbers for each 1D torsion
    sym_nums = np.array([mode.get_symmetry_number() for mode in modes])
    # Get the number of torsions (in this case the dimensionality)
    n_d = ape_obj.n_rotors

    # Define the model and model parameters
    beta = 1/(constants.kB*T)*constants.E_h
    logpE = lambda E: -E    # E must be dimensionless
    logp = LogPrior(energy_array, T, sym_nums)
    energy_fn = Energy(get_energy, ape_obj)
    with pm.Model() as model:
        xi = pm.DensityDist('xi', logp, shape=n_d)
        x = pm.DensityDist('x', logp, shape=n_d, testval=xi)
        v = x
        #Etrial = pm.Deterministic('Etrial', beta*energy_fn(v))
        Etrial = pm.Deterministic('Etrial', -logp(v))
        Eprior = pm.Deterministic('Eprior', -logp(v))
        DeltaE = pm.Deterministic('DeltaE', Etrial-Eprior)
        E_obs = pm.DensityDist('E_obs', lambda E: logpE(E), observed={'E':DeltaE})
    with model:
        step = [pm.NUTS(x), pm.Metropolis(xi)]
        trace = pm.sample(100, tune=400, step=step, chains=20, cores=4,
               discard_tuned_samples=True)
        #ppc = pm.sample_posterior_predictive(trace, var_names=['x','xi','E_obs'])
    return trace

def plot_MC_torsion_result(trace_obj, ape_obj, T=300):
    beta = 1/(constants.kB*T)*constants.E_h
    NModes = ape_obj.NModes
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
    #ape_obj = APE('/Users/lancebettinson/Documents/entropy/um-vt/PROPANE/propane_fast.q.out', protocol="MC")
    #ape_obj.parse()
    T = 300
    #butane = APE('/Users/lancebettinson/Documents/entropy/um-vt/n-BUTANE/n-butane-umvt/n-BUTANE-freq.out', protocol="MC")
    #butane.parse()
    #diglyc = APE('/Users/lancebettinson/Documents/entropy/um-vt/DIGLYCINE/DIGLYCINE.out',
    #        protocol="MC")
    #diglyc.parse()
    #n_d = ape_obj.n_rotors
    #trace = run_loglike(ape_obj,T=T)
    #n_d = butane.n_rotors
    #trace = run_loglike(butane,T=T)
    #trace = run_loglike(diglyc,T=T)
    #idata = az.from_pymc3(trace, posterior_predictive=ppc)
    #az.plot_ppc(idata)
    etoh = APE('/Users/lancebettinson/Documents/entropy/um-vt/EtOH/EtOH.out',protocol='MC')
    etoh.parse()
    trace = run_loglike(etoh,T=T)
    plot_MC_torsion_result(trace, etoh, T=T)
    #plot_MC_torsion_result(trace, ape_obj,T=T)
    #plot_MC_torsion_result(trace, butane,T=T)
    #plot_MC_torsion_result(trace, diglyc, T=T)
    exit()
    if n_d >= 2:
        import corner
        #samples=np.vstack(trace['mod_x'])
        samples=np.vstack(trace['x'])
        fig = corner.corner(samples, plot_density=True, labels=["$x_{{{0}}}$".format(i) for i in range(1, n_d+1)])

        # Extract the axes
        axes = np.array(figure.axes).reshape((ndim, ndim))
        
        # Loop over the diagonal
        for i in range(ndim):
            ax = axes[i, i]
            ax.axvline(value1[i], color="g")
            ax.axvline(value2[i], color="r")
    plt.show()
