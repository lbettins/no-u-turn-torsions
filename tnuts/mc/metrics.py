#!/usr/bin/env python3
#####################################################
#                                                   #
# SUPPORT:  https://dfm.io/posts/pymc3-mass-matrix/ #
#           Stan Manual 34.2                        #
#                                                   #
#####################################################
import pymc3 as pm
import numpy as np
from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull

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

def get_step_for_trace(trace=None, model=None, covi=None,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    model = pm.modelcontext(model)

    # If not given, use the uncoupled metric;
    # if that not given, use trivial metric (identity mass matrix)
    if trace is None:
        if covi is None:
            potential = QuadPotentialFull(np.eye(model.ndim))
            return pm.NUTS(potential=potential, **kwargs)
        else:
            return pm.NUTS(scaling=covi, is_cov=True, **kwargs)

    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in trace._straces.values():
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1

    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)

    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)

    # Use the sample covariance as the inverse metric
    potential = QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)
