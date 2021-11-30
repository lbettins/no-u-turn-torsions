#!usr/bin/env python3
import warnings
import os
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import rmgpy.constants as constants
import copy
from tnuts.common import log_trajectory
from scipy.integrate import quad

class LogPrior(tt.Op):
    itypes=[tt.dvector]
    otypes=[tt.dscalar]
    def __init__(self, fn_array, T, sym_n_array):
        self.beta = 1/(constants.kB*T)*constants.E_h
        self.Epriors = fn_array
        self.sym_ns = sym_n_array
        self.logZs = [np.log(quad(lambda phi:\
                np.exp(-self.beta*fn(phi)), -np.pi/s, np.pi/s)[0])\
                for fn,s in zip(self.Epriors, self.sym_ns)]
        self.ndim = len(fn_array)
        self.dlogp = LogPriorGrad(self.Epriors, self.beta, self.ndim)
        self.count = 0
        self.L = 2*np.pi/self.sym_ns
        self.center_mod = lambda x: \
                ((x.transpose() % self.L) \
                - self.L*((x.transpose() % self.L) // ((self.L)/2))).transpose()
        output_dir = os.path.expandvars("$SCRATCH")
        traj_log = os.path.join(output_dir, 'nuts_out', 'logprior')
        self.traj_log = traj_log

    def perform(self, node, inputs, outputs):
        theta, = inputs
        theta = self.center_mod(theta)
        result = 0
        grads = np.zeros(self.ndim)
        for i in range(self.ndim):
            result -= self.beta*self.Epriors[i](theta[i])
            #        + self.logZs[i]
            grads[i] = -self.beta*self.Epriors[i](theta[i], 1)
        self.count += 1
        #log_trajectory(self.traj_log, theta, result, grads)
        outputs[0][0] = np.array(result)

    def grad(self, inputs, g):
        theta, = np.array(inputs)
        theta = self.center_mod(theta)
        return [g[0] * self.dlogp(theta)]

class LogPriorGrad(tt.Op):
    itypes=[tt.dvector]
    otypes=[tt.dvector]
    def __init__(self, prior_fns, beta, ndim):
        self.fns = prior_fns
        self.ndim = ndim
        self.beta = beta

    def perform(self, node, inputs, outputs):
        theta, = inputs
        grads = np.zeros(self.ndim)
        for i in range(self.ndim):
            grads[i] = -self.beta*self.fns[i](theta[i], 1)
        outputs[0][0] = grads

class Energy(tt.Op):
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    def __init__(self, geom, beta):
        self.geom = geom
        self.beta = beta
        self.get_e_elect = self.geom.get_energy_grad_at
        self.egrad = GetGrad(self.geom, self.beta)
        self.L = 2*np.pi/self.geom.symmetry_numbers
        self.center_mod = lambda x: \
                ((x.transpose() % self.L) \
                - self.L*((x.transpose() % self.L) // ((self.L)/2))).transpose()

    def perform(self, node, inputs, outputs):
        theta, = inputs  # this will contain my variables
        theta = self.center_mod(theta)
        result = self.get_e_elect(theta, which="energy")
        outputs[0][0] = np.array(self.beta*result) # output the log-likelihood

    def grad(self, inputs, g):
        theta, = inputs  # our parameter
        theta = self.center_mod(theta)
        grad = self.egrad(theta)
        return [g[0]*grad]

class GetGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]
    def __init__(self, geom, beta):
        self.geom = geom
        self.beta = beta
        self.get_grad = self.geom.get_energy_grad_at

    def perform(self, node, inputs, outputs):
        theta, = inputs
        #grad = self.get_grad(theta, self.ape_obj, n=self.n)
        grad = self.get_grad(theta, which="grad")
        outputs[0][0] = self.beta*grad

class CalcI(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]
    def __init__(self, geom):
        self.geom = geom
        self.calc_I = self.geom.calc_I

    def perform(self, node, inputs, outputs):
        theta, = inputs
        #grad = self.get_grad(theta, self.ape_obj, n=self.n)
        I = self.calc_I(theta)
        outputs[0][0] = I


