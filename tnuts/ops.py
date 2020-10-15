import warnings
import numpy as np
import theano
import theano.tensor as tt
import pymc3 as pm
import rmgpy.constants as constants
import copy

class LogPrior(tt.Op):
    itypes=[tt.dvector]
    otypes=[tt.dscalar]

    def __init__(self, fn_array, T, sym_n_array):
        self.beta = 1/(constants.kB*T)*constants.E_h
        self.Epriors = fn_array
        self.sym_ns = sym_n_array
        self.ndim = len(fn_array)
        self.dlogp = LogPriorGrad(self.Epriors, self.beta, self.ndim)

    def perform(self, node, inputs, outputs):
        theta, = inputs
        theta %= 2*np.pi/self.sym_ns
        result = 0
        for i in range(self.ndim):
            #theta[i] -= 2*np.pi/self.sym_ns[i] if theta[i] > np.pi/self.sym_ns[i] else 0
            #thetai = theta%(2*np.pi/self.sym_ns)
            result -= self.beta*self.Epriors[i](theta[i])
        outputs[0][0] = np.array(result)

    def grad(self, inputs, g):
        theta, = np.array(inputs)
        theta %= 2*np.pi/self.sym_ns
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
        #print("LogPrior grad:",theta, -1.0/self.beta*grads)
        outputs[0][0] = grads

class Energy(tt.Op):
    itypes = [tt.dvector] # expects a vector of parameter values when called
    otypes = [tt.dscalar] # outputs a single scalar value (the log likelihood)
    def __init__(self, fn, ape_obj, sym_n_array, grad_fn):
        self.get_e_elect = fn
        self.ape_obj = ape_obj
        self.sym_ns = sym_n_array
        self.n = JobN() # To keep track of jobs
        self.xcur = 0
        self.egrad = GetGrad(grad_fn, self.ape_obj, self.n)
 
    def perform(self, node, inputs, outputs):
        theta, = inputs  # this will contain my variables
        theta %= 2*np.pi/self.sym_ns
        #dx = theta - self.xcur
        #ape_obj = copy.deepcopy(self.ape_obj)
        result = self.get_e_elect(theta, self.ape_obj, n=self.n)
        self.n += 1
        self.xcur = theta
        #print('x', inputs, 'energy', result)
        outputs[0][0] = np.array(result) # output the log-likelihood

    def grad(self, inputs, g):
        theta, = inputs  # our parameter
        theta %= 2*np.pi/self.sym_ns
        return [g[0]*self.egrad(theta)]

class GetGrad(tt.Op):
    itypes = [tt.dvector]
    otypes = [tt.dvector]
    def __init__(self, fn, ape_obj, n):
        self.get_grad = fn
        self.ape_obj = ape_obj
        self.n = n

    def perform(self, node, inputs, outputs):
        theta, = inputs
        self.n += 1
        #ape_obj = copy.deepcopy(self.ape_obj)
        E,grad = self.get_grad(theta, self.ape_obj, n=self.n)
        #print("PosteriorGrad:",theta,grad)
        outputs[0][0] = grad

class JobN:
    def __init__(self, n=0):
        self.n = abs(n)
        self.sign = 1 if n >= 0 else -1
    def __add__(self, other):
        self.n += other # So + passes by reference
        return JobN(self.sign*self.n)
    def __sub__(self, other):
        self.n -= other
        return JobN(self.sign*self.n)
    def __str__(self):
        return '{}'.format(self.sign*self.n)
    def __repr__(self):
        return '%i' % (self.sign*self.n)

#class EnergyGrad(tt.Op):
#    itypes = [tt.dvector]
#    otypes = [tt.dvector]
#
#    def __init__(self, fn, ape_obj):
#        self.get_e_elect = fn
#        self.ape_obj = ape_obj
#        self.n = JobN(n=-1)
#
#    def perform(self, node, inputs, outputs):
#        theta, = inputs
#        def lnlike(values):
#            return self.get_e_elect(values, self.ape_obj, n=self.n)
#        self.n -= 1
#        grads = gradients(theta, lnlike, abseps=np.pi/32)
#        outputs[0][0] = grads
