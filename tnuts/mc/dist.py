#!usr/bin/env python3
import pymc3 as pm
import numpy as np
from scipy.integrate import quad
import rmgpy.constants as constants
import theano.tensor as tt
from pymc3.distributions.dist_math import bound
#import pymc3_ext as pmx
#from pymc3_ext.distributions import transforms as tr

#class Angle(pm.Continuous):
#    """An angle constrained to be in the range -pi to pi
#    The actual sampling is performed in the two dimensional vector space
#    ``(sin(theta), cos(theta))`` so that the sampler doesn't see a
#    discontinuity at pi.
#    """
#    def __init__(self, *args, **kwargs):
#        transform = kwargs.pop("transform", None)
#        if transform is None:
#            if "regularized" in kwargs:
#                transform = tr.AngleTransform(
#                    regularized=kwargs.pop("regularized")
#                )
#            else:
#                transform = tr.angle
#        kwargs["transform"] = transform
#
#        shape = kwargs.get("shape", None)
#        if shape is None:
#            testval = 0.0
#        else:
#            testval = np.zeros(shape)
#        kwargs["testval"] = kwargs.pop("testval", testval)
#        super().__init__(*args, **kwargs)
#
#    def _random(self, size=None):
#        return np.random.uniform(-np.pi, np.pi, size)
#
#    def random(self, point=None, size=None):
#        return generate_samples(
#            self._random,
#            dist_shape=self.shape,
#            broadcast_shape=self.shape,
#            size=size,
#        )
#
#    def logp(self, value):
#        return tt.zeros_like(tt.as_tensor_variable(value))


class MyDist(pm.Continuous):
    def __init__(self, logp, modes, T, *args, **kwargs):
        super(MyDist, self).__init__(*args, **kwargs)
        self.fn = logp          # A Theano op
        self.modes = modes
        fns = [mode.get_spline_fn() for mode in self.modes]
        self.syms = np.array([mode.get_symmetry_number()\
                for mode in modes])
        beta = 1/(constants.kB*T)*constants.E_h
        fs = [lambda x: np.exp(-beta*fn(x)) for fn in fns]
        self.Zs = [quad(f, -np.pi/s, np.pi/s)[0]\
                for f,s in zip(fs,self.syms)]
        #self.cdfs = [lambda x: quad(\
        #        lambda phi: f(phi)/Z, -np.pi/s, x/s)[0]\
        #        for f,Z,s in zip(fs,Zs,sym_nums)]

    #def random(self, point=None, size=None):
    #    pi = np.random.rand(size)
    #    roots = fsolve(lambda xval: self.cdfs(xval) - pi, 0)[0]
    #    return roots

    def logp(self, value):
        logp = self.fn
        syms = self.syms
        return logp(value)
        return bound(logp(value), value>=-np.pi/syms, value<np.pi/syms)

class MyPeriodic(pm.Continuous):
    """An periodic parameter in a given range
    Like the :class:`Angle` distribution, the actual sampling is performed in
    a two dimensional vector space ``(sin(theta), cos(theta))`` and then
    transformed into the range ``[lower, upper)``.
    """
    def __init__(self, logp, modes, **kwargs):
        self.fn = logp          # A Theano op
        self.modes = modes
        self.syms = np.array([mode.get_symmetry_number()\
                for mode in modes])
        self.lower = np.array([-np.pi/s for s in self.syms])
        self.upper = np.array([np.pi/s for s in self.syms])

        lower = self.lower
        upper = self.upper
        transform = kwargs.pop("transform", None)
        if transform is None:
            transform = tr.PeriodicTransform(
                lower=lower,
                upper=upper,
                regularized=kwargs.pop("regularized", 10.0),
            )
        kwargs["transform"] = transform

        shape = kwargs.get("shape", None)
        if shape is None:
            testval = 0.5 * (lower + upper)
        else:
            testval = 0.5 * (lower + upper) + np.zeros(shape)
        kwargs["testval"] = kwargs.pop("testval", testval)
        super().__init__(**kwargs)

    def _random(self, size=None):
        return np.random.uniform(self.lower, self.upper, size)

    def random(self, point=None, size=None):
        return generate_samples(
            self._random,
            dist_shape=self.shape,
            broadcast_shape=self.shape,
            size=size,
        )

    def logp(self, value):
        logp = self.fn
        return logp(value)
