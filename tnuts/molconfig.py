import os
import copy
import numpy as np
import rmgpy.constants as constants
from arkane.statmech import determine_rotor_symmetry
from ape.qchem import QChemLog
from arkane.common import symbol_by_number
from ape.job.job import Job
from ape.InternalCoordinates import get_RedundantCoords, getXYZ

def get_geometry_at(x, samp_obj):
    """
    x_i = [0, ... ]
    x   = [x0, x1, x2, ... ]

    Therefore:
    dx  = [x0, x1, x2, ... ] = x

    Finite steps taken as dx beginning from initial geom (x_i)
    x can be scalar, as long as dimensions match n_rotors
    """
    x = np.atleast_1d(x)

    # Create a copy of the internal object at x_i
    internal = copy.deepcopy(samp_obj.internal)

    # Number of dihedrals
    n_rotors = samp_obj.n_rotors

    scans = [samp_obj.rotors_dict[n+1]['scan']\
            for n in range(n_rotors)]

    scan_indices = internal.B_indices[-n_rotors:]
    torsion_inds = [len(internal.B_indices) - n_rotors +\
            scan_indices.index([ind-1 for ind in scan]) \
            for scan in scans]
    B = internal.B
    Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
    nrow = B.shsamp[0]

    # Displacement identity vector for each torsion
    qks = [np.zeros(nrow, dtype=int) for n in range(n_rotors)]
    for i,ind in enumerate(torsion_inds):
        qks[i][ind] = 1

    # Initial geometry (at equilibrium position)
    geom = copy.deepcopy(samp_obj.cart_coords)

    def transform_geom(x, internal, qk):
        """
        For reasons unknown to me, internal coordinate transformation needs to be
        incremental. It seems pi/3 is the limit, pi/2 breaks the bank.
        These transformations are done in intervals of pi/6, until the last transformation
        which is done in x%(pi/6).
        """
        if x > np.pi/6:
            return internal.transform_int_step((qk*np.pi/6).reshape(-1,))\
                    + transform_geom(x-np.pi/6, internal, qk)
        else:
            return internal.transform_int_step((qk*x).reshape(-1,))
    
    if np.isscalar(x):
        geom += transform_geom(x, internal, qks[0])
    else:
        for i,xi in enumerate(x):
            geom += transform_geom(xi, internal, qks[i])
    return getXYZ(samp_obj.symbols, geom) 

def get_energy_at(x, samp_obj, T):
    x = np.atleast_1d(x)
    beta = 1/(constants.kB*T)*constants.E_h
    #xyz = get_geometry_at(x, samp_obj)
    #TODO Insert code that submits QChem Job to get energy at x
    return result

if __name__ == '__main__':
    freq_file = '/Users/lancebettinson/Documents/entropy/um-vt/PROPANE/propane_fast.q.out'
    from samp.main import APE
    samp = APE(freq_file,protocol='MC')
    samp.parse()
    get_geometry_at([26*2*np.pi/360,11*2*np.pi/360], samp)
