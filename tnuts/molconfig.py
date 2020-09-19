import os
import copy
import numpy as np
import uuid
import subprocess
from arkane.common import symbol_by_number
from ape.common import get_electronic_energy
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
    internal = copy.deepcopy(samp_obj.torsion_internal)

    # Number of dihedrals
    n_rotors = samp_obj.n_rotors

    scans = [samp_obj.rotors_dict[n+1]['scan']\
            for n in range(n_rotors)]
    scan_indices = internal.B_indices[-n_rotors:]

    torsion_inds = [len(internal.B_indices) - n_rotors +\
            scan_indices.index([ind-1 for ind in scan]) for scan in scans]
    B = internal.B
    Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
    nrow = B.shape[0]

    # Displacement identity vector for each torsion
    qks = [np.zeros(nrow, dtype=int) for n in range(n_rotors)]
    for i,ind in enumerate(torsion_inds):
        qks[i][ind] = 1
        if internal.prim_coords[ind] > 0:
            qks[i] *= -1

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
        for qki,xi in zip(qks,x):
            geom += transform_geom(xi, internal, qki)
    return getXYZ(samp_obj.symbols, geom) 

def get_energy_at(x, samp_obj, n):
    print(x)
    x = np.atleast_1d(x)
    xyz = get_geometry_at(x, samp_obj)
    if not os.path.exists(samp_obj.output_directory):
        os.makedirs(samp_obj.output_directory)
    path = os.path.join(samp_obj.output_directory, 'nuts_out', samp_obj.label)
    file_name = '{}_{}'.format(n, uuid.uuid4().hex)
    n += 1
    if not os.path.exists(path):
        os.makedirs(path)
    if not samp_obj.is_QM_MM_INTERFACE:
        E = get_electronic_energy(xyz, path, file_name, 1, 
            charge=samp_obj.charge, multiplicity=samp_obj.spin_multiplicity,
            level_of_theory=samp_obj.level_of_theory, basis=samp_obj.basis,
            unrestricted=samp_obj.unrestricted) - samp_obj.e_elect
    else:
        E = get_electronic_energy(xyz, path, file_name, 1,
            charge=samp_obj.charge, multiplicity=samp_obj.spin_multiplicity,
            level_of_theory=samp_obj.level_of_theory, basis=samp_obj.basis,
            unrestricted=samp_obj.unrestricted,
            is_QM_MM_INTERFACE=samp_obj.is_QM_MM_INTERFACE,
            QM_USER_CONNECT=samp_obj.QM_USER_CONNECT,
            QM_ATOMS=samp_obj.QM_ATOMS,
            force_field_params=samp_obj.force_field_params,
            fixed_molecule_string=samp_obj.fixed_molecule_string,
            opt=samp_obj.opt,
            number_of_fixed_atoms=samp_obj.number_of_fixed_atoms) - samp_obj.e_elect
    subprocess.Popen(['rm {input_path}/{file_name}.q.out'.format(input_path=path, file_name=file_name)], shell=True)
    return E

if __name__ == '__main__':
    directory = '/Users/lancebettinson/Documents/entropy/um-vt/MeOOH'
    freq_file = os.path.join(directory,'MeOOH.out')
    label = 'MeOOH'
    from ape.sampling import SamplingJob
    samp_obj = SamplingJob(label,freq_file,output_directory=directory,
            protocol='TNUTS')
    samp_obj.parse()
    samp_obj.sampling()
    get_geometry_at([26*2*np.pi/360,11*2*np.pi/360], samp_obj)
