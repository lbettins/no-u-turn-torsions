import os
import copy
import numpy as np
import uuid
import subprocess
from arkane.common import symbol_by_number
from ape.common import get_electronic_energy as get_e_elect
from ape.InternalCoordinates import get_RedundantCoords, getXYZ
from tnuts.common import get_energy_gradient

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
    qk = np.zeros(nrow, dtype=int)
    for i,ind in enumerate(torsion_inds):
        qks[i][ind] = 1
        qk[ind] = 1
        if internal.prim_coords[ind] > 0:
            qks[i][ind] *= -1
            qk[ind] *= -1
    print(qk)
    
    # Initial geometry (at equilibrium position)
    geom = copy.deepcopy(samp_obj.cart_coords)

    def transform_geom(x, internal, qk):
        """
        Internal coordinate transformation needs to be
        incremental. It seems pi/3 is the limit, pi/2 breaks the bank.
        These transformations are done in intervals of pi/6, until the last transformation
        which is done in x%(pi/6).
        """
        if x > np.pi/6:
            return internal.transform_int_step((qk*np.pi/6).reshape(-1,))\
                    + transform_geom(x-np.pi/6, internal, qk)
        elif x < -np.pi/6:
            return internal.transform_int_step((-qk*np.pi/6).reshape(-1,))\
                    + transform_geom(x+np.pi/6, internal, qk)
        else:
            return internal.transform_int_step((qk*x).reshape(-1,))
    
    if np.isscalar(x):
        geom += transform_geom(x%(2*np.pi), internal, qks[0])
    else:
        for qki,xi in zip(qks,x):
            geom += transform_geom(xi%(2*np.pi), internal, qki)
    return getXYZ(samp_obj.symbols, internal.cart_coords), internal
    return getXYZ(samp_obj.symbols, geom) 

def get_energy_at(x, samp_obj, n):
    x = np.atleast_1d(x)
    xyz, internal = get_geometry_at(x, samp_obj)
    if not os.path.exists(samp_obj.output_directory):
        os.makedirs(samp_obj.output_directory)
    path = os.path.join(samp_obj.output_directory, 'nuts_out', samp_obj.label)
    file_name = '{}_{}'.format(n, uuid.uuid4().hex)
    if not os.path.exists(path):
        os.makedirs(path)
    if not samp_obj.is_QM_MM_INTERFACE:
        E = get_e_elect(xyz, path, file_name, samp_obj.ncpus, 
            charge=samp_obj.charge, multiplicity=samp_obj.spin_multiplicity,
            level_of_theory=samp_obj.level_of_theory, basis=samp_obj.basis,
            unrestricted=samp_obj.unrestricted) - samp_obj.e_elect
    else:
        E = get_e_elect(xyz, path, file_name, samp_obj.ncpus,
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
    subprocess.Popen(['rm {input_path}/{file_name}.q.out'.format(input_path=path,
        file_name=file_name)], shell=True)
    return E

def get_grad_at(x, samp_obj, n, 
        level_of_theory=None, basis=None,
        abseps=np.pi/32):
    if level_of_theory is None:
        level_of_theory = samp_obj.level_of_theory
    if basis is None:
        basis = samp_obj.basis

    x = np.atleast_1d(x)
    grads = np.zeros(len(x))
    scans = [samp_obj.rotors_dict[n+1]['scan']\
            for n in range(samp_obj.n_rotors)]
    scan_indices = samp_obj.torsion_internal.B_indices[-samp_obj.n_rotors:]
    torsion_inds = [len(samp_obj.torsion_internal.B_indices) - samp_obj.n_rotors +\
            scan_indices.index([ind-1 for ind in scan]) for scan in scans]
    signs = [-1 if samp_obj.torsion_internal.prim_coords[ind] > 0 else 1\
            for ind in torsion_inds]

    # set steps
    if isinstance(abseps, float):
        eps = abseps*np.ones(len(x))
    elif isinstance(abseps, (list, np.ndarray)):
        if len(abseps) != len(x):
            raise ValueError("Problem with input absolute step sizes")
        eps = np.array(abseps)
    else:
        raise RuntimeError("Absolute step sizes are not a recognised type!")

    # Bookkeeping
    if not os.path.exists(samp_obj.output_directory):
        os.makedirs(samp_obj.output_directory)
    path = os.path.join(samp_obj.output_directory, 'nuts_out', samp_obj.label)
    file_name = '{}_{}'.format(n, uuid.uuid4().hex)
    if not os.path.exists(path):
        os.makedirs(path)

    # Define args and kwargs for running jobs
    if not samp_obj.is_QM_MM_INTERFACE:
        kwargs = dict(charge=samp_obj.charge,
            multiplicity=samp_obj.spin_multiplicity,
            level_of_theory=level_of_theory, basis=basis,
            unrestricted=samp_obj.unrestricted)
    else:
        kwargs = dict(charge=samp_obj.charge,
            multiplicity=samp_obj.spin_multiplicity,
            level_of_theory=level_of_theory, basis=basis,
            unrestricted=samp_obj.unrestricted,
            is_QM_MM_INTERFACE=samp_obj.is_QM_MM_INTERFACE,
            QM_USER_CONNECT=samp_obj.QM_USER_CONNECT,
            QM_ATOMS=samp_obj.QM_ATOMS,
            force_field_params=samp_obj.force_field_params,
            fixed_molecule_string=samp_obj.fixed_molecule_string,
            opt=samp_obj.opt,
            number_of_fixed_atoms=samp_obj.number_of_fixed_atoms)
    args = (path, file_name, samp_obj.ncpus)

    xyz, internal = get_geometry_at(x, samp_obj)
    E,grad = get_energy_gradient(xyz,*args,**kwargs)
    B = internal.B_prim
    Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)

    grad = Bt_inv.dot(grad)[torsion_inds] 
    grad *= signs

    subprocess.Popen(['rm {input_path}/{file_name}.q.out'.format(input_path=path,
        file_name=file_name)], shell=True)

    return E,grad

if __name__ == '__main__':
    directory = '/Users/lancebettinson/Documents/entropy/um-vt/PROPIONIC_ACID'
    freq_file = os.path.join(directory,'propanoic.out')
    label = 'propanoic'
    from ape.sampling import SamplingJob
    samp_obj = SamplingJob(label,freq_file,output_directory=directory,
            protocol='TNUTS')
    samp_obj.parse()
    samp_obj.sampling()
    xyz = get_geometry_at([26*2*np.pi/360,11*2*np.pi/360,45*2*np.pi/360], samp_obj)
    print(xyz)
    
