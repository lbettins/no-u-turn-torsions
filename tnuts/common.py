# -*- coding: utf-8 -*-

"""
APE common module
"""
import os
import math
import copy
import numpy as np
import rmgpy.constants as constants
from arkane.statmech import is_linear
from tnuts.job.job import Job
from ape.qchem import QChemLog
from tnuts.qchem import load_gradient

def evolve_dihedral_by(dq, internal, cart_rms_thresh=1e-15):
    remaining_int_step = dq
    prev_cart_coords = copy.deepcopy(internal.cart_coords)
    cur_cart_coords = internal.cart_coords.copy()
    cur_internals = internal.prim_coords
    target_internals = cur_internals + dq

    B_prim = internal.B_prim
    # Bt_inv may be overriden in other coordiante systems so we
    # calculate it 'manually' here.
    Bt_inv_prim = np.linalg.pinv(B_prim.dot(B_prim.T)).dot(B_prim)

    last_rms = 9999
    prev_internals = cur_internals
    internal.backtransform_failed = True
    internal.prev_cross = None
    nloop = 1000
    for i in range(nloop):
        cart_step = Bt_inv_prim.T.dot(remaining_int_step)
        if internal.nHcap is not None:
            cart_step[-(internal.nHcap*3):] = 0 # to creat the QMMM boundary in QMMM system # Shih-Cheng Li
        # Recalculate exact Bt_inv every cycle. Costly.
        cart_rms = np.sqrt(np.mean(cart_step**2))
        # Update cartesian coordinates
        cur_cart_coords += cart_step
        # Determine new internal coordinates
        new_internals = internal.update_internals(cur_cart_coords, prev_internals)
        remaining_int_step = target_internals - new_internals
        internal_rms = np.sqrt(np.mean(remaining_int_step**2))
        internal.log(f"Cycle {i}: rms(Δcart)={cart_rms:1.4e}, "
                 f"rms(Δinternal) = {internal_rms:1.5e}")

        # This assumes the first cart_rms won't be > 9999 ;)
        if (cart_rms < last_rms):
            # Store results of the conversion cycle for laster use, if
            # the internal-cartesian-transformation goes bad.
            best_cycle = (copy.deepcopy(cur_cart_coords), copy.deepcopy(new_internals.copy()))
            best_cycle_ind = i
            last_rms = cart_rms
            ratio = 1
        elif i != 0:
            cur_cart_coords, new_internals = best_cycle
            remaining_int_step = target_internals - new_internals
            # Reduce the moving step to avoid failing
            ratio *= 2
            remaining_int_step /= ratio
            if ratio > 16:
                break
            else:
                continue
        else:
            raise Exception("Internal-cartesian back-transformation already "
                            "failed in the first step. Aborting!")
        prev_internals = new_internals

        last_rms = cart_rms
        if cart_rms < cart_rms_thresh:
            internal.log("Internal to cartesian transformation converged!")
            internal.backtransform_failed = False
            break
        internal._prim_coords = np.array(new_internals)
    internal.log("")
    internal.cart_coords = cur_cart_coords
    return cur_cart_coords

def get_energy_gradient(xyz, path, file_name, ncpus, charge=None, multiplicity=None, level_of_theory=None, basis=None, unrestricted=None, \
        is_QM_MM_INTERFACE=None, QM_USER_CONNECT=None, QM_ATOMS=None, force_field_params=None, fixed_molecule_string=None, opt=None, number_of_fixed_atoms=None):
    #file_name = 'output'
    if is_QM_MM_INTERFACE:
        # Create geometry format of QM/MM system 
        # <Atom> <X> <Y> <Z> <MM atom type> <Bond 1> <Bond 2> <Bond 3> <Bond 4>
        # For example:
        # O 7.256000 1.298000 9.826000 -1  185  186  0 0
        # O 6.404000 1.114000 12.310000 -1  186  713  0 0
        # O 4.077000 1.069000 0.082000 -1  188  187  0 0
        # H 1.825000 1.405000 12.197000 -3  714  0  0 0
        # H 2.151000 1.129000 9.563000 -3  189  0  0 0
        # -----------------------------------
        QMMM_xyz_string = ''
        for i, xyz in enumerate(xyz.split('\n')):
            QMMM_xyz_string += " ".join([xyz, QM_USER_CONNECT[i]]) + '\n'
            if i == len(QM_ATOMS)-1:
                break
        QMMM_xyz_string += fixed_molecule_string
        job = Job(QMMM_xyz_string, path, file_name,jobtype='opt', ncpus=ncpus, charge=charge, multiplicity=multiplicity, \
            level_of_theory=level_of_theory, basis=basis, unrestricted=unrestricted, QM_atoms=QM_ATOMS, \
            force_field_params=force_field_params, opt=opt, number_of_fixed_atoms=number_of_fixed_atoms)
    else:
        job = Job(xyz, path, file_name,jobtype='opt', ncpus=ncpus, charge=charge, multiplicity=multiplicity, \
            level_of_theory=level_of_theory, basis=basis, unrestricted=unrestricted)
    
    # Write Q-Chem input file
    job.write_input_file()

    # Job submission
    job.submit()

    # Parse output file to get the calculated electronic energy
    output_file_path = os.path.join(path, '{}.q.out'.format(file_name))
    E,grad = load_gradient(QChemLog(output_file_path)) # cartesian, Hartree
    return E,grad

def log_trajectory(filename, x, E, DE):
    with open(filename,'a') as f:
        f.write('{}\t{}\t{}\n'.format(x,E,DE))
        f.close()
