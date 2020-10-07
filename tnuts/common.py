# -*- coding: utf-8 -*-

"""
APE common module
"""

import os
import math
import copy
import logging
import numpy as np

import rmgpy.constants as constants

from arkane.statmech import is_linear

from tnuts.job.job import Job
from ape.qchem import QChemLog
from tnuts.qchem import load_gradient

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
    grad = load_gradient(QChemLog(output_file_path)) # cartesian, Hartree

    return grad
