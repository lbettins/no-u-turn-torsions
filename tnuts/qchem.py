#!/usr/bin/env python3

"""
TNUTS QChem module
Used to parse QChem output files when APE won't do the job
"""

import numpy as np
import math

def get_level_of_theory(QChemLog):
    lvl_of_theory = None
    basis = None
    lot_kwargs = {}
    with open(QChemLog.path, 'r') as f:
        line = f.readline()
        while line != '':
            if 'basis' in line.lower():
                if line.lower().split()[0] == 'basis':
                    basis = line.split()[-1]
            elif 'ecp' in line.lower():
                if line.lower().split()[0] == 'ecp':
                    #basis += "\necp\tdef2ecp"
                    pass
            elif 'method' in line.lower() or 'exchange' in line.lower() and len(line.split()) == 2:
                lvl_of_theory = line.split()[-1]
                if 'omega' in lvl_of_theory.lower():
                    lvl_of_theory = lvl_of_theory.replace('omega','w')
            if basis and lvl_of_theory:
                lot_kwargs['basis'] = basis
                lot_kwargs['level_of_theory'] = lvl_of_theory
                return lot_kwargs
            line = f.readline()
        f.close()
        lot_kwargs['basis'] = basis
        lot_kwargs['level_of_theory'] = lvl_of_theory
        return charge, spin_multiplicity, basis, lvl_of_theory

def load_gradient(QChemLog):
    natoms = 0
    E = None
    grad = None
    with open(QChemLog.path, 'r') as f:
        line = f.readline()
        while line != '':
            # Automatically determine the number of atoms
            if 'Standard Nuclear Orientation' in line and natoms == 0:
                for i in range(3):
                    line = f.readline()
                while '----------------------------------------------------' not in line:
                    natoms += 1
                    line = f.readline()
            if 'Total energy in the final basis set' in line:
                E = float(line.split()[-1])
            # Read gradient
            if 'Gradient of SCF Energy' in line:
                grad = np.zeros(3*natoms, np.float64)
                for i in range(int(math.ceil(natoms / 6.0))):
                    # Header row
                    line = f.readline()
                    # grad element x,y,z rows
                    for j in range(3):
                        data = f.readline().split()[1:]
                        for k in range(len(data)):
                            grad[i*6*3 + 3*k + j] = float(data[k])
                # Convert from atomic units (Hartree/Bohr_radius^2) to J/m
                #grad *= 4.35974417e-18 / 5.291772108e-11
            line = f.readline()
        f.close()
    #if grad is None:
    #    grad = np.zeros(3*natoms, np.float64)
    return E,grad

if __name__ == "__main__":
    from ape.qchem import QChemLog
    log = QChemLog("/Users/lancebettinson/Documents/entropy/um-vt/MeOOH/nuts_out/gradfile.q.out")
    print(log.path)
    print(log.load_energy())
    print(load_gradient(log))
