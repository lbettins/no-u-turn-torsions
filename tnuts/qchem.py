#!/usr/bin/env python3

"""
TNUTS QChem module
Used to parse QChem output files when APE won't do the job
"""

import numpy as np
import math

def load_gradient(QChemLog):
    natoms = 0
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
    return grad

if __name__ == "__main__":
    from ape.qchem import QChemLog
    log = QChemLog("/Users/lancebettinson/Documents/entropy/um-vt/MeOOH/nuts_out/gradfile.q.out")
    print(log.path)
    print(log.load_energy())
    print(load_gradient(log))
