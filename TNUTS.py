# -*- coding: utf-8 -*-

"""
Monte Carlo Simulations for Coupled Modes
"""

import argparse
import os
from ape.sampling import SamplingJob
from tnuts.main import run_loglike

def parse_command_line_arguments(command_line_args=None):
    
    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')
    parser.add_argument('-n', type=int, help='number of CPUs to run quantum calculation')
    parser.add_argument('-p', type=str, help='the sampling protocol (default: TNUTS)')
    parser.add_argument('-i', type=str, help='the imaginary bonds for QMMM calculation')
    parser.add_argument('-mode', type=int, help='which modes (in a list) the sampling protocol will treat (default: all)')
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')

    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]

    return args

def main():
    """ The main APE executable function"""
    args = parse_command_line_arguments()
    input_file = args.file
    ncpus = args.n
    protocol = args.p
    T = args.T
    project_directory = os.path.abspath(os.path.dirname(args.file))
    if not protocol:
        protocol = 'TNUTS'
        print('This calculation will use TNUTS as sampling protocol')
    elif protocol == 'UMN' or protocol == 'UMVT' or protocol == 'TNUTS':
        print('This calculation will use {} as sampling protocol'.format(protocol))
    if not T:
        T = 300

    # imaginary bonds for QMMM calculation
    # atom indices starts from 1
    imaginary_bonds = args.i
    if args.i is not None:
        imaginary_bonds_string = imaginary_bonds.strip('[').strip(']')
        imaginary_bonds = []
        for bond in imaginary_bonds_string.split(','):
            atom1, atom2 = bond.split('-')
            imaginary_bonds.append([int(atom1), int(atom2)])

    label = input_file.split('.')[0]
    samp_object = SamplingJob(input_file=input_file, label=label, 
            ncpus=ncpus, output_directory=project_directory,
            protocol=protocol,
            level_of_theory='B97-D', basis='6-31G*', thresh=0.5)
    tnuts_trace = run_loglike(samp_object, T)

if __name__ == '__main__':
    main()
