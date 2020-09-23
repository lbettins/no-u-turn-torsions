#!/usr/bin/env python3
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
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')
    parser.add_argument('-ns', type=int, help='number of samples')
    parser.add_argument('-nc', type=int, help='number of chains')
    parser.add_argument('-nburn', type=int, help='number of tuning steps')
    parser.add_argument('-hpc', type=bool, help='if run on cluster')

    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]
    return args

def main():
    args = parse_command_line_arguments()
    input_file = args.file.split('/')[-1]
    ncpus = args.n
    protocol = args.p
    T = args.T
    nsamples = args.ns if args.ns is not None else 1000
    nchains = args.nc if args.nc is not None else 5
    nburn = args.nburn if args.nburn is not None else int(nsamples/5)
    hpc = args.hpc if args.hpc is not None else False
    if not protocol:
        protocol = 'TNUTS'
    if not T:
        T = 300
    project_directory = os.path.abspath(os.path.dirname(args.file))

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
    print(label)
    print(input_file)
    print(project_directory)
    samp_object = SamplingJob(
            input_file=os.path.join(project_directory,input_file),
            label=label, 
            ncpus=ncpus, output_directory=project_directory,
            protocol=protocol,
            level_of_theory='B97-D', basis='6-31G*', thresh=0.5)
    run_loglike(samp_object, T, nsamples=nsamples, nchains=nchains,
        tune=nburn, ncpus=ncpus, hpc=hpc)

if __name__ == '__main__':
    main()
