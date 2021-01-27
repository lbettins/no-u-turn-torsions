#!/usr/bin/env python3

"""
Thermodynamic Calculations Following Monte Carlo Simulations for Coupled Modes
"""
import argparse
import os
import dill as pickle
import numpy as np
from tnuts.thermo.thermo import MCThermoJob as MCThermo
from ape.qchem import QChemLog

def parse_command_line_arguments(command_line_args=None):
    
    parser = argparse.ArgumentParser(description='Thermo for TNUTS')
    parser.add_argument('label', metavar='FILE', type=str, nargs=1,
                        help='the label describing the sampling job')
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')

    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.label = args.label[0]
    return args

def get_files(directory, label=None):
    """
    Get sample file given the desired label
    """
    if label is None:
        pfiles = [f for f in os.listdir(directory)\
                if '.p' in f]
    else:
        pfiles = [f for f in os.listdir(directory)\
                if label in f and '.p' in f]
    return pfiles

def unpickle(abs_path_to_pickle):
    with open(abs_path_to_pickle, 'rb') as f:
        model_dict = pickle.load(f)
    return model_dict

def get_jobs(directory, label, T):
    files = get_files(directory, label)
    jobs = []
    for f in files:
        pkl_args = unpickle(os.path.join(directory, f))
        samp_obj = pkl_args['samp_obj']
        samp_obj.project_directory = os.path.expandvars('$SCRATCH')
        samp_obj.output_directory = os.path.expandvars('$SCRATCH')
        samp_obj.path = os.path.expandvars('$SCRATCH')
        #samp_obj.input_file = \
        #        os.path.join(samp_obj.project_directory,
        #            's13_b97d_def2.out')
        samp_obj.input_file = \
                os.path.join(samp_obj.project_directory,
                        's17.out')
        for t in T:
            thermo_args = (pkl_args['trace'], t)
            thermo_kwargs = {'samp_obj' : pkl_args['samp_obj'],\
                    'model' : pkl_args['model'],
                    'sampT' : pkl_args['T']}
            jobs.append(MCThermo(*thermo_args, **thermo_kwargs))
    print(jobs)
    return jobs

def execute_(jobs):
    for job in jobs:
        print("Running job", job)
        job.execute()

def main():
    curdir = os.path.abspath(os.path.curdir)
    args = parse_command_line_arguments()
    label = args.label.split('/')[-1]
    T = np.atleast_1d(args.T) if args.T is not None\
            else np.linspace(100,2000,39)
    jobs = get_jobs(curdir, label, T)
    execute_(jobs)

    # SCRATCH directory must be set before script runs
    # This is done for ease of transferability to different hardware setups
    output_directory = os.path.expandvars('$SCRATCH')

if __name__ == '__main__':
    main()
