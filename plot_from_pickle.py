#!/usr/bin/env python3
import arviz as az
import argparse
import dill as pickle
import numpy as np
import pymc3 as pm
import rmgpy.constants as constants
import os
import warnings
warnings.filterwarnings("ignore")
import theano
import theano.tensor as tt
from scipy.integrate import quad
from scipy import stats
from pymc3.distributions import Interpolated
from tnuts.ops import Energy
from tnuts.molconfig import get_energy_at, get_grad_at
from tnuts.mode import dicts_to_NModes
from tnuts.geometry import Geometry
from ape.sampling import SamplingJob
from tnuts.main import plot_MC_torsion_result

print('Running on PyMC3 v{}'.format(pm.__version__))

def parse_command_line_arguments(command_line_args=None):

    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')
    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]
    return args

def main():
    args = parse_command_line_arguments()
    pkl = args.file.split('/')[-1]
    project_directory = os.path.abspath(os.path.dirname(args.file))
    with open(os.path.join(project_directory,pkl), 'rb') as pklfile:
        pkl_dict = pickle.load(pklfile)
    samp_obj = SamplingJob(
            input_file='/Users/lancebettinson/Documents/entropy/um-vt/EtOH/EtOH_hf.out',
            label='EtOH_hf',
            ncpus=4, output_directory=os.path.expandvars('$SCRATCH'),
            protocol='TNUTS',
            level_of_theory='HF', basis='sto-3g', thresh=0.5)
    print(pkl_dict)
    pkl_dict['T'] = 400
    LogP, Z, modes = generate_umvt_logprior(samp_obj,pkl_dict['T'])
    print(Z,'versus',pkl_dict['trace']['a'])
    plot_MC_torsion_result(pkl_dict['trace'],modes,pkl_dict['T'])
    

def generate_umvt_logprior(samp_obj, T):
    from tnuts.ops import LogPrior
    # Get the torsions from the APE object
    samp_obj.parse()
    # With APE updates, should edit APE sampling.py to [only] sample torsions
    xyz_dict, energy_dict, mode_dict = samp_obj.sampling()
    modes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=samp_obj)
    # Get the 1D Potential Energy functions
    energy_array = [mode.get_spline_fn() for mode in modes]
    # Get the symmetry numbers for each 1D torsion
    sym_nums = np.array([mode.get_symmetry_number() for mode in modes])
    Z = np.array([quad(lambda x: fn(x), 0, 2*np.pi/s)[0]\
            for fn,s in zip(energy_array,sym_nums)]).sum()

    # Return theano op LogPrior and partition function
    return LogPrior(energy_array, T, sym_nums), Z, modes
    
if __name__=='__main__':
    main()
