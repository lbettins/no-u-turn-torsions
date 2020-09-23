#!/usr/bin/env python

import argparse
import os

jobscript="""
#!/bin/bash
#SBATCH --job-name={label}.job
#SBATCH --time=168:00:00
#SBATCH --partition=mhg
#SBATCH --account=mhg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ncpus}
#SBATCH --output={label}.o%j
#SBATCH --error={label}.o%j

#.bashrc will have $TNUTS aliased and activate needed environment
source $HOME/.bashrc
chmod +x $TNUTS/TNUTS.py
$TNUTS/./TNUTS.py {abs_path_to_freq} -T {T} -n {ncpus} -ns {nsamples} -nc {nchains} -nburn {nburn} -hpc 1
#rm -r $QCSCRATCH/nuts_output.o$SLURM_JOBID
"""

def parse_command_line_arguments(command_line_args=None):

    parser = argparse.ArgumentParser(description='Automated Property Estimator (APE)')
    parser.add_argument('file', metavar='FILE', type=str, nargs=1,
                        help='a file describing the job to execute')
    parser.add_argument('-n', type=int, help='number of CPUs to run quantum calculation')
    parser.add_argument('-i', type=str, help='the imaginary bonds for QMMM calculation')
    parser.add_argument('-T', type=int, help='Temperature in Kelvin')
    parser.add_argument('-ns', type=int, help='number of samples')
    parser.add_argument('-nc', type=int, help='number of chains')
    parser.add_argument('-nburn', type=int, help='number of tuning steps')
    parser.add_argument('-dry', type=bool, help='just generate jobscript')

    args = parser.parse_args(command_line_args)
    args = parser.parse_args()
    args.file = args.file[0]
    return args

def main():
    args = parse_command_line_arguments()
    T = args.T if args.T else 300
    nsamples = args.ns if args.ns is not None else 1000
    nchains = args.nc if args.nc is not None else 5
    nburn = args.nburn if args.nburn is not None else int(nsamples/5)

    input_file = args.file.split('/')[-1]
    project_directory = os.path.abspath(os.path.dirname(args.file))
    label = '{}_{}_{}'.format(input_file.split('.')[0],nchains,nsamples)
    abs_path_to_freq = os.path.join(project_directory, input_file)
    ncpus = args.n
    jobfile = os.path.join(project_directory, '{}.job'.format(label))
    with open(jobfile,'w') as f:
        f.write(jobscript.format(label=label,ncpus=ncpus,
            abs_path_to_freq=abs_path_to_freq,
            T=T,
            nsamples=nsamples, nchains=nchains,
            nburn=nburn))
        f.close()
    if not args.dry:
        run(jobfile)

def run(jobfile):
    os.system('sbatch {}'.format(jobfile))

if __name__ == '__main__':
    main()
