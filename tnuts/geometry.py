import os
import copy
import uuid
import numpy as np
import subprocess
import time
from ape.common import get_electronic_energy as get_e_elect
from ape.InternalCoordinates import getXYZ
from rmgpy.statmech import Conformer
import rmgpy.constants as constants
from tnuts.common import evolve_dihedral_by, get_energy_gradient, log_trajectory
from tnuts.mode import dicts_to_NModes

class Geometry:
    def __init__(self, samp_obj, internal, syms):
        self.count = 0

        self.samp_obj = samp_obj
        self.n_rotors = samp_obj.n_rotors

        self.internal0 = copy.deepcopy(samp_obj.torsion_internal)
        self.geom0 = copy.deepcopy(self.internal0.cart_coords)

        self.prev_internal = copy.deepcopy(self.internal0)
        self.internal = internal
        self.geom = self.internal.cart_coords

        self.conformer = copy.deepcopy(samp_obj.conformer)

        # Current configuration
        self.xcur = np.zeros(self.n_rotors)

        # Output energy/grad file
        self.dict = {}

        # Number of dihedrals and reference (equilibrium) dihedrals
        self.scans = [np.array(samp_obj.rotors_dict[n+1]['scan'])\
                for n in range(self.n_rotors)]
        self.pivots = [np.array(samp_obj.rotors_dict[n+1]['pivots'])\
                for n in range(self.n_rotors)]
        self.tops = [np.array(samp_obj.rotors_dict[n+1]['top'])\
                for n in range(self.n_rotors)]
        scan_indices = internal.B_indices[-self.n_rotors:]
        self.dihedrals0 = np.array([self.internal.calc_dihedral(self.internal0.c3d, self.scans[i]-1) for i in range(self.n_rotors)])

        # Indices of dihedrals in B matrix
        self.torsion_inds = [len(internal.B_indices) - self.n_rotors +\
                scan_indices.index([ind-1 for ind in scan]) for scan in self.scans]
        B = internal.B
        Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
        nrow = B.shape[0]
        # Displacement identity vector for all torsions
        self.qk = np.zeros(nrow, dtype=int)
        self.signs = np.ones(self.n_rotors)
        for i,ind in enumerate(self.torsion_inds):
            self.qk[ind] = 1
            if internal.prim_coords[ind] > 0:
                self.qk[ind] *= -1
                self.signs[i] *= -1
        print("Qks are",self.qk)
        self.symmetry_numbers = np.atleast_1d(syms)
        self.L = 2*np.pi/self.symmetry_numbers
        self.center_mod = lambda x: \
                ((x.transpose() % self.L) \
                - self.L*((x.transpose() % self.L) // ((self.L)/2))).transpose()

        # Bookkeeping
        if not os.path.exists(samp_obj.output_directory):
            os.makedirs(samp_obj.output_directory)
        self.path = os.path.join(samp_obj.output_directory, 'nuts_out', samp_obj.label)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        n = 0
        traj_log = os.path.join(samp_obj.output_directory, 'nuts_out', '{}_trajectories_{}.txt')
        while os.path.exists(traj_log.format(samp_obj.label, n)):
            n += 1
        self.traj_log = traj_log.format(samp_obj.label, n)

        # Timing
        self.qchem_runtime = 0.
        self.coord_runtime = 0.

        # Define args and kwargs for running jobs
        if not samp_obj.is_QM_MM_INTERFACE:
            self.kwargs = dict(charge=samp_obj.charge,
                multiplicity=samp_obj.spin_multiplicity,
                level_of_theory=samp_obj.level_of_theory, basis=samp_obj.basis,
                unrestricted=samp_obj.unrestricted)
        else:
            self.kwargs = dict(charge=samp_obj.charge,
                multiplicity=samp_obj.spin_multiplicity,
                level_of_theory=samp_obj.level_of_theory, basis=samp_obj.basis,
                unrestricted=samp_obj.unrestricted,
                is_QM_MM_INTERFACE=samp_obj.is_QM_MM_INTERFACE,
                QM_USER_CONNECT=samp_obj.QM_USER_CONNECT,
                QM_ATOMS=samp_obj.QM_ATOMS,
                force_field_params=samp_obj.force_field_params,
                fixed_molecule_string=samp_obj.fixed_molecule_string,
                opt=samp_obj.opt,
                number_of_fixed_atoms=samp_obj.number_of_fixed_atoms)

    def reset(self):
        self.internal = copy.deepcopy(self.internal0)
        self.geom = self.internal.cart_coords
        self.xcur = np.zeros(self.n_rotors)

    def hard_reset(self):
        self.reset()
        self.clear_dict()

    def clear_dict(self):
        self.dict = {}

    def check_dihedral(self, target):
        dihedrals = np.array([self.internal.calc_dihedral(self.internal.c3d, self.scans[i]-1) for i in range(self.n_rotors)])
        
        net = self.center_mod(dihedrals - self.dihedrals0)
        self.xcur = net
        err = -self.signs*np.around(net - target, 5)
        return net, err

    def transform_geometry_to(self, x,
            incr_step=np.pi/12, err_thresh=1e-5, cart_rms_thresh=1E-9,
            prev_fail=False):
        step = np.zeros(len(self.qk))
        tic = time.perf_counter()
        cur_dihedral, error = self.check_dihedral(x)
        MAXCOUNT = 3*(np.pi // incr_step)
        count = 0
        while (np.abs(error) > err_thresh).any():
            for err,ind in zip(error,self.torsion_inds):
                step[ind] = self.qk[ind]*min(np.abs(err), np.abs(incr_step))*np.sign(err)
            try:
                self.internal.transform_int_step(step, cart_rms_thresh=cart_rms_thresh)
            except:
                self.hard_reset()
                if prev_fail:
                    return getXYZ(self.samp_obj.symbols, self.prev_internal.cart_coords)
                return self.transform_geometry_to(x,prev_fail=True)
            cur_dihedral, error = self.check_dihedral(x)
            count += 1
            if count > MAXCOUNT:    # not ideal
                print("ERROR in convergence; max n iterations reached")
                print("Current position:",self.xcur)
                print(getXYZ(self.samp_obj.symbols, self.prev_internal.cart_coords))
                return getXYZ(self.samp_obj.symbols, self.prev_internal.cart_coords)
        toc = time.perf_counter()
        #print("CONVERGED", 'Time:', toc-tic, "seconds with", count, "iterations.")
        self.prev_internal = copy.deepcopy(self.internal)
        return getXYZ(self.samp_obj.symbols, self.internal.cart_coords)

    def get_energy_grad_at(self, x, which="energy"):
        """
        Returns E, gradient. Stores both values to dict with positional key.
        """
        #roundx = 2*np.around(x/2,2)
        try:
        # IF the gradient or energy has already been calculated for the position,
        # find the desired value and return it immediately
            egrad = self.dict[tuple(x)]
            if which == "energy":
                log_trajectory(self.traj_log,x,egrad[0],egrad[1])
                return egrad[0]
            else:
                self.clear_dict()
                return egrad[1]
        # OTHERWISE we need to calculate it below
        except KeyError:
            pass
         
        #Get the energy/gradient from QChem
        tic = time.perf_counter()
        prev_xyz = getXYZ(self.samp_obj.symbols, self.internal.cart_coords)
        prev_x = tuple(self.xcur)
        #xyz = self.get_geometry_at(tuple(roundx))
        #xyz = self.get_geometry_at(x)
        xyz = self.transform_geometry_to(x)
        toc = time.perf_counter()
        self.coord_runtime += toc-tic

        file_name = '{}_{}'.format(self.count, uuid.uuid4().hex)
        args = (self.path, file_name, self.samp_obj.ncpus)

        # Get and time energy/gradient
        tic = time.perf_counter()
        E,grad = get_energy_gradient(xyz,*args,**self.kwargs)
        toc = time.perf_counter()
        self.qchem_runtime += toc-tic

        # if the job succeeds, proceed normally
        if grad is not None and E is not None:
            E -= self.samp_obj.e_elect  # All in Hartree
            if E > 1.:
                print("Energy is:", E)
                print("Errors in coordinate transformations:")
                print("Previous:", prev_x, "\n",
                    prev_xyz, "\n"
                    "Current:", tuple(x), "\n",
                    xyz)
                self.hard_reset()
                return self.get_energy_grad_at(x,which=which)

            B = self.internal.B_prim
            Bt_inv = np.linalg.pinv(B.dot(B.T)).dot(B)
    
            grad = Bt_inv.dot(grad)[self.torsion_inds]
            #signs = np.ones(self.n_rotors)
            #for i,ind in enumerate(self.torsion_inds):
            #    if self.internal.prim_coords[ind] > 0:
            #        signs[i] *= -1
            #if not (self.signs == signs).all():
            #    print(self.signs,signs)
            #grad *= signs

            #grad *= self.signs  # In Hartree per rad
            #grad *= -1
            #E += ((x-roundx)*grad).sum()
            log_trajectory(self.traj_log,x,E,grad)
            subprocess.Popen(['rm {input_path}/{file_name}.q.out'.format(input_path=self.path,
                file_name=file_name)], shell=True)
            self.Ecur = E
            self.gradcur = grad
            # Set the energy/gradient in memory and return
            self.dict[tuple(x)] = (E, grad)
        # or else reset internals and try again
        else:
            if E is not None:
                print("Energy is", E)
            print("Errors in coordinate transformations:")
            print("Previous:", prev_x, "\n",
                    prev_xyz, "\n"
                    "Current:", tuple(x), "\n",
                    xyz)
            self.hard_reset()
            return self.get_energy_grad_at(x,which=which)
            
        self.count += 1
        if (self.count % 100) == 0:
            self.reset()
        elif (self.count % 200) == 0:
            self.hard_reset()
        if which == "energy":
            return E
        else:
            return grad

    def calc_I(self, phi):
        xyz = self.transform_geometry_to(phi)
        coordinates = self.internal.c3d
        self.conformer.coordinates = (coordinates, "angstroms")
        I = []
        for i in range(self.n_rotors):
            I.append(
                    self.conformer.get_internal_reduced_moment_of_inertia(
                        self.pivots[i], self.tops[i])*constants.Na * 1e23) # amu*Å^2
        return np.array(I)

if __name__ == '__main__':
    directory = '/Users/lancebettinson/Documents/entropy/um-vt/MeOOH'
    freq_file = os.path.join(directory,'MeOOH.out')
    label = 'MeOOH'
    from ape.sampling import SamplingJob
    samp_obj = SamplingJob(label,freq_file,output_directory=directory,
            protocol='TNUTS')
    samp_obj.parse()
    samp_obj.csv_path = os.path.join(directory,
            'MeOOH_sampling_result.csv')
    xyz_dict, energy_dict, mode_dict = samp_obj.sampling()
    tmodes = dicts_to_NModes(mode_dict, energy_dict, xyz_dict,
                samp_obj=samp_obj, just_tors=True)
    syms = np.array([mode.get_symmetry_number() for mode in tmodes])
    geom = Geometry(samp_obj, samp_obj.torsion_internal, syms)
    
    x = np.random.random((10,2))
    for xi in x:
        print("coordinate transformation at",xi)
        I = geom.calc_I(xi)
        print(I)
