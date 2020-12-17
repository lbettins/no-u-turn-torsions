from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema
import os
import scipy.integrate as integrate
import numpy as np
import rmgpy.constants as constants
import copy

"""
The class NMode organizes key values and parameters for each normal mode under UM-VT.
Values are populated by reading the UM-VT output file.
Both sampling q (x) and PES (v or y) values are included.
Spline populates the continuous PES x and y values and returns them.
"""
class NMode:
    def __init__(self, n=0, v=0.0, x=0.0, sample_geoms=None, 
            tors=False, scan=None, v_ho=None, v_um=None, zpe_um=None,
            I=None, mu=None, k=None, min_elec=None, sigma=None, eig=None, 
            eigv=None, H=None):
        self.n = n  #Mode #
        if isinstance(v, int):
            self.v_sample = [v]  # Equilibrium position is referenced at 0
            self.x_sample = [x]  # ^
            self.sample_geometries = [sample_geoms]  # ^^These are parallel!
            self.step_size = self.x_sample[1]-self.x_sample[0]
        else:
            self.v_sample = v
            self.x_sample = x
            self.sample_geometries = sample_geoms
            self.step_size = None
        self.spline_fn = None
        self.x_interp = None  # Splined continuous PES.
        self.y_interp = None  # ^These are parallel!
        self.n_wells = None
        self.wells = []  # Keep track of wells, even for sb (length = 1)!
        self.local_minima = []
        self.local_maxima = []
        self.isTors = tors  # Modes are initialized as sb by default
        self.scan = scan    # dihedral angle atom indices
        self.v_ho = v_ho #in cm-1
        self.v_um = v_um #in cm-1
        self.min_elec = min_elec #in Hartree/particle
        self.sigma = sigma #symmetry number for torsion
        self.eig = None
        self.eigv = None
        self.H = None
        if v_ho:
            v_ho = np.array(v_ho)
            self.zpe_ho = constants.h*(v_ho*constants.c*100)*\
                            constants.Na/2.0/4184 #in kcal/mol
        else:
            self.zpe_ho = None
        self.zpe_um = zpe_um
        self.I = I      #[amu*angstrom^2]
        self.mu = mu    #[amu]
        self.k = k      #[1/s^2]
        if self.I and self.mu:
            raise ValueError("I and mu exist simultaneously!")
        self.z = None   #partition function contribution for this mode

    def get_classical_partition_fn(self,T):
        if not self.sigma or not self.v_sample:
            raise ValueError("No symmetry number or energy samples")
        if not self.isTors:
            raise ValueError("Can only calculate for torsions")
        beta = 1/(constants.kB*T)*constants.E_h
        fn = mode.get_spline_fn()
        s = mode.get_symmetry_number()
        self.z = quad(lambda x: np.exp(-beta*fn(x)), -np.pi/s, np.pi/s)[0]
        return self.z

    def set_mode_number(self,n):
        if self.n and type(self.n) is int:
            self.n = [self.n].append(n)
        elif self.n and type(self.n) is list:
            if n not in self.n:
                self.n.append(n)
            else:
                raise ValueError("NMode already indexed at",n)
        else:
            self.n = n

    def get_mode_number(self):
        """
        Return mode number
        """
        return self.n

    def set_H(self, H):
        self.H = H

    def get_H(self):
        return self.H

    def get_eigs(self):
        return np.linalg.eigh(self.H)

    def get_I(self):
        """
        Return reduced moment of inertia [amu*angstrom^2]
        """
        if not self.is_tors():
            raise TypeError("Mode",self.n,"is not torsional!")
        elif self.I:
            return self.I
        else:
            raise ValueError("Moment of inertia undefined for mode",self.n)

    def get_mu(self):
        """
        Return reduced mass [amu]
        """
        if self.is_tors():
            raise TypeError("Mode",self.n,"is torsional!")
        elif self.mu:
            return self.mu
        else:
            raise ValueError("Reduced mass undefined for mode",self.n)

    def get_k(self):
        """
        Return force constant [1/s^2]
        """
        if self.k:
            return self.k
        elif self.is_tors():
            raise ValueError("Torsion constant undefined for mode",self.n)
        else:
            raise ValueError("Spring constant undefined for mode",self.n)

    def set_um_freq_zpe(self, v_um, zpe_um):
        """
        Set fundamental freq and ZPE of a mode.
        """
        if not self.v_um:
            self.v_um = v_um #in cm^-1 please
        elif v_um != self.v_um:
            raise ValueError("UM freq already exists for this mode and does not match previous value!")
        else:
            pass
        if not self.zpe_um:
            self.zpe_um = zpe_um #in kcal/mol please
        elif zpe_um != self.zpe_um:
            raise ValueError("UM zpe already exists for this mode and does not match previous value!")
        else:
            pass

    def set_ho_freq_zpe(self, v_ho):
        if not v_ho and not zpe_ho:
            self.v_ho = v_ho #in cm^-1 please
            self.zpe_ho = constants.h*v_ho*(constants.c*100)/\
                            2.0/constants.E_h #in kcal/mol
        else:
            print("Warning: HO freq or zpe already exists for this mode!")
            print("Previous v:",self.v_ho,"Current v:",v_ho)
            pass
            #raise ValueError("HO freq or zpe already exists for this mode!")

    def get_ho_freq_zpe(self):
        return self.v_ho, self.zpe_ho

    # Insert sample to the left of (before) first sample
    def push_sample(self, position, E, geom):
        self.v_sample.insert(0, E)
        self.x_sample.insert(0, position)
        self.sample_geometries.insert(0, geom)

    # Insert sample to the right of (after) last sample
    def push_back_sample(self, position, E, geom):
        self.v_sample.append(E)
        self.x_sample.append(position)
        self.sample_geometries.append(geom)

    def classify_tors(self, is_tors):  # Set whether this mode is a torsion or not
        self.isTors = is_tors

    def is_tors(self):  # Return whether this mode is a torsion or not
        return self.isTors

    def get_symmetry_number(self):
        if self.is_tors():
            return self.sigma
        else:
            return 1

    def get_x_sample(self):  # Return current sampling in x
        return np.array(self.x_sample)

    def get_v_sample(self):  # Return current sampling in y
        return np.array(self.v_sample)

    def get_step_size(self):
        return self.step_size

    def set_x_sample(self, x_array):
        self.x_sample = x_array

    def set_v_sample(self, v_array):
        self.v_sample = v_array

    def get_geoms(self):
        return self.sample_geometries

    def get_geometry(self, sample):
        return self.sample_geometries[sample]

    def get_equilibrium_geometry(self):
        return self.sample_geometries[self.x_sample.index(0.0)]

    def get_y_interp(self):
        return self.y_interp

    def get_x_interp(self):
        return self.x_interp

    def get_spline_fn(self):
        if self.spline_fn:
            return self.spline_fn
        else:
            self.spline(1000)
            return self.spline_fn

    def pes(self,x):
        cs = self.get_spline_fn()
        if x <= self.x_sample[0] or x >= self.x_sample[-1]:
            cs_copy = copy.deepcopy(cs)
            for i,c in enumerate(cs_copy.c[0]):
                # Coefficients of order x^3 changed to 0
                cs_copy.c[0,i] = 0.
            #print("x^3 coeffs:",cs.c[0],cs_copy.c[0])
            return cs_copy(x)
        else:
            #print("x^3 coeffs:",cs.c[0])
            return cs(x)
        
    def get_sample_mins(self):
        def in_rads(q):
            return self.is_tors() and not q[-1] % np.pi

        xsample = self.get_x_sample()
        ysample = self.get_v_sample()
        geom_sample_mins = []
        min_i = argrelextrema(ysample, np.less, mode='wrap')[0]
        max_i = argrelextrema(ysample, np.greater, mode='wrap')[0]
        if not len(min_i) == len(max_i):
            #print("Mins:",min_i)
            #print("Maxs:",max_i)
            if not min_i[0] == 0:
                min_i = np.append(0,min_i) 
            else:
                raise ValueError("Peaks and troughs have different lengths!")
        #REVIST THE FOLLOWING CODE
        if min_i[-1] == len(xsample)-1: #Avoids treating endpoint as a min
            min_i[-1] = 0
        #if self.is_tors() and not min_i[0] == 0:
        #    min_i = np.append(0, min_i)
        for i in min_i:
            geom_sample_mins.append(self.get_geometry(i))
        xmins = np.take(xsample, min_i)
        ymins = np.take(ysample, min_i)
        if not len(min_i) == len(max_i) and self.is_tors():
            if not min_i[0] == min_i[-1]-36:
                raise ValueError('Mins and maxs not the same length! Likely duplicate mins somewhere')
            else:
                print("-- seems like endpoints doublecounted. Fixed.")
        return xmins, ymins, geom_sample_mins

    def spline(self, N):  # Not-a-knot spline. Requires sampling first.
        if self.spline_fn and self.k and self.mu:
            return self.x_interp, self.y_interp
        xsample = self.get_x_sample()
        ysample = self.get_v_sample()
        if self.is_tors():
            if ysample[0] - ysample[-1] < 1E-3:
                ysample[-1] = ysample[0]
            else:
                print("V(360) =", ysample[-1], "and V(0) = ", ysample[0])
                raise ValueError("V(360) != V(0) -- torsion is not periodic!")
            f = CubicSpline(xsample, ysample, bc_type='periodic')
        else:
            if self.k and self.mu:
                K = self.mu*self.k*constants.amu*(10**(-20))/constants.E_h 
                f = CubicSpline(xsample, ysample, extrapolate=True,
                        bc_type=((2,K),(2,K)))
            else:
                f = CubicSpline(xsample, ysample, extrapolate=True)
                print("Warning, not using analytical second derivative at endpoints!")
                f = CubicSpline(xsample, ysample, extrapolate=True,
                    bc_type=((2,f(0,2)),(2,f(0,2))))
        xi = xsample[0]
        xf = xsample[-1]
        xnew = np.linspace(xi, xf, N)
        self.spline_fn = f      # Populate data structure
        self.x_interp = xnew    # ^
        self.y_interp = f(xnew) # ^^
        if not self.wells:
            self.add_wells()
        return self.x_interp, self.y_interp  # Return result

    def add_wells(self):
        def is_edgecase(peak):
            return peak == 0

        def in_rads(q):
            return self.is_tors() and not q[-1] % np.pi

        pes = self.get_y_interp()
        q = self.get_x_interp()
        spline_fn = self.get_spline_fn()
        well_domain = []  # Initialize well domain
        # Indices of local minima (incomplete -- no endpts)
        mins = argrelextrema(pes, np.less, mode='wrap')[0]
        # Indices of local maxima (no endpts, complete if @ eq)
        maxs = argrelextrema(pes, np.greater, mode='wrap')[0]
        if self.is_tors():
            n_peaks = len(maxs)
            for nth_peak in range(0, n_peaks):
                if is_edgecase(nth_peak):
                    # Piecewise domain if on boundary
                    well_domain.append([q[0], q[maxs[0]]])
                    well_domain.append([q[maxs[-1]], q[-1]])
                else:
                    well_domain.append(
                        [q[maxs[nth_peak-1]], q[maxs[nth_peak]]])
                #print(well_domain)
                well = Well(self.n, nth_peak+1, spline_fn, well_domain, in_rads(q))
                self.wells.append(well)
                well_domain = []  # Reset well domain
        else:
            well_domain.append([q[0], q[-1]])
            well = Well(self.n, 1, spline_fn, well_domain)
            self.wells.append(well)
        self.local_maxima = maxs
        self.local_minima = mins
        self.nwells = len(self.wells)

    def get_wells(self):
        return self.wells

    def print_geoms(self):
        for geom in self.get_geoms():
            for line in geom:
                print(line)
            print("")
        if not len(self.get_x_sample()) == len(self.get_geoms()):
            print("Error, geoms not same length as PES samples!")

    def print_all(self):  # Self-explanatory
        print(self.get_x_sample())
        print(self.get_v_sample())
        self.print_geoms()
        print(self.is_tors())
    
    def __str__(self):
        header = '\n\t********** Mode %i **********\n\n' % self.n
        ho_freq = 'HO frequency (cm-1): %.10f\n' % self.v_ho
        ho_zpe = 'HO zero point vibrational energy (kcal/mol): %.10f\n\n' % self.zpe_ho
        um_freq = 'UM frequency (cm-1): %.10f\n' % self.v_um
        um_zpe = 'UM zero point vibrational energy (kcal/mol): %.10f\n' % self.zpe_um
        return header+ho_freq+ho_zpe+um_freq+um_zpe 

"""
The class Well is used to store PES discretizations and integrations.
Integrations are done on initialization using spline function (can be from NMode) as input.
Super simple class.
"""
class Well:
    N = 1000  # GRID class variable

    def __init__(self, n, well_n, pes_fn, well_domain, x_in_rads=False):
        intgrl = 0
        flag = 0
        self.n = n                      # Mode number
        self.well_n = well_n            # Well number
        self.x_in_rads = x_in_rads      # bool -- is x in radians?
        self.pes_fn = pes_fn            # function of PES from NMode spline
        for rnge in well_domain:  # well_domain is length 1 for non-edge cases
            temp_domain = np.linspace(rnge[0], rnge[-1], self.N)
            temp_pes = pes_fn(temp_domain)
            temp_well_domain = [rnge[0], rnge[-1]]
            if not flag:                # if not an edgecase
                xdomain = temp_domain   # For graphing
                pes = temp_pes          # ^
                well_domain = temp_well_domain  # now well_domain is [xi, xf]
                #self.minimum = min(pes)
                flag = 1
            else:  # if edgecase, wrap last half peak around to first half peak
                if not self.x_in_rads:
                    xdomain = np.append(temp_domain[:-1]-360, xdomain)
                    well_domain[0] = temp_well_domain[0]-360
                else:
                    xdomain = np.append(temp_domain[:-1]-2*np.pi, xdomain)
                    well_domain[0] = temp_well_domain[0]-2*np.pi
                pes = np.append(temp_pes[:-1], pes)
            #intgrl = intgrl + \
            #    integrate.quad(lambda x: np.exp(pes_fn(x)), rnge[0], rnge[-1])[0]
        self.well_domain = well_domain
        self.pg_integral = intgrl
        self.pes = pes
        self.xdomain = xdomain
        self.xe = None
        self.v_ho = None
        self.zpe_ho = None
        self.path = None
        self.nmode = None       # Well as represented by NMode class
        # The following variables are set by UM sampling in set_well_UM
        self.ape = None
        self.um_mode_dict = None
        self.um_e_dict = None
        self.um_xyz_dict = None

    def get_well_n(self):
        return self.well_n

    def set_rel_u(self, U):
        """
        Set the energy relative to the absolute min
        """
        self.u = U

    def get_rel_u(self):
        return self.u

    def set_ho_freq_zpe(self, v_ho):
        """
        Set fundamental freq and ZPE of a well from HO approximation 
        """
        self.v_ho = v_ho # in cm^-1
        self.zpe_ho = constants.h*v_ho*(constants.c*100)/\
                            2.0/constants.E_h # in kcal/mol
        if self.nmode:
            self.nmode.set_ho_freq_zpe(v_ho)    
        else:
            print("Warning: NMode not defined for well, cannot set values.")

    def get_ho_freq_zpe(self):
        if self.v_ho and self.zpe_ho:
            return self.v_ho, self.zpe_ho
        return self.nmode.get_ho_freq_zpe()

    def set_um_freq_zpe(self, v_um, zpe_um):
        """
        Set fundamental freq and ZPE of a well.
        """
        if self.nmode:
            self.nmode.set_um_freq_zpe(v_um, zpe_um) 
        else:
            raise ValueError("NMode not defined for well, cannot set values.")

    def get_um_freq_zpe(self):
        return self.nmode.get_um_freq_zpe()

    def get_pes_fn(self):
        return self.well_domain, self.pes_fn

    def get_pes(self):  # for graphing
        return self.xdomain, self.pes

    def get_integral(self,T):
        return self.integral

    def set_filepath(self, path):
        """
        The path from reoptimization of sample min
        """
        if not self.path:
            self.path = path

    def get_filepath(self):
        """
        The path from reoptimization of sample min
        Serves as input for APE
        """
        return self.path

    def calc_well_thermo(self,T=298.15,p=100000):
        #
        from ape.calcThermo import ThermoJob
        thermo = ThermoJob(ape,polynomial_dict,m_dict,e_dict,xyz_dict,T=T,P=p)
        thermo.calcThermo(print_HOhf_result=False, zpe_of_Hohf=ape.zpe)
        
    def set_well_UM(self):
        from ape.main import APE
        from ape.main import dict_to_NMode
        from ape.FitPES import cubic_spline_interpolations
        print("Mode is",self.n)
        print("Well is",self.well_n)
        well_ape = APE(self.path,protocol='UMN',ncpus=4,which_modes=[self.n],well_n=self.well_n)
        well_ape.parse()
        self.ape = well_ape
        xyz_dict,e_dict,m_dict = well_ape.sampling()
        mode = dict_to_NMode(self.n,m_dict,e_dict,xyz_dict)
        self.nmode = mode
        self.um_mode_dict = m_dict
        self.um_e_dict = e_dict
        self.um_xyz_dict = xyz_dict

    def set_xe(self, xe):
        self.xe = xe

    def get_xe(self):
        return self.xe

    def get_well_nmode(self):
        return self.nmode

    def get_UM_dicts(self):
        return self.um_mode_dict, self.um_e_dict, self.um_xyz_dict

    def __str__(self):
        header = '\n\t********** Well %i **********\n\n' % self.well_n
        nmode = str(self.get_well_nmode())
        return header + nmode

def dict_to_NMode(mode, m_dict, e_dict, xyz_dict,
        rotors_dict=[], samp_obj=None):
    try:
        v_um = m_dict[mode]['v_um']
        zpe_um = m_dict[mode]['zpe_um']
        min_elect = m_dict[mode]['min_elect']
    except KeyError:
        try:
            v_ho = m_dict[mode]['v_ho']
        except KeyError:
            v_ho = 0
            pass
        m_dict[mode]['v_um'] = 0
        m_dict[mode]['zpe_um'] = 0
        v_um = 0
        zpe_um = 0
        min_elect = 0
        pass

    step = m_dict[mode]['step_size']
    if samp_obj.protocol == 'UMVT' or samp_obj.protocol == 'UMN'or samp_obj.protocol == 'TNUTS':
        samples = sorted(e_dict[mode].keys())
        xvals = [sample*step for sample in samples] #in rads
        vvals = [e_dict[mode][sample] for sample in samples] #in Hartree
        if xyz_dict:
            geoms = [xyz_dict[mode][sample] for sample in samples]
        else:
            geoms = None

    elif samp_obj.protocol == 'CMT':
        # Basically do the same construction,
        # but project n samples onto N 1d dihedral axes
        # Thus samples is n*N rather than n^N
        # i.e. N*1-d rather than N-d
        nsamples = m_dict[mode]['nsamples']
        samples = np.array([np.array(range(0,n)) \
                for n in nsamples])
        xvals = np.array([sample_array*step[i] \
                for i,sample_array in enumerate(samples)])
        vvals = e_dict[samp_obj.tors_modes[0]]\
                [samp_obj.tors_modes[0]]# Energy grid n^N
        geoms = None #TODO
    is_tors = (True if m_dict[mode]['mode'] == 'tors' else False)
    if is_tors:
        I = m_dict[mode]['M']   # amu * angstrom^2
        k = m_dict[mode]['K']   # 1/s^2
        sigma = m_dict[mode]['symmetry_number']
        ##### FOR NOW #####
        #if samp_obj.frags:
        #    sigma = 1
        ##################
        try:
            scan = rotors_dict[mode]['scan']
            print("Scan is",scan)
        except IndexError:
            scan = 0
            pass
        new_mode = NMode(n=mode,v=vvals,x=xvals,sample_geoms=geoms,
                tors=is_tors,scan=scan,
                v_ho=v_ho,v_um=v_um,zpe_um=zpe_um,
                I=I,k=k,min_elec=min_elect,sigma=sigma)
    else:
        mu = m_dict[mode]['M']  # amu
        k = m_dict[mode]['K']   # 1/s^2
        new_mode = NMode(
                n=mode,v=vvals,x=xvals,sample_geoms=geoms,
                tors=is_tors,v_ho=v_ho,v_um=v_um,zpe_um=zpe_um,
                mu=mu,k=k,min_elec=min_elect)
    return new_mode

def dicts_to_NModes(m_dict, e_dict, xyz_dict,
        rotors_dict=[], samp_obj=None, just_tors=False):
    # Get array of NMode types for easy use in PG, MCHO
    modes = []
    tmodes = []
    continue_toggle = False
    for mode in sorted(m_dict.keys()):
        if continue_toggle and m_dict[mode]['mode'] == 'tors':
            modes[0].set_mode_number(mode)
            continue
        nmode = dict_to_NMode(mode, m_dict, e_dict, xyz_dict,
                rotors_dict, samp_obj)
        modes.append(nmode)
        if mode['mode'] == 'tors':
            tmodes.append(nmode)
        if samp_obj.protocol == 'CMT':
            continue_toggle = True   # Only need one NMode object w/vectorized quantities
    samp_obj.NModes = modes
    samp_obj.tmodes = tmodes
    if just_tors:
        return tmodes
    return modes
