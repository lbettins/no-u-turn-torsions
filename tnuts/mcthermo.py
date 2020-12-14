# -*- coding: utf-8 -*-
import os
import math
import logging

import rmgpy.constants as constants
from arkane.output import prettify

from ape.FitPES import from_sampling_result, cubic_spline_interpolations
from ape.schrodinger import SetAnharmonicH
from ape.statmech import Statmech
from ape.thermo import ThermoJob

from tnuts.common import get_freq

class MCThermoJob:
    """
    The class to calculate thermodynamic properties, including E S G Cp
    """
    def __init__(self, trace, T, samp_obj=None, P=101325):
        self.T = T
        self.P = P #Pa
        self.trace = trace
        self.samp_obj = samp_obj
        self.conformer = self.samp_obj.conformer
        self.Thermo = ThermoJob(self.samp_obj.label,
                self.samp_obj.input_file,
                output_directory=self.samp_obj.output_directory, 
                P=self.P)
        self.Thermo.load_save()
        thermo.load_save()

    def calcThermo(self, print_HOhf_result=True):
        self.conformer = None
        E_trans, S_trans, Cv_trans, Q_trans =\
                self.calcTransThermo()
        E_rot, S_rot, Cv_rot, Q_rot =\
                self.calcRotThermo()
        E0, E_vib, S_vib, Cv_vib, Q_vib =\
                self.calcVibThermo(protocol="PG")

    def calcVibThermo(self, sb_protocol="HO", t_protocol="PG", t_subprotocol="CC",
            sb_subprotocol="MC"):
        E0_sb, E_sb, S_sb, Cv_sb, Q_sb =\
                self.calcSBThermo(protocol=sb_protocol) # for non-torsions
        E0_t, E_t, S_t, Cv_t, Q_t =\
                self.calcTThermo(protocol=t_protocol, subprotocol=t_subprotocol,
                        sb_protocol=sb_protocol, sb_subprotocol=sb_subprotocol) # for torsions
        return (E0_sb+E0_t), (E_sb+E_t), (S_sb+S_t), (Cv_sb+Cv_t), (Q_sb*Q_t)

    def calcTClassical(self, subprotocol):
        if not subprotocol in ["CC", "UC", "UU"]:
            raise TypeError(
                "Invalid subprotocol: valid options are 'CC', 'UC', or 'UU'.")
        ntors = self.samp_obj.n_rotors
        beta = 1/(constants.kB*T)*constants.E_h # beta in Hartree
        if "C" in subprotocol:
            # Calculate coupled torsional PES (V) contribution (all torsions)
            QV = self.Z*np.mean(self.trace.a)
            EV = np.mean(self.trace.bE)/beta * Hartree2kcal # E in kcal/mol
            SV = constants.kB(np.log(QV) + np.mean(self.trace.bE))*\
                    J2cal  # S in cal/mol.K
            CvV = beta/self.T*\
                    (np.var(self.trace.bE/beta))*Hartree2kcal # Cv in kcal/mol.K
            if "CC" in subprotocol:
                # Calculate coupled torsional kinetic (T) contribution
                # NOTE NOTE NOTE NOTE
                D = self.trace.get_sampler_stats("mass_matrix") # NEED TO CONVERT TO SI
        if "U" in subprotocol:
            # Calculate uncoupled torsional kinetic (T) contribution
            #NOTE NOTE NOTE NOTE
            D = np.diag([])
            if "UU" in subprotocol:
                # Calculate uncoupled torsional PES (V) contribution
                QV, EV, SV, CvV = 1, 0, 0, 0
                pass
        beta_si = 1./(constants.kB*T)
        prefactor = 1./(2*np.pi*beta_si*np.power(constants.hbar,2.))
        QT = np.power(prefactor*np.linalg.det(D), ntors/2)
        ET = ntors/(2*beta_si) * J2cal/1000     # ET in kcal/mol
        ST = constants.kB*(np.log(QT) + ntors/2.) * J2cal   #S in cal/mol.K
        CvT = constants.kB*ntors/2. * J2cal/1000    # Cv in kcal/mol.K

        Qtclass = QV * QT
        Etclass = EV + ET
        Stclass = SV + ST
        Cvtclass = CvV + CvT
        return Etclass, Stclass, Cvtclass, Qtclass

    def calcPGFactor(self, sb_protocol, sb_subprotocol):
        if not sb_protocol in ["HO", "UMVT"]:
            raise TypeError(
                    "Invalid protocol for stretches/bends: valid options\
                            are 'HO' or 'UMVT'.")
        F, E0divQtclass, DE, DS, DCv = 1, 0, 0, 0, 0
        if sb_protocol == "HO":
            if not sb_subprotocol in ["HO", "UMVT", "MC"]:
                raise TypeError(
                    "Invalid subprotocol for stretches/bends: valid options\
                            are 'HO', 'UMVT', or 'MC'.")
            kwargs = {'samp_obj' : self.samp_obj, 'D' : D, 'Thermo_obj' : self.Thermo}
            v = get_freq(protocol = sb_subprotocol, **kwargs) # Get freqs (cm-1)
            v *= 3E10   # 
            for i in range(ntors):
                qzpe = np.exp(-beta_si*hbar*v[i]/2)
                q = qzpe/(1-np.exp(-beta_si*hbar*v[i]))
                qc = np.power(hbar*v[i]*beta_si, -1)
                F *= q/qc
                E0divQtclass += 1./qc * (hbar*v[i]/2)
                DE += e-ec
                DS += s-sc
                DCv += cv-cvc
        elif sb_protocol == "UMVT":
            for mode in tors_modes:
                # i should reflect mode number
                v, e0, e, s, f, q, cv =\
                        self.Thermo.SolvEig(mode, self.T)
                qc = 
                F *= q/qc
                E0divQtclass += 1./qc * e0
                DE += e - ec 
                DS += s - sc
                DCv += cv - cvc
        return v, qc, F, E0divQtclass, DE, DS, DCv
        
    def calcTThermo(self, protocol="PG", subprotocol="CC",
            sb_protocol="HO", sb_subprotocol="HO")
        ntors = self.samp_obj.n_rotors
        # NOTE
        # Calc of torsions follows according to supplied protocol:
        if "PG" in protocol:
            # Calculate class torsional partition function
            Etclass, Stclass, Cvtclass, Qtclass =\
                    self.calcTClassical(subprotocol)
            # Calculate SB Ratio (F)
            v, qc, F, E0divQtclass, DE, DS, DCv =\
                    self.calcPGFactor(sb_protocol, sb_subprotocol)
            E0 = E0divQtclass*Qtclass
            E = DE + Etclass
            S = DS + Stclass
            Cv = DCv + Cvtclass
            Q = F*Qtclass
            return E0, E, S, Cv, Q
        elif protocol == "UMVT":
            for mode in tors_modes:
                v, E0, E, S, F, Q, Cv =\
                        self.Thermo.SolvEig(mode, self.T)

    def calcSBThermo(self, protocol):
        ZPE, E_int, S_int, Q_int, Cv_int = 0, 0, 0, 1, 0
        if protocol="HO":
            pass
        elif protocol="UMVT":
            # logging.info("Calculate internal E, S")
            for mode in sorted(self.Thermo.mode_dict.keys()):
                self.result_info.append("\n# \t********** Mode {} **********".format(mode))
                v, e0, E, S, F, Q, Cv = self.SolvEig(mode, T)
                ZPE += e0
                E_int += E
                S_int += S
                Q_int *= Q
                Cv_int += Cv
        return ZPE, E_int, S_int, Cv_int, Q_int

    def calcTransThermo(self):
        # Calculate global translation and rotation E, S
        E_trans = 1.5 * constants.R * self.T / 4184
        S_trans = self.conformer.modes[0].get_entropy(self.T) / 4.184 - constants.R * math.log(self.P / 101325) / 4.184
        Cv_trans = 1.5 * constants.R / 4184 * 1000
        Q_trans = self.conformer.modes[0].get_partition_function(self.T)
        return E_trans, S_trans, Cv_trans, Q_trans

    def calcRotThermo(self):
        E_rot = self.conformer.modes[1].get_enthalpy(self.T) / 4184
        S_rot = self.conformer.modes[1].get_entropy(self.T) / 4.184
        Cv_rot = self.conformer.modes[1].get_heat_capacity(self.T) / 4.184
        Q_rot = self.conformer.modes[1].get_partition_function(self.T)
        return E_rot, S_rot, Cv_rot, Q_rot

        self.result_info.append("\n# \t********** Final results **********\n\n")
        self.result_info.append("# Temperature (K): %.2f" % (T))
        self.result_info.append("# Pressure (Pa): %.0f" % (P))
        self.result_info.append("# Zero point vibrational energy (kcal/mol): %.10f" % (ZPE))
        self.result_info.append("# Translational energy (kcal/mol): %.10f" % (E_trans))
        self.result_info.append("# Translational entropy (cal/mol/K): %.10f" % (S_trans))
        self.result_info.append("# Translational Cv (cal/mol/K): %.10f" % (Cv_trans))
        self.result_info.append("# Rotational energy (kcal/mol): %.10f" % (E_rot))
        self.result_info.append("# Rotational entropy (cal/mol/K): %.10f" % (S_rot))
        self.result_info.append("# Rotational Cv (cal/mol/K): %.10f" % (Cv_rot))
        self.result_info.append("# Internal (rot+vib) energy (kcal/mol): %.10f" % (E_int))
        self.result_info.append("# Internal (tor+vib) entropy (cal/mol/K): %.10f" % (S_int))
        self.result_info.append("# Internal (tor+vib) Cv (cal/mol/K): %.10f" % (Cv_int))
        self.result_info.append("\n")
        self.result_info.append("# Total energy (kcal/mol): %.10f" % (E_trans + E_rot + E_int))
        self.result_info.append("# Total enthalpy (kcal/mol): %.10f" % (E_trans + E_rot + E_int + constants.kB * T * constants.Na / 4184))
        self.result_info.append("# Enthalpy H(%f K)-H(0 K) (kcal/mol):  %.10f" % (T, E_trans + E_rot + E_int + constants.kB * T * constants.Na / 4184 - ZPE))
        self.result_info.append("# Total entropy (cal/mol/K): %.10f" % (S_trans + S_rot + S_int))
        self.result_info.append("# Total Cv (cal/mol/K): %.10f" % (Cv_trans + Cv_rot + Cv_int))
        self.result_info.append("# Overall partition function: %.10f" % (Q_trans * Q_rot * Q_int * self.spin_multiplicity * self.optical_isomers))

        if print_HOhf_result:
            # compare to HOhf model
            E_vib = (self.conformer.modes[2].get_enthalpy(T) + self.zpe_of_Hohf) / 4184
            # E_vib should be calculated by freq...
            S_vib = self.conformer.modes[2].get_entropy(T) / 4.184
            self.result_info.append("\n")
            self.result_info.append("\n# \t********** HOhf results **********\n\n")
            self.result_info.append("# Translational energy (kcal/mol): %.10f" % (E_trans))
            self.result_info.append("# Rotational energy (kcal/mol): %.10f" % (E_rot))
            self.result_info.append("# Vibrational energy (kcal/mol): %.10f" % (E_vib))
            self.result_info.append("# gas constant (RT): %.10f" % (constants.R * T / 4184))
            self.result_info.append("# Translational entropy (cal/mol/K): %.10f" % (S_trans))
            self.result_info.append("# Rotational entropy (cal/mol/K): %.10f" % (S_rot))
            self.result_info.append("# Vibrational entropy (cal/mol/K): %.10f" % (S_vib))
            self.result_info.append("\n")
            self.result_info.append("# Total energy (kcal/mol): %.10f" % (E_trans + E_rot + E_vib))
            self.result_info.append("# Total enthalpy (kcal/mol): %.10f" % (E_trans + E_rot + E_vib + constants.R * T / 4184))
            self.result_info.append("# Enthalpy H(%f K)-H(0 K) (kcal/mol): %.10f" % (T, conformer.get_enthalpy(T) / 4184))
            self.result_info.append("# Total entropy (cal/mol/K): %.10f" % (S_trans + S_rot + S_vib))
            self.result_info.append("# Total Cv (cal/mol/K): %.10f" % (conformer.get_heat_capacity(T) / 4.184))
            self.result_info.append("# Overall partition function: %.10f" % (conformer.get_partition_function(T)))
        
        E0 = (self.conformer.E0.value_si - self.zpe_of_Hohf) * 0.001 / 4.184  + ZPE # in kcal/mol
        E = E_trans + E_rot + E_int # in kcal/mol
        S = S_trans + S_rot + S_int # in cal/mol/K
        F = (E + constants.R * T / 4184 - ZPE) - T * S * 0.001 + E0 # in kcal/mol
        Q = Q_trans * Q_rot * Q_int * self.spin_multiplicity * self.optical_isomers
        Cv = Cv_trans + Cv_rot + Cv_int # in cal/mol/K
        return  E0, E, S, F, Q, Cv
    
    def calcQMMMThermo(self, T, print_HOhf_result=True):
        P = self.P
        conformer = self.conformer
        logging.info("Calculate internal E, S")
        ZPE = 0
        E_int = 0
        S_int = 0
        # F_int = 0
        # Q_int = 1
        Cv_int = 0

        for mode in sorted(self.mode_dict.keys()):
            self.result_info.append("\n# \t********** Mode ",mode," **********\n\n")
            v, e0, E, S, F, Q, Cv = self.SolvEig(mode, T)
            ZPE += e0
            E_int += E
            S_int += S
            # F_int += F
            # Q_int *= Q
            Cv_int += Cv

        self.result_info.append("\n# \t********** Final results **********\n#\n")
        self.result_info.append("# Temperature (K): %.2f" % (T))
        self.result_info.append("# Pressure (Pa): %.0f" % (P))
        self.result_info.append("# Zero point vibrational energy (kcal/mol): %.10f" % (ZPE))
        self.result_info.append("# Internal (rot+vib) energy (kcal/mol): %.10f" % (E_int))
        self.result_info.append("# Internal (tor+vib) entropy (cal/mol/K): %.10f" % (S_int))
        self.result_info.append("# Internal (tor+vib) Cv (cal/mol/K): %.10f" % (Cv_int))

        if print_HOhf_result:
            # compare to HOhf model
            E_vib = (conformer.modes[2].get_enthalpy(T) + self.zpe_of_Hohf) / 4184
            # E_vib should be calculated by freq...
            S_vib = conformer.modes[2].get_entropy(T) / 4.184
            self.result_info.append("\n")
            self.result_info.append("\n# \t********** HOhf results **********\n\n")
            self.result_info.append("# Vibrational energy (kcal/mol): %.10f" % (E_vib))
            self.result_info.append("# Vibrational entropy (cal/mol/K): %.10f" % (S_vib))
        
        self.result_info.append('\n\n\n')

    
    def write_output(self):
        """
        Save the results of the ThermoJob to the `output.py` file located
        in `output_directory`.
        """
        output_file = os.path.join(self.output_directory, 'output.py')
        logging.info('Saving statistical mechanics parameters for {0}...'.format(self.label))
        f = open(output_file, 'a')

        conformer = self.conformer
        coordinates = conformer.coordinates.value_si * 1e10
        number = conformer.number.value_si

        f.write('# Coordinates for {0} in Input Orientation (angstroms):\n'.format(self.label))
        for i in range(coordinates.shape[0]):
            x = coordinates[i, 0]
            y = coordinates[i, 1]
            z = coordinates[i, 2]
            f.write('#   {0} {1:9.4f} {2:9.4f} {3:9.4f}\n'.format(self.symbols[i], x, y, z))
        f.write('\n')

        result = 'conformer(label={0!r}, E0={1!r}, modes={2!r}, spin_multiplicity={3:d}, optical_isomers={4:d}'.format(
            self.label,
            conformer.E0,
            conformer.modes,
            conformer.spin_multiplicity,
            conformer.optical_isomers,
        )
        try:
            result += ', frequency={0!r}'.format(self.sampling.imaginary_frequency)
        except AttributeError:
            pass
        result += ')'
        f.write('{0}\n\n'.format(prettify(result)))

        for line in self.result_info:
            line = line + '\n'
            f.write(line)
        f.write('\n')
        f.close()

    def execute(self):
        logging.info('Calculate thermodynamics for {0}'.format(self.label))
        for T in self.Tlist:
            self.result_info.append('\n\n# Thermodynamics for {0} at {1} K:\n'.format(self.label, T))
            if self.is_QM_MM_INTERFACE:
                self.calcQMMMThermo(T=T, print_HOhf_result=True)
            else:
                self.calcThermo(T=T, print_HOhf_result=True)
        self.write_output()
