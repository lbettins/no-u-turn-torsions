#!/usr/bin/env python3
import os
import math
import logging
import rmgpy.constants as constants
import numpy as np
from arkane.output import prettify
from ape.FitPES import from_sampling_result, cubic_spline_interpolations
from ape.schrodinger import SetAnharmonicH
from ape.statmech import Statmech
from ape.thermo import ThermoJob
from tnuts.mode import dict_to_NMode
from tnuts.thermo.common import get_tors_freqs, solvHO, solvCHO,\
        solvUMClass, get_mass_matrix

class MCThermoJob:
    """
    The class to calculate thermodynamic properties
    Units are finalized in kcal/mol or cal/mol, with inputs in Hartree
    ZPE [=] kcal/mol
    E   [=] kcal/mol
    S   [=] cal/mol.K
    Cv  [=] cal/mol.K
    """
    def __init__(self, trace, T, samp_obj=None, P=101325):
        self.T = T
        self.P = P
        self.trace = trace
        self.samp_obj = samp_obj
        self.conformer = self.samp_obj.conformer
        self.Thermo = ThermoJob(self.samp_obj.label, self.samp_obj.input_file,
                output_directory=self.samp_obj.output_directory, P=self.P)
        self.Thermo.load_save()
        self.tmodes = self.samp_obj.tmodes
        self.NModes = self.samp_obj.NModes

    def calcThermo(self, print_HOhf_result=True):
        """
        Calculate component thermodynamic quantities.
        Unit conversions performed by each operation before returning.
        """
        E_trans, S_trans, Cv_trans, Q_trans =\
                self.calcTransThermo()
        E_rot, S_rot, Cv_rot, Q_rot =\
                self.calcRotThermo()
        E0, E_vib, S_vib, Cv_vib, Q_vib =\
                self.calcVibThermo(protocol="PG")

    def calcVibThermo(self, sb_protocol="HO", t_protocol="PG", t_subprotocol="CC",
            sb_subprotocol="MC"):
        """
        Calculate component thermo quantities for internal modes.
        Internal modes separted into:
            Torsions
                protocols: 'UMVT', 'PG'(coupled, u/c, uncoupled)
            Stretches / Bends
                protocols: 'HO'(ω: ho, umvt, mc), 'UMVT'
        Unit conversions performed by each operation before returning.
        """
        E0_sb, E_sb, S_sb, Cv_sb, Q_sb =\
                self.calcSBThermo(protocol=sb_protocol) # for non-torsions
        E0_t, E_t, S_t, Cv_t, Q_t =\
                self.calcTThermo(protocol=t_protocol, subprotocol=t_subprotocol,
                        sb_protocol=sb_protocol, sb_subprotocol=sb_subprotocol) # for torsions
        return (E0_sb+E0_t), (E_sb+E_t), (S_sb+S_t), (Cv_sb+Cv_t), (Q_sb*Q_t)

    def calcTClassical(self, subprotocol):
        """
        Calculate the classical torsional thermo properties
        using statistical mechanics.
        Unit conversions in situ performed before returning.
        """
        if not subprotocol in ['CC', 'UC', 'UU']:
            raise TypeError("Invalid subprotocol")
        ntors = self.samp_obj.n_rotors
        beta = 1/(constants.kB*self.T)*constants.E_h # beta in Hartree
        # Unit conversion constants:
        J2cal = 1./4.184    # 1cal / 4.184J
        Hartree2kcal = constants.E_h * J2cal*1000   # (J/H)*(1cal/4.184J)*1k
        if 'C' in subprotocol:
            # Calculate coupled torsional PES contribution for all tors
            QV = self.Z*np.mean(self.trace.a)
            EV = np.mean(self.trace.bE)/beta * Hartree2kcal # E in kcal/mol
            SV = constants.kB(np.log(QV) + np.mean(self.trace.bE))*\
                    J2cal  # S in cal/mol.K
            CvV = beta/self.T*\
                    (np.var(self.trace.bE/beta))*Hartree2kcal*1000 # Cv in cal/mol.K
            if "CC" in subprotocol:
                # Calculate coupled torsional kinetic (T) contribution
                # TODO
                D = get_mass_matrix(protocol='coupled') # SI
        if "U" in subprotocol:
            # Calculate uncoupled torsional kinetic (T) contribution
            # TODO
            D = get_mass_matrix(protocol='uncoupled')   # SI, diagonal
            if "UU" in subprotocol:
                # Calculate uncoupled torsional PES (V) contribution
                QV, EV, SV, CvV = 1, 0, 0, 0
                pass
        beta_si = 1./(constants.kB*self.T)
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
        """
        Calculate the thermodynamics associated with PG Q/Cl ratio.
        F refers to the ratio of q/cl partition fns, NOT Helmholtz f.e.
        This carries a quantum term and therefore a ZPE.
        Unit conversions performed by operations prior to returning.
        """
        if not sb_protocol in ["HO", "UMVT"]:
            raise TypeError(
                    "Invalid protocol for stretches/bends: valid options\
                            are 'HO' or 'UMVT'.")
        Qc, F, E0divQtclass, DE, DS, DCv = 1, 1, 0, 0, 0, 0
        if sb_protocol == "HO":
            if not sb_subprotocol in ["HO", "UMVT", "MC"]:
                raise TypeError(
                    "Invalid subprotocol for stretches/bends: valid options\
                            are 'HO', 'UMVT', or 'MC'.")
            kwargs = {'samp_obj' : self.samp_obj, 'D' : D, 'Thermo_obj' : self.Thermo}
            ws = get_tors_freqs(protocol = sb_subprotocol, **kwargs) # Get freqs
            for i,w in enumerate(ws):
                # Calculate quantum HO properties
                e0, e, s, q, cv =\
                        solvHO(w, self.T)
                ec, sc, cvc, qc =\
                        solvCHO(w, self.T)
                F *= q/qc
                Qc *= qc
                E0divQtclass += e0/qc
                DE += e-ec
                DS += s-sc
                DCv += cv-cvc
        elif sb_protocol == "UMVT":
            for mode in sorted(self.Thermo.mode_dict.keys()):
                if self.Thermo.mode_dict[mode]['mode'] != 'tors':
                    continue
                # Calculate quantum properties
                v, e0, e, s, f, q, cv =\
                        self.Thermo.SolvEig(mode, self.T)
                # Calculate classical properties
                NMode = dict_to_NMode(mode, self.Thermo.mode_dict,
                        self.Thermo.energy_dict,
                        [],
                        [], self.samp_obj)
                ec, sc, cvc, qc =\
                        solvUMClass(NMode, self.T)
                F *= q/qc
                Qc *= qc
                E0divQtclass += 1./qc*e0
                DE += e - ec
                DS += s - sc
                DCv += cv - cvc
        return v, Qc, F, E0divQtclass, DE, DS, DCv

    def calcTThermo(self, protocol="PG", subprotocol="CC",\
            sb_protocol="HO", sb_subprotocol="HO"):
        """
        Calculate thermodynamics of internal rotations (torsions).
        Two possibilities:
            1. Pitzer-Gwinn
                - Classical partition function
                    1. Coupled
                    2. Uncoupled
                    3. Hybrid
                - PG factor
                    1. Q_HO/Q_CHO(ω)
                        1. ωΗΟ
                        2. ωUMVT
                        3. ωMC
                    2. Q_UMVT/Q_UMC
            2. UM-VT
        Unit conversions done by operations prior to being returned.
        """
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
        elif protocol == "UMVT":
            E0, E, S, Cv, Q = 0, 0, 0, 0, 1
            for mode in sorted(self.Thermo.mode_dict.keys()):
                if self.Thermo.mode_dict[mode]['mode'] != 'tors':
                    continue
                v, e0, e, s, f, q, cv =\
                        self.Thermo.SolvEig(mode, self.T)
                E0 += e0
                E += e
                S += s
                Cv += cv
                Q *= q
        return E0, E, S, Cv, Q

    def calcSBThermo(self, protocol):
        """
        Calculate thermodynamics of stretches and bends, distinct from tors.
        Relies heavily on outside methods (Yi-Pei Li)
        Two methods, which match PG protocol:
            1. HO
                - use harmonic approximation
            2. UMVT
                - from anharmonic sampling (done prior to MC)
        Unit conversions done in situ before returning
        """
        ZPE, E_int, S_int, Q_int, Cv_int = 0, 0, 0, 1, 0
        if protocol=="HO":
            # Calculate HO thermo for stretches/bends
            #TODO plus unit conversion
            pass
        elif protocol=="UMVT":
            # Calculate UMVT thermo for stretches/bends
            for mode in sorted(self.Thermo.mode_dict.keys()):
                if self.Thermo.mode_dict[mode]['mode'] == 'tors':  # skip torsions
                    continue
                v, e0, E, S, F, Q, Cv = self.SolvEig(mode, self.T)
                ZPE += e0
                E_int += E
                S_int += S
                Q_int *= Q
                Cv_int += Cv
        return ZPE, E_int, S_int, Cv_int, Q_int

    def calcTransThermo(self):
        # Calculate global translation (ideal gas, Sackur-Tetrode)
        # Unit conversion included
        E_trans = 1.5 * constants.R * self.T / 4184
        S_trans = self.conformer.modes[0].get_entropy(self.T) / 4.184 - constants.R * math.log(self.P / 101325) / 4.184
        Cv_trans = 1.5 * constants.R / 4184 * 1000
        Q_trans = self.conformer.modes[0].get_partition_function(self.T)
        return E_trans, S_trans, Cv_trans, Q_trans

    def calcRotThermo(self):
        # Calculate global rotation (rigid rotor)
        # Unit conversion included
        E_rot = self.conformer.modes[1].get_enthalpy(self.T) / 4184
        S_rot = self.conformer.modes[1].get_entropy(self.T) / 4.184
        Cv_rot = self.conformer.modes[1].get_heat_capacity(self.T) / 4.184
        Q_rot = self.conformer.modes[1].get_partition_function(self.T)
        return E_rot, S_rot, Cv_rot, Q_rot
