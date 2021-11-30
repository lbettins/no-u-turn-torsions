#!/usr/bin/env python3
import os
import math
import logging
import rmgpy.constants as constants
import numpy as np
import pandas as pd
from arkane.output import prettify
from ape.FitPES import from_sampling_result, cubic_spline_interpolations
from ape.sampling import SamplingJob
from ape.schrodinger import SetAnharmonicH
from ape.statmech import Statmech
from ape.thermo import ThermoJob
from tnuts.mode import dict_to_NMode
from tnuts.thermo.common import get_tors_freqs, get_sb_freqs, solvHO, solvCHO,\
        solvUMClass, get_mass_matrix, get_data_frame
from tnuts.mc.metrics import get_sample_cov
from tnuts.stats import create_dfs

class MCThermoJob:
    """
    The class to calculate thermodynamic properties
    Units are finalized in kcal/mol or cal/mol, with inputs in Hartree
    ZPE [=] kcal/mol
    E   [=] kcal/mol
    S   [=] cal/mol.K
    Cv  [=] cal/mol.K
    """
    def __init__(self, results_directory, T,
            sampT=300, samp_obj=None, model=None,
            P=101325,
            t_protocols=['PG','UMVT','HO'], t_subprotocols=['C','U'],
            sb_protocols=['HO','UMVT'], sb_subprotocols=['HO','UMVT']):
        self.resdir = results_directory
        self.model = model
        self.sampT = sampT
        self.T = T
        self.rat = self.sampT/self.T
        self.P = P
        ###
        self.acm, self.bEcm, self.Ecvar,\
                self.DEcm, self.S = create_dfs(self.resdir,
                        sampT=self.sampT)
        ###
        self.samp_obj = samp_obj
        # Reiteration of 'load_save' from ape.statmech
        # Care should be taken to move '.csv' sampling result and the
        # '.out' input file to the results directory
        if self.samp_obj is None:
            for f in os.listdir(self.resdir):
                if not '.out' in f:
                    continue
                input_f = f
                break
            input_file = os.path.join(self.resdir, input_f)
            label = input_f.split('.')[0]
            self.samp_obj = SamplingJob(label, input_file, protocol='UMVT')
            self.samp_obj.parse()
        self.conformer = self.samp_obj.conformer
        self.label = self.samp_obj.label
        self.Thermo = ThermoJob(self.label, self.samp_obj.input_file,
                output_directory=self.resdir, P=self.P)
        self.Thermo.load_save()
        #self.tmodes = dicts_to_NModes(self.Thermo.mode_dict,
        #        self.Thermo.energy_dict,
        #        xyz_dict=None,
        #        just_tors=True)
        #self.NModes
        self.t_protocols = np.atleast_1d(t_protocols)
        self.t_subprotocols = np.atleast_1d(t_subprotocols)
        self.sb_protocols = np.atleast_1d(sb_protocols)
        self.sb_subprotocols = np.atleast_1d(sb_subprotocols)

        # Data frame initialization
        # label | mode | prot | sub | sbprot | sbsub | E0 | E | S | Cv | Q 
        # ----- | ---- | ---- | --- | ------ | ----- | -- | - | - | -- | -  
        # Ex.   |      |      |     |        |       |    |   |   |    |   
        # "s15" | "sb" |  NaN |  NaN|  "ho"  | "umvt"|  α | β | γ |  δ | ε 
        # "s15" |"tors"| "pg" | "cc"| "umvt" |  NaN  |  Α | B | Γ |  Δ | Ε 
        # "s15" | "rot"|  NaN |  NaN|  NaN   |  NaN  |  ζ | η | θ |  ι | κ 
        # "s15" |"trns"|  NaN |  NaN|  NaN   |  NaN  |  Ζ | Η | Θ |  Ι | Κ 
        # "s20" |"tors"| "pg" | "uu"|  "ho"  |  "ho" |  λ | μ | ν |  ξ | π 
        # "s1"  |"tors"|"umvt"|  NaN|  NaN   |  NaN  |  Λ | Μ | Ν |  Ξ | Π 
        self.csv = os.path.join(self.resdir, "thermo{T}_{label}.csv".format(T=self.T,label=self.label))
        self.total_thermo = get_data_frame(self.csv)
        self.data_frame = pd.DataFrame({'mode' : []})

    def execute(self):
        self.calcThermo(write=True)
        return self.data_frame, self.csv

    def calcThermo(self, write=True, print_output=True):
        """
        Calculate component thermodynamic quantities.
        Unit conversions performed by each operation.
        Stored in self.data_frame data frame.
        """
        self.calcTransThermo()
        self.calcRotThermo()
        for sb_protocol in self.sb_protocols:
            self.calcSBThermo(protocol=sb_protocol)
        for t_protocol in self.t_protocols:
            if t_protocol == 'PG':
                ############ Pitzer-Gwinn Methods ############
                for t_subprotocol in self.t_subprotocols:
                    ######### PG Classical Partition #########
                    for sb_protocol in self.sb_protocols:
                        ######### Pitzer-Gwinn Factor #########
                        if sb_protocol == 'HO': # PG factor = HO
                            ######### HO ω's #########
                            for sb_subprotocol in self.sb_subprotocols:
                                self.calcTThermo(protocol=t_protocol,
                                        subprotocol=t_subprotocol,
                                        sb_protocol=sb_protocol,
                                        sb_subprotocol=sb_subprotocol)
                        else: # PG factor = UMVT / MC
                            self.calcTThermo(protocol=t_protocol,
                                    subprotocol=t_subprotocol,
                                    sb_protocol=sb_protocol,
                                    sb_subprotocol=None)
            else: # method is not PG (is UMVT)
                self.calcTThermo(protocol=t_protocol, subprotocol=None,
                        sb_protocol=None, sb_subprotocol=None)
        self.total_thermo = pd.concat([self.data_frame], keys=[self.label],
                names=['species'])
        if write:
            self.total_thermo.to_csv(self.csv)
        if print_output:
            pass

    def calcTClassical(self, subprotocol):
        """
        Calculate the classical torsional thermo properties
        using statistical mechanics.
        Unit conversions in situ performed before returning.
        """
        if not subprotocol in ['C', 'U']:
            raise TypeError("Invalid subprotocol")
        ntors = self.samp_obj.n_rotors
        beta = 1/(constants.kB*self.T)*constants.E_h # beta in Hartree
        #betac = beta*self.rat
        print("Ratio of temperatures is", self.rat)
        # Unit conversion constants:
        J2cal = 1./4.184    # 1cal / 4.184J
        Hartree2kcal = constants.E_h*\
                constants.Na*J2cal/1000   # (J/H)*(1cal/4.184J)*1k
        #################################################
        # Calculate  torsional kinetic (T) contribution #
        #################################################
        D = get_mass_matrix(None, self.T,
                self.Thermo.mode_dict,
                protocol="uncoupled") # SI
        R = 1.985877534e-3       # kcal/mol.K
        beta_si = 1./(constants.kB*self.T)
        prefactor = 1./(2*np.pi*beta_si*np.power(constants.hbar,2.))
        QT = np.power(prefactor, ntors/2)*np.power(np.linalg.det(D), 0.5)
        ET = 0.5*ntors*R*self.T     # ET in kcal/mol
        ST = R*(np.log(QT) + ntors/2.) * 1000   #S in cal/mol.K
        CvT = R*ntors/2. * 1000    # Cv in cal/mol.K

        EtclassU, StclassU, CvtclassU, QtclassU, Qv = 0, 0, 0, 1, 1
        for mode in sorted(self.Thermo.mode_dict.keys()):
            if self.Thermo.mode_dict[mode]['mode'] != 'tors':
                continue
            # Calculate classical properties
            NMode = dict_to_NMode(mode, self.Thermo.mode_dict,
                    self.Thermo.energy_dict,
                    [],
                    [], self.samp_obj)
            ec, sc, qc, qv, cvc =\
                    solvUMClass(NMode, self.T)
            EtclassU += ec
            StclassU += sc
            CvtclassU += cvc
            QtclassU *= qc
            Qv *= qv
        if "U" in subprotocol:
            return EtclassU, StclassU, CvtclassU, QtclassU
        elif 'C' in subprotocol:
            # Calculate coupled torsional PES contribution for all tors
            #QV = Qv*np.power(np.mean(self.trace.a), self.rat)
            #print("Kinetic pf, coupled:", QT)
            #print("Potential partition function, un/coupled:", Qv, QV)
            #print("Product:", QT*QV)
            ############# CHECK THIS ####################
            #EV = np.mean(self.trace.bE*self.rat)*R*self.T # E in kcal/mol
            #SV = R*(np.log(QV) + np.mean(self.trace.bE)*self.rat)*1000\
            #    # S in cal/mol.K
            #CvV = beta/self.T*\
            #        (np.var(self.trace.bE/betac))\
            #        *Hartree2kcal*1000 # Cv in cal/mol.K
            #QtclassC = QV*QT
            #EtclassC = EV+ET
            #StclassC = SV+ST
            #CvtclassC = CvV+CvT
            ############################################
            # CUMULATIVE MEAN ##########################
            ############################################
            Qcm = Qv*np.power(self.acm, self.rat) #* QT
            print("partition fn, fperturb;",Qcm)
            print("prior partition:",Qv)
            #Qcm = self.qcm * QT
            #print(Qcm)
            #print(np.log(Qcm))
            Qcm.columns = Qcm.columns.str.replace("a", "E")
            Ecm = self.bEcm * self.rat*R*self.T + ET
            Scm = R*(np.log(Qcm).add(self.bEcm*self.rat))*1000 + ST # cal/mol.K
            
            print("Entropy is:", Scm)
            print("kinetic entropy is", ST)
            #Scm = R*self.S*1000
            #Qcm = np.exp(-self.S - Ecm)
            #print(Scm)
            Cvcm = beta/self.T*\
                    self.Ecvar*Hartree2kcal*1000 + CvT # cal/mol.K
            return Ecm, Scm, Cvcm, Qcm
            return EtclassC, StclassC, CvtclassC, QtclassC

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
        Qc, F, E0, DE, DS, DCv = 1, 1, 0, 0, 0, 0
        if sb_protocol == "HO":
            if not sb_subprotocol in ["HO", "UMVT", "MC"]:
                raise TypeError(
                    "Invalid subprotocol for stretches/bends: valid options\
                            are 'HO', 'UMVT', or 'MC'.")
            kwargs = {'samp_obj' : self.samp_obj, 'T' : self.T, 'Thermo_obj' : self.Thermo}
            ws = get_tors_freqs(protocol = sb_subprotocol, **kwargs) # s^-1
            for i,w in enumerate(ws):
                # Calculate quantum HO properties
                e0, e, s, q, cv =\
                        solvHO(w, self.T)
                ec, sc, qc, cvc =\
                        solvCHO(w, self.T)
                F *= q/qc
                Qc *= qc
                E0 += e0
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
                ec, sc, qc, qv, cvc =\
                        solvUMClass(NMode, self.T)
                F *= q/qc
                Qc *= qc
                E0 += e0
                DE += e - ec
                DS += s - sc
                DCv += cv - cvc
        v=None
        return v, Qc, F, E0, DE, DS, DCv

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
            v, qc, F, E0, DE, DS, DCv =\
                    self.calcPGFactor(sb_protocol, sb_subprotocol)
            ### Data frames if coupled
            E0 = E0
            E = DE + Etclass
            S = DS + Stclass
            Cv = DCv + Cvtclass
            Q = F*Qtclass
            Qc = Qtclass
            ###
            # SAVE THEM!!!
            if subprotocol == 'C':
                if sb_protocol == 'UMVT':
                    print("PG Entropy is", DS)
                    R = 1.985877534e-3       # kcal/mol.K
                    Hcm = E - E0 + self.trans_dict['e'] + self.rot_dict['e'] + R*self.T
                    Hcm += self.ho_dict['e'] - self.ho_dict['e0']
                    Scm = S + self.trans_dict['s'] + self.rot_dict['s'] +\
                            self.ho_dict['s']
                    Cvcm = Cv + self.trans_dict['cv'] + self.rot_dict['cv'] +\
                            self.ho_dict['cv']
                    Qpgcm = Q * self.ho_dict['q']

                    name = os.path.join(self.resdir,'{}_{}K.csv')
                    Hcm.to_csv(name.format('H',self.T))
                    Scm.to_csv(name.format('S',self.T))
                    Cvcm.to_csv(name.format('Cv',self.T))
                    Qpgcm.to_csv(name.format('Qpg',self.T))
                    ####
                #For storing in data frame
                E = np.mean(E.iloc[-1])
                S = np.mean(S.iloc[-1])
                Cv = np.mean(Cv.iloc[-1])
                Q = np.mean(Q.iloc[-1])
                Qc = np.mean(Qtclass.iloc[-1])
                ####
        elif protocol == "UMVT":
            E0, E, S, Cv, Q, F, Qc = 0, 0, 0, 0, 1, None, None
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
        elif protocol == 'HO':
            E0, E, S, Cv, Q, F, Qc = 0, 0, 0, 0, 1, None, None
            kwargs = {'samp_obj' : self.samp_obj, 'T' : self.T, 'Thermo_obj' : self.Thermo}
            ws = get_tors_freqs(protocol=protocol, **kwargs) # s^-1
            for i,w in enumerate(ws):
                # Calculate quantum HO properties
                e0, e, s, q, cv =\
                        solvHO(w, self.T)
                E0 += e0
                E += e
                S += s
                Cv += cv
                Q *= q
        t_dict = {'mode' : 'tors', 'T' : self.T,
                'protocol' : protocol, 'subprotocol' : subprotocol,
                'sb_protocol' : sb_protocol,
                'sb_subprotocol' : sb_subprotocol,
                'e0' : E0, 'e' : E, 's' : S, 'cv' : Cv, 'q' : Q,
                'qc' : Qc, 'f' : F}
        self.data_frame = self.data_frame.append(t_dict, ignore_index=True)

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
            freqs = get_sb_freqs(self.Thermo.mode_dict)
            for w in freqs:
                e0, e, s, q, cv =\
                        solvHO(w, self.T)
                ZPE += e0
                E_int += e
                S_int += s
                Q_int *= q
                Cv_int += cv
        elif protocol=="UMVT":
            # Calculate UMVT thermo for stretches/bends
            for mode in sorted(self.Thermo.mode_dict.keys()):
                if self.Thermo.mode_dict[mode]['mode'] == 'tors':  # skip torsions
                    continue
                v, e0, E, S, F, Q, Cv = self.Thermo.SolvEig(mode, self.T)
                ZPE += e0
                E_int += E
                S_int += S
                Q_int *= Q
                Cv_int += Cv
        sb_dict = {'mode' : 'sb', 'T' : self.T,
                'sb_protocol' : protocol,
                'e0' : ZPE, 'e' : E_int, 's' : S_int, 'cv' : Cv_int,
                    'q' : Q_int}
        if protocol=='UMVT':
            self.umn_dict = sb_dict
        elif protocol=='HO':
            self.ho_dict = sb_dict
        self.data_frame = self.data_frame.append(sb_dict, ignore_index=True)

    def calcTransThermo(self):
        # Calculate global translation (ideal gas, Sackur-Tetrode)
        # Unit conversion included
        E_trans = 1.5 * constants.R * self.T / 4184
        S_trans = self.conformer.modes[0].get_entropy(self.T) / 4.184 - constants.R * math.log(self.P / 101325) / 4.184
        Cv_trans = 1.5 * constants.R / 4184 * 1000
        Q_trans = self.conformer.modes[0].get_partition_function(self.T)
        self.trans_dict = {'mode' : 'trans', 'T' : self.T, 
                'e' : E_trans, 's' : S_trans,
                    'cv' : Cv_trans, 'q' : Q_trans}
        self.data_frame = self.data_frame.append(self.trans_dict,
                ignore_index=True)

    def calcRotThermo(self):
        # Calculate global rotation (rigid rotor)
        # Unit conversion included
        E_rot = self.conformer.modes[1].get_enthalpy(self.T) / 4184
        S_rot = self.conformer.modes[1].get_entropy(self.T) / 4.184
        Cv_rot = self.conformer.modes[1].get_heat_capacity(self.T) / 4.184
        Q_rot = self.conformer.modes[1].get_partition_function(self.T)
        self.rot_dict = {'mode' : 'rot', 'T' : self.T,
                'e' : E_rot, 's' : S_rot, 'cv' :
                    Cv_rot, 'q' : Q_rot}
        self.data_frame = self.data_frame.append(self.rot_dict,
                ignore_index=True)

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

if __name__ == '__main__':
    from tnuts.thermo.main import main
    main()
