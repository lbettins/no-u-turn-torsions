#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import rmgpy.constants as constants
from ape.common import get_internal_rotation_freq

def get_data_frame(csv_filepath):
    if not os.path.exists(csv_filepath):
        # return empty data frame
        return pd.DataFrame({"mode" : []})
    else:
        # return already existing .csv
        return pd.read_csv(csv_filepath)

def get_sb_freqs(mode_dict, protocol='HO'):
    freqs = []
    for mode in sorted(mode_dict.keys()):
        freqs.append(np.sqrt(mode_dict[mode]['K']))
    return freqs    # ω in 1/s

def get_tors_freqs(protocol='HO', D=None, Thermo_obj=None, samp_obj=None,
        T=None):
    """
    Get torsional frequencies under given protocol.
    Ex. 'UMVT' returns anharmonic frequency directly from Hamiltonian
    eigenvalues.
    """
    if protocol not in ["HO", "UMVT", "MC"]:
            raise TypeError("Protocol not recognized, pls choose 'HO', 'UMVT', or 'MC'.")
    freqs = []
    for mode in sorted(Thermo_obj.mode_dict.keys()):
        if Thermo_obj.mode_dict[mode]['mode'] != 'tors':
            continue
        if protocol == "HO":
        # Determine rotors and which normal mode freq corresponds to the
        # desired torsion
        #rotors = [[rotor['pivots'], rotor['top']] for rotor in
        #        samp_obj.rotors_dict.values()]
        #for i in range(samp_obj.n_rotors):
        #    target_rotor = rotors[i]
        #    int_freq = get_internal_rotation_freq(samp_obj.conformer,
        #            samp_obj.hessian, target_rotor, rotors, samp_obj.linearity,
        #            samp_obj.n_vib, is_QM_MM_ΙΝΤΕRFACE=samp_obj.is_QM_MM_ΙΝΤΕRFACE,
        #            label=samp_obj.label)
        #    freqs.append(int_freq * (2*np.pi*constants.c*100)) # in s^-1
            int_freq = np.sqrt(Thermo_obj.mode_dict[mode]['K'])
        elif protocol == "UMVT":
            # Determine anharmonic frequencies of all torsions.
            solv_eig_out = Thermo_obj.SolvEig(mode, T)
            int_freq = solv_eig_out[0]*(2*np.pi*constants.c*100) # in s^-1
        freqs.append(int_freq)
    return freqs

def get_mass_matrix(trace, model, T, mode_dict, protocol="uncoupled"):
    if protocol == "coupled":
        return get_mass_matrix(trace, model, T, mode_dict, protocol="uncoupled")
        from tnuts.mc.metrics import get_sample_cov
        # # # # # # # # # # # # # # # # # # #
        #    Σtrue = β/(Mω^2) = β/κ         # ω = 1/s for optimal M
        #   =>  Σsim = βΜ^-1                #
        #    => M = β(Σsim)^-1              #
        # # # # # # # # # # # # # # # # # # #
        sig = get_sample_cov(trace, model)[0]
        beta = 1/(constants.kB*T)
        w = get_tors_freqs()
        M = beta/sig        # Units?
        return M
    elif protocol == "uncoupled":
        M = []
        for mode in sorted(mode_dict.keys()):
            if mode_dict[mode]['mode'] != 'tors':
                continue
            M.append(mode_dict[mode]['M'])  # Units of amu*angstrom^2
        M = np.diag(M)*constants.amu*(1e-20)   # convert to SI kg*m^2
        return M

def solvHO(w, T):
    """Solve quantum HO given frequency and temperature"""
    # Constants ####################
    hbar = constants.hbar   # Js
    R = 1.985877534e-3  # kcal/mol.K
    Na = constants.Na
    beta = 1./constants.kB/T    # J/K
    J2kcal = 0.000239006
    ################################
    ZPE = 0.5*hbar*w    # J
    q = np.power(2*np.sinh(beta*ZPE), -1)
    e0 = ZPE*Na*J2kcal  # kcal/mol
    e = e0 / np.tanh(beta*ZPE)  # kcal/mol
    f = -R*T*np.log(q)  # kcal/mol
    s = (e-f)/T * 1000  # cal/mol.K
    cv = R*1000*np.power(beta*ZPE/np.sinh(beta*ZPE), 2) # cal/mol.K
    return e0, e, s, q, cv

def solvCHO(w, T):
    """Solve classical HO partition function"""
    # Constants ####################
    hbar = constants.hbar   # Js
    beta = 1./constants.kB/T    # J/K
    R = 1.985877534e-3  # kcal/mol.K
    ################################
    q = np.power(beta*hbar*w, -1)
    e = R*T             # kcal/mol
    f = -R*T*np.log(q)  # kcal/mol
    s = (e-f)/T * 1000  # cal/mol.K
    cv = R*1000        # cal/mol.K
    return e, s, q, cv

def solvUMClass(nmode, T):
    """Solve classical UM partition function for torsions"""
    R = 1.985877534e-3  # kcal/mol.K
    qc = nmode.get_classical_partition_fn(T)
    qv = nmode.get_classical_partition_fn(T, protocol='V')
    ec = nmode.get_average_energy(T)        # kcal/mol (T + V)
    fc = nmode.get_helmholtz_free_energy(T) # kcal/mol
    sc = (ec-fc)/T * 1000                   # cal/mol.K
    #var_e = nmode.get_energy_fluctuation(T) # (kcal/mol)^2
    cvc = nmode.get_heat_capacity(T)*1000   # cal/mol.K
    return ec, sc, qc, qv, cvc

#SCRATCH
#e = (e0 + hbar*w/(np.exp(beta*hbar*w)-1))*J2kcal*Na
#s = R*1000\
#        *(-np.log(1-np.exp(-hbar*w*beta)) +\
#        (hbar*w*beta)*np.exp(-hbar*w*beta)/(1-np.exp(-hbar*w*beta)))
