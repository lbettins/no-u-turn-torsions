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

def get_tors_freqs(protocol='HO', D=None, Thermo_obj=None, samp_obj=None,
        T=None):
    """
    Get torsional frequencies under given protocol.
    Ex. 'UMVT' returns anharmonic frequency directly from Hamiltonian
    eigenvalues.
    """
    freqs = []
    if protocol not in ["HO", "UMVT", "MC"]:
        raise TypeError("Protocol not recognized, pls choose 'HO', 'UMVT', or 'MC'.")
    if protocol == "HO":
        # Determine rotors and which normal mode freq corresponds to the
        # desired torsion
        rotors = [[rotor['pivots'], rotor['top']] for rotor in
                samp_obj.rotors_dict.values()]
        for i in range(samp_obj.n_rotors):
            target_rotor = rotors[i]
            int_freq = get_internal_rotation_freq(samp_obj.conformer,
                    samp_obj.hessian, target_rotor, rotors, samp_obj.linearity,
                    samp_obj.n_vib, is_QM_MM_ΙΝΤΕRFACE=samp_obj.is_QM_MM_ΙΝΤΕRFACE,
                    label=samp_obj.label)
            freqs.append(int_freq * (2*np.pi*constants.c*100)) # in s^-1
    elif protocol == "UMVT":
        # Determine anharmonic frequencies of all torsions.
        for mode in (Thermo_obj.mode_dict.keys()):
            if Thermo_obj.mode_dict[mode]['mode'] != 'tors':
                continue
            solv_eig_out = Thermo_obj.SolvEig(mode, T)
            int_freq = solv_eig_out[0] * (2*np.pi*constants.c*100) # in s^-1
            freqs.append(int_freq)
    elif protocol == "MC":
        # Solve eigenval problem |K - w^2 D| = 0
        if D is None:
            raise TypeError("Protocol",protocol,"requires covariance mx D.")
        pass
    return freqs

def get_mass_matrix(trace, model, protocol="uncoupled"):
    if protocol == "coupled":
        from tnuts.mc.metrics import get_sample_cov
        # TODO: unit conversions
        return get_sample_cov(trace, model)[0]
    elif protocol == "uncoupled":
        return

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
    ec = nmode.get_average_energy(T)        # kcal/mol
    fc = nmode.get_helmholtz_free_energy(T) # kcal/mol
    sc = (ec-fc)/T * 1000                   # cal/mol.K
    var_e = nmode.get_energy_fluctuation(T) # (kcal/mol)^2
    cvc = np.power(R*T**2, -1)*var_e * 1000 # cal/mol.K
    return ec, sc, qc, cvc

#SCRATCH
#e = (e0 + hbar*w/(np.exp(beta*hbar*w)-1))*J2kcal*Na
#s = R*1000\
#        *(-np.log(1-np.exp(-hbar*w*beta)) +\
#        (hbar*w*beta)*np.exp(-hbar*w*beta)/(1-np.exp(-hbar*w*beta)))
