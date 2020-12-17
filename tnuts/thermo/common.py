#!/usr/bin/env python3
import numpy as np
import rmgpy.constants as constants
from ape.common import get_internal_rotation_freq

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

def solvHO(w, T):
    """Solve quantum HO given frequency and temperature"""
    beta = 1./constants.kB/T
    ZPE = -constants.hbar*w / 2.
    Qzpe = np.exp(-beta*ZPE)
    _Qho = np.power(1-np.exp(-constants.hbar*w), -1)
    q = Qzpe / _Qho
    # TODO
    return ZPE, e, s, q, cv

def solvCHO(w, T):
    """Solve classical HO partition function"""
    beta = 1./constants.kB/T
    qc = np.power(beta * constants.hbar*w, -1)
    # TODO
    return ec, sc, cvc, qc

def solvUMClass(nmode, T):
    """Solve classical UM partition function for torsions"""
    beta = 1./constants.kB/T
    qt = np.sqrt(nmode.get_I()/(2*np.pi*beta))
    qv = nmode.get_classical_partition_fn(T)
    qc = qt * qv
    # TODO
    return ec, sc, cvc, qc
