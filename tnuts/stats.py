#!/usr/bin/env python3
"""
Useful statistics
Used to parse QChem output files when APE won't do the job
"""
import numpy as np
import math
import pandas as pd
import os
import rmgpy.constants as constants

def create_dfs(dirname, sampT=300):
    Edict = {}
    phidict = {}
    adict = {}
    for f in os.listdir(dirname):
        if not '.npy' in f:
            continue
        if 'E' in f:
            with open(os.path.join(dirname,f),'rb') as npy:
                Edict[f[:-4]] = np.load(npy)
        elif 'phi' in f:
            with open(os.path.join(dirname,f),'rb') as npy:
                phidict[f[:-4]] = np.load(npy)
        elif 'a' in f:
            with open(os.path.join(dirname,f),'rb') as npy:
                adict[f[:-4]] = np.load(npy)

    # RAW DATA FRAMES ACROSS ALL CHAINS
    Edf = pd.DataFrame(Edict).replace(0.000000, np.nan)
    adf = pd.DataFrame(adict)
    DEdf = np.log(adf)
    Edf.to_csv(os.path.join(dirname,'Edf.csv'))
    adf.to_csv(os.path.join(dirname,'adf.csv'))
    DEdf.to_csv(os.path.join(dirname, 'DEdf.csv'))

    # CUMULATIVE MEAN DATA FRAMES ACROSS ALL CHAINS
    Ecm = Edf.expanding(axis=0).mean(skipna=True)
    acm = adf.expanding(axis=0).mean(skipna=True)
    DEcm = DEdf.expanding(axis=0).mean(skipna=True)
    qcm = np.exp(-Edf).expanding(axis=0).mean(skipna=True)
    Ecm.to_csv(os.path.join(dirname,'Ecm.csv'))
    acm.to_csv(os.path.join(dirname,'acm.csv'))
    qcm.to_csv(os.path.join(dirname,'qcm.csv'))
    DEcm.to_csv(os.path.join(dirname,'DEcm.csv)'))
    
    s = pd.DataFrame()
    for it in Edf.index:
        miniEdf = Edf[:(it+1)]
        pi = np.exp(-miniEdf)/np.exp(-miniEdf).sum()
        s = s.append(-(pi*np.log(pi)).sum(),
                ignore_index=True)
    s.to_csv(os.path.join(dirname,'s.csv'))

    print(Ecm.iloc[-1])
    print(np.mean(Ecm.iloc[-1]))
    #Ecm.plot()
    #plt.show()
    #adf.plot()
    #plt.show()

    # CUMULATIVE VARIANCE DATA FRAMES ACROSS ALL CHAINS
    betasamp = 1/constants.kB/sampT * constants.E_h
    Ecv = (Edf/betasamp).expanding(axis=0).var(skipna=True)
    acv = adf.expanding(axis=0).var(skipna=True)
    Ecv.to_csv(os.path.join(dirname,'Ecv.csv'))
    acv.to_csv(os.path.join(dirname,'acv.csv'))

    # CUMULATIVE VARIANCE OF VARIANCE ACROSS ALL CHAINS
    

    # CUMULATIVE beta ENERGY VARIANCE
    bEcv = Edf.expanding(axis=0).var(skipna=True)

    #Ecv.plot()
    #plt.show()
    return acm, Ecm, Ecv, DEcm, s
    return Edf, adf, phidict
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    acm, bEcm, Ecvar = create_dfs(
            os.path.expandvars('$SCRATCH/results/m1/NUTS/umvt'), 300)
    #edf, adf, phidict = create_dfs('/Users/lancebettinson/scratch/test-results/propanoic_hf/NUTS/umvt')
