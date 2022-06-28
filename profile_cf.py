#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:44:21 2022

@author: Luis Antonio Souto Arias

@Software: Pycharm

Profile characteristic function of the Q-Hawkes and Hawkes processes.
"""
#%% Load modules

import sys
import numpy as np
import cProfile

from hawkes_class import Hawkes
from qhawkes_class import QHawkes

np.set_printoptions(threshold=sys.maxsize)

def prof_qhawk_hawk(cfe,cfh,u,v,N):
    for i in range(N):
        for j in u:
            for k in v:
                cfe(j,k)
                cfh(j,k)
    return 1
    

## Initialize parameters
a  = 2.9                   # Clustering rate
b  = 3.                    # Expiration rate
hb = 1                     # Baseline intensity
Q0 = 2                     # Initial memory
h0 = hb + a*Q0             # Initial intensity
T  = 0.1                   # Maturity
S0 = np.array([9,10,11])   # Asset's initial value
m  = -0.1                  # Mean of log-jump size
s2 = 0.1                   # Volatility of log-jump size

# Characteristic function and cumulants of the jump size distribution
eJ = np.exp(m+s2**2/2)-1
fu = lambda u: np.exp(1j*u*m-s2**2*u**2/2)
cm = [m,m**2+s2**2,m**3+3*m*s2**2,m**4+6*m**2*s2**2+3*s2**4]

qhawk  = QHawkes(a,b,hb,Q0,cm,eJ,fu)
hawkes = Hawkes(a,b,hb,Q0,cm,eJ,fu)

## Profile the characteristic functions

cme1 = qhawk.mean_cj(T).mean()
cme2 = qhawk.var_cj(T)
cme4 = 0
cme  = np.array([cme1,cme2,cme4])

# Interval [a,b]
L  = 7
a1 = cme[0] - L*np.sqrt(cme[1]+np.sqrt(cme[2]))
b1 = cme[0] + L*np.sqrt(cme[1]+np.sqrt(cme[2]))

# Number of Fourier cosine coefficients
N1 = 2**6
N2 = 2**0

a2 = -0.5
b2 = N2-0.5
k1 = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]
k2 = (np.arange(N2)*np.pi/(b2-a2))

cfe = lambda u,v: qhawk.cf_cj(u,v,T,Q0)
cfh = lambda u,v: hawkes.cf_cj(u,v,T,Q0)

cProfile.run('prof_qhawk_hawk(cfe,cfh,k1,k2,10)')
##

