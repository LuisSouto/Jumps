#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:44:21 2022

@author: luis_souto

Profile characteristic function
"""
#%% Load modules

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import re
import pstats

import cos_method as COS
import gbmjd_class as ajd
from hawkes_class import Hawkes
from qhawkes_class import ESEP
from scipy.interpolate import CubicSpline

np.set_printoptions(threshold=sys.maxsize)

def funcprof(cf,t,N):
    for i in range(N):
        cf(t)
    return 1

def prof_esep_hawk(cfe,cfh,u,v,N):
    for i in range(N):
        for j in u:
            for k in v:
                cfe(j,k)
                cfh(j,k)
    return 1
    

#%% Initialize parameters
a  = 2.9                   # Intensity jump size
b  = 3.                    # Memory kernel rate
hb = 1                    # Intesity baseline
Q0 = 0                    # Initial memory
h0 = hb + a*Q0             # Initial intensity
T  = 0.1                     # Maturity
N  = 100000                # MC paths
S0 = np.array([9,10,11])   # Asset's initial value
K  = 10                    # Strike
X0 = np.log(S0/K)          # Normalized log-stock
r  = 0.1                  # Interest rate
s  = 0.3                   # Volatility brownian motion
m  = -0.1                  # Mean of jump size
s2 = 0.1                   # Volatility of jump size
M  = 1

# Characteristic function and cumulants of the jump size distribution
eJ = np.exp(m+s2**2/2)-1
fu = lambda u: np.exp(1j*u*m-s2**2*u**2/2);
cm = [m,m**2+s2**2,m**3+3*m*s2**2,m**4+6*m**2*s2**2+3*s2**4]

esep = ESEP(a,b,hb,Q0,cm,eJ,fu)
ejd  = ajd.GBMJD(S0,r,s,esep)

hawkes = Hawkes(a,b,hb,Q0,cm,eJ,fu)
hjd    = ajd.GBMJD(S0,r,s,hawkes)

#%% Price Bermudan Put

cme1 = ejd.mean(T).mean()-np.log(K)
cme2 = ejd.var(T)
cme4 = 0 #ejd.cumulant4(T)
cme  = np.array([cme1,cme2,cme4])
cfe  = lambda u,v,t,Q: ejd.cf(u,v,t,1,Q)

# Time step
dt = T/M

# Interval [a,b]
L = 7
a1 = cme[0] - L*np.sqrt(cme[1]+np.sqrt(cme[2]))
b1 = cme[0] + L*np.sqrt(cme[1]+np.sqrt(cme[2]))

# Number of Fourier cosine coefficients
N1 = 2**6
N2 = 2**6
J  = 2**4

a2 = -0.5
b2 = N2-0.5
# vi = np.tile(np.arange(2**4),(2**4))
vi = np.arange(N2)
wi = np.ones((N2,))
k1  = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]
k2  = (np.arange(N2)*np.pi/(b2-a2))
wi /= (b2-a2)

cf = lambda t: esep.cf_cj(0,k2,t,vi[:,np.newaxis,np.newaxis])
cfe = lambda u,v: esep.cf_cj(u,v,T,Q0)
cfh = lambda u,v: hawkes.cf_cj(u,v,T,Q0)

# cProfile.run('esep.cf_cj(k1,k2,T,vi[:,np.newaxis,np.newaxis])')
cProfile.run('prof_esep_hawk(cfe,cfh,k1,k2,100)')

# profiler = cProfile.Profile()
# profiler.enable()
# esep.cf_cj(k1,k2,T,vi[:,np.newaxis,np.newaxis])
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats('ncalls')
# stats.print_stats()