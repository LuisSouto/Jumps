#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:52:22 2022

@author: luis_souto

Study of the interarrival times of the Q-Hawkes process.
"""

## %% Load modules

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import cos_method as COS
from esep_class import ESEP
from scipy.special import binom

np.set_printoptions(threshold=sys.maxsize)

## %% Initialize parameters
st = 'E'
T  = 4                     # Maturity
nT = 300
N  = 50000                # MC paths

# Jump parameters
a  = 1.                   # Intensity jump size
b  = 2.                    # Memory kernel rate
hb = 1.5                     # Intesity baseline
Q0 = 1                     # Initial memory
h0 = hb + a*Q0             # Initial intensity
mj = 1                  # Mean of jump size
sj = 0.0                   # Volatility of jump size

# Characteristic function and cumulants of the jump size distribution
eJ = np.exp(mj+sj**2/2)-1
fu = lambda u: np.exp(1j*u*mj-sj**2*u**2/2)
cm = [mj,mj**2+sj**2,mj**3+3*mj*sj**2,mj**4+6*mj**2*sj**2+3*sj**4]

## %% Simulate the processes
esep  = ESEP(a,b,hb,Q0,cm,eJ,fu)
J     = np.random.standard_normal((N,))
Tx,B  = esep.simul(N,T)
Ne,Qe = esep.compute_intensity(T,Tx,B)

## Get the kth arrival time
kth = 0
Te = Tx*(B>0)
Te[Te<=0] = 1.1*T    # This ensures larger values than the actual arrival times
# Te1 = np.partition(Te,kth,axis=0)[kth]
Te2 = np.partition(Te,(kth,kth+1),axis=0)[kth:kth+2]
Te1 = Te2[0]
Te12 = Te2[1,(Te2[0]<=T)]-Te2[0,(Te2[0]<=T)]
Te1 = Te1[Te1<=T]
# Te12 = Te12[(Te12+Te1)<=T]

## Histogram of first arrival time
t0 = 1.1*T
nt = 100
t  = np.linspace(0,t0,nt)
dt = t[1]-t[0]
Te1_counts  = np.cumsum(np.histogram(Te1,t,density=True)[0])
Te1_norm    = Te1.size*dt/N
Te12_counts = np.cumsum(np.histogram(Te12,t,density=True)[0])
Te12_norm   = Te12.size*dt/N

kQ = np.arange(Q0+1)[:,np.newaxis]
wQ = np.array([hb+a,b])[:,np.newaxis]/(hb+a+b)
# fT = lambda t: (binom(Q0,kQ)*(a/b)**kQ*(b/(a+b))**Q0*esep.inter_time_dist(t,kQ+1)).sum(0)
fT = lambda t: (wQ*esep.inter_time_dist(t,kQ+1)).sum(0)

plt.figure()
plt.plot(t,esep.inter_time_dist(t,Q0),'r*')
plt.plot(t,fT(t),'g^')
plt.bar(t[:-1],Te12_counts*Te12_norm,width=dt)
plt.legend((r'$T_1$ theory','$T_{1,2}$ theory','Empirical'),fontsize=12)
plt.xlabel('Time',fontsize=16)
plt.ylabel('Cumulative distribution',fontsize=16)
plt.xlim([0,T])
plt.show()


##

