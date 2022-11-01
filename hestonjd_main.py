#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:41:22 2022

@author: Luis Antonio Souto Arias

@Software: PyCharm

Pricing of European and Bermudan options with the Heston jump-diffusion model.
It can be used to replicate the results in our paper "A new self-exciting
jump-diffusion model for option pricing".
"""

## Load modules

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import py_vollib.black_scholes.implied_volatility as bs
import time
from scipy.interpolate import CubicSpline

import cos_method as COS
from hawkes_class import Hawkes
from qhawkes_class import QHawkes
from heston_class import Heston
from jumpdiff_class import JumpDiffusion
from poisson_class import Poisson
from pricing_functions import price_vanilla

np.set_printoptions(threshold=sys.maxsize)

## Characteristic function Bates model for Bermudan options

def bates_cf_bermudan(u,t,n,heston_cf_bermudan,poi_cf):
    cfV,vV = heston_cf_bermudan(u,t,n)
    return cfV*poi_cf(u,t),vV

## %% Initialize parameters
st = 'E'                   # Scenario
T  = 1                     # Maturity
N  = 10000                 # MC paths
S0 = np.array([9,10,11])   # Asset's initial value
K  = 10                    # Strike
X0 = np.log(S0/K)          # Normalized log-stock
r  = 0.1                   # Interest rate

# Diffusion parameters
V0   = 0.0625   # Initial value
rho  = 0.1      # Correlation
eta  = 0.9      # Vol of vol
lamb = 5        # Speed of mean-reversion
nu   = 0.16     # Long-term variance

# Jump parameters
a  = 2.                   # Clustering rate
b  = 3.                    # Expiration rate
hb = 1.1                   # Baseline intensity
Q0 = 2                     # Initial activation
h0 = hb + a*Q0             # Initial intensity
mj = -0.3                   # Mean of log-jump size
sj = 0.4                   # Volatility of log-jump size

# Save settings in external file
d = {'a': a,'b': b,'hb': hb,'Q0': Q0,'mj':mj,'sj':sj,
     'V0':V0,'rho':rho,'eta':eta,'lamb':lamb,'nu':nu,'r':r}
mysetting = pd.DataFrame(d,dtype=np.float32,index=[0])
mysetting.to_csv('setting'+st+'.csv',
                 index=False)

# Characteristic function and cumulants of the log-jump size distribution.
# Here we assume a normal distribution.
eJ = np.exp(mj+sj**2/2)-1
fu = lambda u: np.exp(1j*u*mj-sj**2*u**2/2)
cm = [mj,mj**2+sj**2,mj**3+3*mj*sj**2,mj**4+6*mj**2*sj**2+3*sj**4]

## Initialize the classes
qhawk    = QHawkes(a,b,hb,Q0,cm,eJ,fu)
hawk     = Hawkes(a,b,hb,Q0,cm,eJ,fu)
poi      = Poisson(a,b,hb,Q0,cm,eJ,fu)
heston   = Heston(S0[0],V0,r,lamb,nu,eta,rho)
heston_e = JumpDiffusion(heston,qhawk)
heston_h = JumpDiffusion(heston,hawk)
heston_p = JumpDiffusion(heston,poi)

## Compute the density with the COS method
x0 = np.array([0,V0,Q0])

# Heston-Q-Hawkes
cfx  = lambda u: heston_e.cf(u,0,0,T,x0)         # Characteristic function
cme  = heston_e.cumulants_cos(T)
cme[0] -= np.log(K)
L    = 5
ac   = cme[0]-L*np.sqrt(cme[1]+np.sqrt(cme[2]))  # Boundaries to plot
bc   = cme[0]+L*np.sqrt(cme[1]+np.sqrt(cme[2]))
xc   = np.linspace(ac,bc,100)
fx2  = COS.density(xc,cfx,cme)

# Heston-Hawkes
cfh  = lambda u: heston_h.cf(u,0,0,T,x0)
cmh  = heston_h.cumulants_cos(T)
cmh[0] -= np.log(K)
L    = 5
ac   = cmh[0]-L*np.sqrt(cmh[1]+np.sqrt(cmh[2]))
bc   = cmh[0]+L*np.sqrt(cmh[1]+np.sqrt(cmh[2]))
xh   = np.linspace(ac,bc,100)
fxh  = COS.density(xh,cfh,cmh)

## Plot the results
plt.figure()
plt.plot(xc,fx2,'r*',markersize=7)
plt.plot(xh,fxh,'bx',markersize=7)
plt.plot(xc,np.zeros_like(xc),'k--')
plt.xlabel('log-asset price',fontsize=16)
plt.ylabel('density function',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes'),fontsize=12)
plt.show()

## Price European put options
Kv  = S0[0]*np.linspace(0.8,1.2,21,endpoint=True)  # Strike grid
Tv  = np.linspace(0.1,2,20,endpoint=True)          # Maturity grid
nK  = Kv.size
nT  = Tv.size

Pq,IVq,Delta_q = price_vanilla(heston_e,S0[0],Tv,Kv,r,x0,alpha=-1)
Ph,IVh,Delta_h = price_vanilla(heston_h,S0[0],Tv,Kv,r,x0,alpha=-1)
Pp,IVp,Delta_p = price_vanilla(heston_p,S0[0],Tv,Kv,r,x0,alpha=-1)

## Plot the results for a particular strike and maturity

# Fixed strike
idK = 10
plt.figure()
plt.plot(Tv,IVq[:,idK],'r*')
plt.plot(Tv,IVh[:,idK],'bx')
plt.plot(Tv,IVp[:,idK],'g^')
plt.xlabel('Maturity',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12,loc='lower right')
plt.show()

plt.figure()
plt.plot(Tv,Delta_q[:,idK],'r*')
plt.plot(Tv,Delta_h[:,idK],'bx')
plt.plot(Tv,Delta_p[:,idK],'g^')
plt.xlabel('Maturity',fontsize=16)
plt.ylabel(r'Delta $\Delta$',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12,loc='lower right')
plt.show()

# Fixed maturity
idT = 9
plt.figure()
plt.plot(Kv,IVq[idT,:],'r*')
plt.plot(Kv,IVh[idT,:],'bx')
plt.plot(Kv,IVp[idT,:],'g^')
plt.xlabel('Strike',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

plt.figure()
plt.plot(Kv,Delta_q[idT,:],'r*')
plt.plot(Kv,Delta_h[idT,:],'bx')
plt.plot(Kv,Delta_p[idT,:],'g^')
plt.xlabel('Strike',fontsize=16)
plt.ylabel(r'Delta $\Delta$',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Price Bermudan put options
cfV     = lambda u,t,n: heston.cf_bermudan(u,Tv[idT],t,n)
cfPoi   = lambda u,t:   poi.cf_cj(u,0,t,Q0)
cfBates = lambda u,t,n: bates_cf_bermudan(u,t,n,cfV,cfPoi)
cfQ     = lambda u,t,n: qhawk.cf_bermudan(u,t,n)
cfH     = lambda u,t,n: hawk.cf_bermudan(u,t,n)

av,bv = heston.logvar_bounds(Tv[idT])   # Boundaries for the Heston variance

nM   = 10
M    = np.arange(1,nM+1)                # Number of exercise dates
Pq2  = np.zeros((nM,))                  # Prices
Pp2  = np.zeros_like(Pq2)
Ph2  = np.zeros_like(Pq2)
IVbe = np.zeros_like(Pq2)               # Implied volatilities
IVbh = np.zeros_like(Pq2)
IVbp = np.zeros_like(Pq2)

time_e = 0
time_h = 0
time_p = 0
nrep   = 1                              # Number of repetitions (for profiling purposes)
for k in range(nrep):
    for i in range(nM):
        # Heston-Q-Hawkes
        cme      = heston_e.cumulants_cos(Tv[idT]/M[i])
        cme[0]  -= np.log(Kv[idK])
        time1    = time.time()
        Pb,Vb,Qb = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cme,M[i],cfV,
                                       cfQ,N1=2**6,n1=2**5,n2=2**5)
        time_e += (time.time()-time1)
        Pq2[i]  = CubicSpline(Vb,Pb[:,Q0,:],axis=0)(V0)
        IVbe[i] = bs.implied_volatility(Pq2[i],S0[0],Kv[idK],Tv[idT],r,'p')
        #
        # Bates model
        cmp         = heston_p.cumulants_cos(Tv[idT]/M[i])
        cmp[0]     -= np.log(Kv[idK])
        time1       = time.time()
        Pbp,Vbp,Qbp = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cmp,M[i],cfBates,
                                          N1=2**6,n1=2**5)
        time_p += (time.time()-time1)
        Pp2[i]  = CubicSpline(Vbp,Pbp,axis=0)(V0)
        IVbp[i] = bs.implied_volatility(Pp2[i],S0[0],Kv[idK],Tv[idT],r,'p')

        # Heston-Hawkes
        cmh         = heston_h.cumulants_cos(Tv[idT]/M[i])
        cmh[0]     -= np.log(Kv[idK])
        time1       = time.time()
        Pbh,Vbh,Qbh = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cmh,M[i],cfV,
                                          cfH,N1=2**6,n1=2**5,n2=2**5)
        time_h += (time.time()-time1)
        Pbh     = CubicSpline(Qbh,Pbh,axis=1)
        Ph2[i]  = CubicSpline(Vbh,Pbh(Q0))(V0)
        IVbh[i] = bs.implied_volatility(Ph2[i],S0[0],Kv[idK],Tv[idT],r,'p')

print(time_e/nrep,time_h/nrep,time_p/nrep)

## Plot the results
plt.figure()
plt.plot(M,IVbe,'r*')
plt.plot(M,IVbh,'bx')
plt.plot(M,IVbp,'g^')
plt.xlabel('Exercise dates',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Sensitivity analysis alpha
av  = b*np.linspace(0.01,0.999,20,endpoint=True)
na  = av.size

Pae,IVae = heston_e.sens_an(S0[0],T,S0[0],av,'a')
Pah,IVah = heston_h.sens_an(S0[0],T,S0[0],av,'a')

## Plot the results
plt.figure()
plt.plot(av,IVae,'r*')
plt.plot(av,IVah,'bx')
plt.xlabel(r'Clustering rate $\alpha$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12,loc='lower right')
plt.show()

## Sensitivity analysis beta
bv  = a*np.linspace(1.01,10,20,endpoint=True)
nb  = bv.size

Pbe,IVbe = heston_e.sens_an(S0[0],T,S0[0],bv,'b')
Pbh,IVbh = heston_h.sens_an(S0[0],T,S0[0],bv,'b')

## Plot the results
plt.figure()
plt.plot(bv,IVbe,'r*')
plt.plot(bv,IVbh,'bx')
plt.xlabel(r'Expiration rate $\beta$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Sensitivity analysis hb
hbv  = np.linspace(0.05,5,20,endpoint=True)
nhb  = hbv.size

Phbe,IVhbe = heston_e.sens_an(S0[0],T,S0[0],hbv,['hb'])
Phbh,IVhbh = heston_h.sens_an(S0[0],T,S0[0],hbv,['hb'])

## Plot the results
plt.figure()
plt.plot(hbv,IVhbe,'r*')
plt.plot(hbv,IVhbh,'bx')
plt.xlabel(r'Baseline intensity $\lambda^*$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Sensitivity analysis Q0
Q0v  = np.linspace(0.05,5,20,endpoint=True)
nQ0  = Q0v.size

PQ0e,IVQ0e = heston_e.sens_an(S0[0],T,S0[0],Q0v,['Q0'])
PQ0h,IVQ0h = heston_h.sens_an(S0[0],T,S0[0],Q0v,['Q0'])

## Plot the results
plt.figure()
plt.plot(Q0v,IVQ0e,'r*')
plt.plot(Q0v,IVQ0h,'bx')
plt.xlabel(r'Initial value $Q_0$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Sensitivity analysis mj
mjv = np.linspace(-2,1,20,endpoint=True)
nmj = mjv.size
eJv = np.exp(mjv+sj**2/2)-1
cmv = np.array([mjv,mjv**2+sj**2,mjv**3+3*mjv*sj**2,mjv**4+6*mjv**2*sj**2+3*sj**4]).T
mjlist = []
fui = [lambda u,m=m: np.exp(1j*u*m-sj**2*u**2/2) for m in mjv]
for i in range(nmj):
    mjlist.append([cmv[i],eJv[i],fui[i]])
mjlist = np.array(mjlist,dtype='object')

Pmje,IVmje = heston_e.sens_an(S0[0],T,S0[0],mjlist,['mJ','eJ','cfJ'])
Pmjh,IVmjh = heston_h.sens_an(S0[0],T,S0[0],mjlist,['mJ','eJ','cfJ'])

## Plot the results
plt.figure()
plt.plot(mjv,IVmje,'r*')
plt.plot(mjv,IVmjh,'bx')
plt.xlabel(r'Jump size expectation $\mu_Y$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Sensitivity analysis sj
sjv = np.linspace(0,0.4,20,endpoint=True)
mjv = np.ones_like(sjv)*mj
nsj = sjv.size
eJv = np.exp(mjv+sjv**2/2)-1
cmv = np.array([mjv,mjv**2+sjv**2,mjv**3+3*mjv*sjv**2,mjv**4+6*mjv**2*sjv**2+3*sjv**4]).T
mjlist = []
fui = [lambda u,s=s: np.exp(1j*u*mj-s**2*u**2/2) for s in sjv]
for i in range(mjv.size):
    mjlist.append([cmv[i],eJv[i],fui[i]])
mjlist = np.array(mjlist,dtype='object')

Psje,IVsje = heston_e.sens_an(S0[0],T,S0[0],mjlist,['mJ','eJ','cfJ'])
Psjh,IVsjh = heston_h.sens_an(S0[0],T,S0[0],mjlist,['mJ','eJ','cfJ'])

## Plot the results
plt.figure()
plt.plot(sjv,IVsje,'r*')
plt.plot(sjv,IVsjh,'bx')
plt.xlabel(r'Jump size volatility $\sigma_Y$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

##

