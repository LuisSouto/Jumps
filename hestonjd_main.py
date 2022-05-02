#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:41:22 2022

@author: luis_souto

Pricing of European and Bermudan options with the Heston jump-diffusion model.
"""

## %% Load modules

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as colorm
import py_vollib.black_scholes.implied_volatility as bs
import time
from scipy.optimize import newton
from scipy.interpolate import CubicSpline


import cos_method as COS
from hawkes_class import Hawkes
from esep_class import ESEP
from heston_class import Heston
from hestonjd_class import HestonJD
from poisson_class import Poisson

np.set_printoptions(threshold=sys.maxsize)

## %% Initialize parameters
T  = 1                     # Maturity
nT = 300
N  = 50000                # MC paths
S0 = np.array([9,10,11])   # Asset's initial value
K  = 10                    # Strike
X0 = np.log(S0/K)          # Normalized log-stock
r  = 0.1                  # Interest rate

# Diffusion parameters
V0   = 0.0625
rho  = 0.7
eta  = 0.2     # Vol of vol
lamb = 5       # Speed of mean-reversion
nu   = 0.16    # Long-term variance

# V0   = 0.0348
# rho  = -0.64
# eta  = 0.39
# lamb = 1.15
# nu   = 0.0348

# Jump parameters
a  = 2.                   # Intensity jump size
b  = 3.                    # Memory kernel rate
hb = 1                     # Intesity baseline
Q0 = 2                     # Initial memory
h0 = hb + a*Q0             # Initial intensity
mj = 0.3                  # Mean of jump size
sj = 0.4                   # Volatility of jump size

# Characteristic function and cumulants of the jump size distribution
eJ = np.exp(mj+sj**2/2)-1
fu = lambda u: np.exp(1j*u*mj-sj**2*u**2/2)
cm = [mj,mj**2+sj**2,mj**3+3*mj*sj**2,mj**4+6*mj**2*sj**2+3*sj**4]

## %% Simulate the processes
esep = ESEP(a,b,hb,Q0,cm,eJ,fu)
hawk = Hawkes(a,b,hb,Q0,cm,eJ,fu)
heston = Heston(S0[0],V0,r,lamb,nu,eta,rho)
heston_e = HestonJD(heston,esep)
heston_h = HestonJD(heston,hawk)

J     = np.random.standard_normal((N,))
Tx,B  = esep.simul(N,T)
Th,Bh = hawk.simul(N,T)
Ne,Qe = esep.compute_intensity(T,Tx,B)
Nh,Ih = hawk.compute_intensity(T,Th,Bh)
Nh = Nh.squeeze()
S,V   = heston.simul(N,nT,T)
S -= np.log(K)
Mq = Ne*mj+sj*J*np.sqrt(Ne)-eJ*a*((Tx<T)*((T-Tx)*B)).sum(0)-eJ*h0*T
Sj = S[:,-1]+Mq
# Sj,V,Qe = heston_e.simul(N,nT,T)
# Sj -= np.log(K)
Mh = (-eJ*hb*T+eJ/b*(h0-hb)*(np.exp(-b*T)-1)
      +eJ*a/b*(Bh*(Th<T)*(np.exp(-b*(T-Th))-1)).sum(0)+Nh*mj+sj*J*np.sqrt(Nh))
Sh = S[:,-1]+Mh

## %% Compute the density with the COS method
x0 = np.array([X0[0],V0,Q0])
# Q-Hawkes jump
cfx  = lambda u: heston_e.cf(u,0,0,T,x0)
cme1 = heston_e.mean(T).mean()-np.log(K)
cme2 = heston_e.var(T)
cme4 = 0 # heston_e.cumulant4(T)
cme  = np.array([cme1,cme2,cme4])
L    = 5
ac   = cme[0]-L*np.sqrt(cme[1]+np.sqrt(cme[2]))
bc   = cme[0]+L*np.sqrt(cme[1]+np.sqrt(cme[2]))
xc   = np.linspace(ac,bc,500)
fx2  = COS.density(xc,cfx,cme)

# Hawkes jump
cfh  = lambda u: heston_h.cf(u,0,0,T,x0)
cmh1 = heston_h.mean(T).mean()-np.log(K)
cmh2 = heston_h.var(T)
cmh4 = 0 # heston_e.cumulant4(T)
cmh  = np.array([cmh1,cmh2,cmh4])
L    = 5
ac   = cmh[0]-L*np.sqrt(cmh[1]+np.sqrt(cmh[2]))
bc   = cmh[0]+L*np.sqrt(cmh[1]+np.sqrt(cmh[2]))
xh   = np.linspace(ac,bc,500)
fxh  = COS.density(xh,cfh,cmh)

## %% Plot the results
plt.figure()
# plt.hist(Se[0,Qe==id],xc,density=True)
# plt.plot(xc,fx2[id,:]/fx[id])
plt.hist(S[:,-1],xc,density=True,alpha=0.5)
plt.hist(Sj,xc,density=True,alpha=0.5)
plt.hist(Sh,xc,density=True,alpha=0.5)
plt.plot(xc,fx2)
plt.plot(xh,fxh)
plt.legend(('Q-Hawkes (COS)','Hawkes (COS)','Pure Heston','Heston-ESEP'))
plt.show()

## Price European options
x0    = np.array([0,V0,Q0])
Kv    = S0[0]*np.arange(0.8,1.25,0.05)
Tv    = np.array([0.1,0.25,0.5,0.75,1,2])
nK    = Kv.size
nT    = Tv.size
P     = np.zeros((nT,nK))
Ppoi  = np.zeros_like(P)
Ph    = np.zeros_like(P)
IVe   = np.zeros_like(P)
IVh   = np.zeros_like(P)
IVp   = np.zeros_like(P)

poi      = Poisson(a,b,hb,Q0,cm,eJ,fu)
heston_p = HestonJD(heston,poi)
for i in range(nT):
    for j in range(nK):
        # ESEP prices
        cfe = lambda u: heston_e.cf(u,0,0,Tv[i],x0)

        # Cumulants
        cm1 = heston_e.mean(Tv[i])-np.log(Kv[j])
        cm2 = heston_e.var(Tv[i])
        cm4 = 0 # heston_e.cumulant4(Tv[i])

        # Implied volatility
        P[i,j] = COS.put(S0[0],Kv[j],Tv[i],r,cfe,[cm1,cm2,cm4])
        IVe[i,j] = bs.implied_volatility(P[i,j],S0[0],Kv[j],Tv[i],r,'p')

        # Poisson jump-diffusion
        cfp  = lambda u: heston_p.cf(u,0,0,Tv[i],x0)
        cm1  = heston_p.mean(Tv[i])-np.log(Kv[j])
        cm2  = heston_p.var(Tv[i])
        cm4  = 0 # heston_p.cumulant4(Tv[i])

        # Implied volatility
        Ppoi[i,j] = COS.put(S0[0],Kv[j],Tv[i],r,cfp,[cm1,cm2,cm4])
        IVp[i,j] = bs.implied_volatility(Ppoi[i,j],S0[0],Kv[j],Tv[i],r,'p')

        # Hawkes jump-diffusion
        cfh = lambda u: heston_h.cf(u,0,0,Tv[i],x0)

        # Cumulants
        cm1 = heston_h.mean(Tv[i])-np.log(Kv[j])
        cm2 = heston_h.var(Tv[i])
        cm4 = 0 # heston_h.cumulant4(Tv[i])

        # Implied volatility
        Ph[i,j] = COS.put(S0[0],Kv[j],Tv[i],r,cfh,[cm1,cm2,cm4])
        IVh[i,j] = bs.implied_volatility(Ph[i,j],S0[0],Kv[j],Tv[i],r,'p')

## %% Plot the results
idK = -1
plt.figure()
plt.plot(Tv,IVe[:,idK])
plt.plot(Tv,IVh[:,idK])
plt.plot(Tv,IVp[:,idK])
plt.xlabel('Maturity',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Queue-H','Hawkes','Poisson'),fontsize=12)
plt.show()

idT = -1
plt.figure()
plt.plot(Kv,IVe[idT,:])
plt.plot(Kv,IVh[idT,:])
plt.plot(Kv,IVp[idT,:])
plt.xlabel('Strike',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Queue-H','Hawkes','Poisson'),fontsize=12)
plt.show()

## Reconstruct the density with the COS method
qj  = np.arange(2**6)
vj  = np.linspace(-15,-2,300)
vj  = np.linspace(1e-6,0.3,300)
Si,Vi = np.meshgrid(xc,vj)
cfS = lambda u: (heston.cfcondv_dens(u,T,X0[0],vj[:,np.newaxis,np.newaxis],V0)
                 *esep.cf_integral(u,qj,T,Q0,vectorize=False))
# cfS = lambda u: (np.exp(1j*u*(X0[0]+(r-0.5*V0**2)*T)-u**2*T*V0**2/2)
#                  *esep.cf_integral(u,Q0,T,Q0,vectorize=False))
fSV = COS.density(xc,cfS,cme)
print(fSV.shape)
logV = np.log(V[:,-1])
hval = plt.hist2d(Sj,V[:,-1],100,density=True)
Xedg,Yedg = np.meshgrid(hval[1][:-1],hval[2][:-1])
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Xedg,Yedg,hval[0].T,cmap=colorm.coolwarm,alpha=0.5,
                linewidth=0,antialiased=True)
ax.plot_surface(Si,Vi,fSV.sum(1),cmap=colorm.Spectral,alpha=0.5,
                linewidth=0,antialiased=True)
plt.show()


##
plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(Si,Vi,fSV.sum(1),cmap=colorm.Spectral,
                linewidth=0,antialiased=False)
plt.show()


## Test Bermudan
M = 3
cfV = lambda u,t,vt,vs: heston.cfcondv_dens(u,t,0,vt,vs)
cfQ = lambda u,t,qt,qs: esep.cf_integral(u,qt,t,qs,vectorize=True)

tola = 1e-8      # Bound on left-tail
tolb = 1e-6      # Bound on right-tail
mV   = np.log(V0*np.exp(-lamb*T)+nu*(1-np.exp(-lamb*T)))
q    = 2*lamb*nu/eta**2-1
c    = 2*lamb/((1-np.exp(-lamb*T))*eta**2)
av   = mV - 5/(1+q)
bv   = mV + 1/(1+q)

# Density of log-variance and its first derivative
fv   = lambda v: heston.dens_logvar(T,np.exp(v),V0)
fv_v = lambda v: heston.dens_logvar_dv(T,np.exp(v),V0)

while(np.abs(fv(av))>tola):
    av -= fv(av)/fv_v(av)

while(np.abs(fv(bv))>tolb):
    bv -= fv(bv)/fv_v(bv)

Vb,Qb,Pb = COS.bermudan_put_3D(S0,Kv[-3],T,r,cme,av,bv,cfV,cfQ,M)
Pb = CubicSpline(Vb,Pb[:,Q0,:],axis=0)

plt.figure()
plt.plot(Vb,Pb(Vb))
plt.show()

print(Pb(V0))

## Price Bermudan
# M  = np.arange(1,11) # Exercise dates
# nM = M.size
# P2     = np.zeros((nM,))
# Ppoi2  = np.zeros((nM,))
# Ph2    = np.zeros((nM,))
# for i in range(nM):
#     # ESEP prices
#     cfe = lambda u,v,w,t,x: heston_e.cf(u,v,w,t,x)
#
#     # Cumulants
#     cm1 = heston_e.mean(T)-np.log(K)
#     cm2 = heston_e.var(T)
#     cm4 = 0
#     cms = np.array([cm1,cm2,cm4])
#
#     # COS method calls y puts
#     Qe,Pe = COS.bermudan_put_nolevy_v2(S0,K,T,r,cms,0,0,cfe,M[i],jump=True)
#     P2[i] = Pe[Q0]
#
#     # Poisson jump-diffusion
#     cfp  = lambda u,v,t,Q: np.tile(pjd.cf(u,0,t,1,Q0),(Q.size,1,v.size))
#
#     cm1  = pjd.mean(T)-np.log(K)
#     cm2  = pjd.var(T)
#     cm4  = pjd.cumulant4(T)
#     cms = np.array([cm1,cm2,cm4])
#
#     Qe,Pe = COS.bermudan_put_nolevy_v2(S0,K,T,r,cms,0,0,cfp,M[i],jump=True)
#     Ppoi2[i] = Pe[Q0]
#
#     # Hawkes jump-diffusion
#     cfh = lambda u,v,t,Q: hjd.cf(u,v,t,1,Q)
#
#     # Cumulants
#     cm1 = hjd.mean(T)-np.log(K)
#     cm2 = hjd.var(T)
#     cm4 = hjd.cumulant4(T)
#     cms = np.array([cm1,cm2,cm4])
#
#     # COS method calls y puts
#     ah  = 0
#     bh  = 2**5
#     Qh,Pbh = COS.bermudan_put_nolevy_v2(S0,K,T,r,cms,ah,bh,cfh,M[i],jump=False)
#     Pbh = CubicSpline(Qh,Pbh,axis=0)
#     Ph2[i]  = Pbh(Q0)
#
