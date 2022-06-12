#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:32:34 2022

Pricing of Bermudan options with the ESEP jump-diffusion

@author: luis_souto
"""

## %% Load modules

import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile


import cos_method as COS
import gbmjd_class as ajd
from hawkes_class import Hawkes
from poisson_class import Poisson
from esep_class import ESEP
from scipy.interpolate import CubicSpline

np.set_printoptions(threshold=sys.maxsize)

# %% Profiling function for density
def prof_func(f,t,N=100):
    for i in range(N):
        f(t)

## %% Initialize parameters
a  = 2.9                   # Intensity jump size
b  = 3.                    # Memory kernel rate
hb = 1.1                    # Intensity baseline
Q0 = 2                    # Initial memory
h0 = hb + a*Q0             # Initial intensity
T  = 1                     # Maturity
N  = 20000                # MC paths
S0 = np.array([9,10,11])   # Asset's initial value
K  = 10                    # Strike
X0 = np.log(S0/K)          # Normalized log-stock
r  = 0.1                  # Interest rate
s  = 0.3                   # Volatility brownian motion
m  = -0.1                  # Mean of jump size
s2 = 0.1                   # Volatility of jump size
st = 'B'

# Characteristic function and cumulants of the jump size distribution
eJ = np.exp(m+s2**2/2)-1
fu = lambda u: np.exp(1j*u*m-s2**2*u**2/2)
cm = [m,m**2+s2**2,m**3+3*m*s2**2,m**4+6*m**2*s2**2+3*s2**4]

## %% Simulate the processes
np.random.seed(5)
# ESEP jump times
esep = ESEP(a,b,hb,Q0,cm,eJ,fu)
Tx,B = esep.simul(N,T)

# Hawkes jump times 
hawkes = Hawkes(a,b,hb,Q0,cm,eJ,fu)
Th,Bh  = hawkes.simul(N,T)

# Poisson counts
hp = (b*hb/(a-b)+h0)*np.exp((a-b)*T)-b*hb/(a-b)
Np = np.random.poisson(hp*T,(N,))

# Brownian motion and jump sizes
W = np.random.standard_normal((N,))
J = np.random.standard_normal((N,))

# Hawkes process
t = np.linspace(0,T,100,endpoint=True)
Nh,Ih = hawkes.compute_intensity(t,Th,Bh)
Ne,Qe = esep.compute_intensity(t,Tx,B)
Ie = hb + a*Qe

# ESEP process
Ne,Qe = esep.compute_intensity(T,Tx,B)
Se = (X0[:,np.newaxis]+(r-eJ*h0-s**2/2)*T+s*np.sqrt(T)*W
      +Ne*m+s2*J*np.sqrt(Ne)-eJ*a*((Tx<T)*((T-Tx)*B)).sum(0))

## %% Plot intensities
id = 22
plt.figure()
plt.plot(t,Ie[id,:],'r*-')
plt.plot(t,Ih[:,id],'bx-')
# plt.plot(Th[:,id],hb*Bh[:,id],'ko')
plt.xlabel('Time',fontsize=16)
plt.ylabel('Intensity',fontsize=16)
plt.xlim([0,T])
# plt.ylim([0.99*hb,1.05*Ih[:,id].max()])
plt.title(r'Scenario '+st,fontsize=20)
plt.legend(('Q-Hawkes','Hawkes'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation ICCF2022 Wuppertal/figures/intensities'+st+'.png')
plt.show()
##
id = 6
plt.figure()
plt.plot(t,Ie[id,:])
plt.plot(Tx[:,id]*(B[:,id]>0),hb*(B[:,id]>0),'ko')
plt.xlabel('Time',fontsize=16)
plt.ylabel('Intensity',fontsize=16)
plt.xlim([0,T])
plt.ylim([0.99*hb,1.05*Ie[id,:].max()])
plt.title(r'Set '+st,fontsize=20)
plt.legend(('Intensity','Jumps'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation Industry Week/figures/ese_intensity'+st+'.png')
plt.show()


## %% Compare the simulated density to the one from the COS method

x   = np.arange(25)
cfq = lambda u: esep.cf(u,T,Q0)
fx  = COS.density_dis(x, cfq)
fxs = (Qe==x[:,np.newaxis]).mean(-1)

plt.figure()
plt.plot(x,fxs)
plt.plot(x,fx)
plt.legend(('Simulation','COS method'))
plt.show()

print((x*fxs).sum(),(x*fx).sum())
print((x*(x-1)*fxs).sum(),(x*(x-1)*fx).sum())

## %% Compare with the Hawkes
xh = np.linspace(0.0,25,40)
cfh = lambda u: hawkes.cf_int(u,T,Q0)
cmh1 = (Ih[-1,:].mean()-hb)/a
cmh2 = Ih[-1,:].var()/a**2
cmh4 = 0
cmh = np.array([cmh1,cmh2,cmh4])
fh  = COS.density(xh,cfh,cmh,N=2**8)

plt.figure()
plt.plot(x,fx,'r*')
# plt.plot(xh,fh,'bx')
plt.hist((Ih[-1,:]-hb)/a,xh,density=True)
plt.xlabel('Intensity',fontsize=16)
plt.ylabel('Density',fontsize=16)
plt.title(r'Scenario '+st,fontsize=20)
plt.legend(('Queue-H','Hawkes'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation ICCF2022 Wuppertal/figures/density'+st+'.png')
plt.show()

## %% Compare the bivariate density to the COS method

ejd  = ajd.GBMJD(S0,r,s,esep)
cfx  = lambda u,v: ejd.cf(u,v,T,S0[0]/K,Q0)
cme1 = ejd.mean(T).mean()-np.log(K)
cme2 = ejd.var(T)
cme4 = 0 #ejd.cumulant4(T)
cme  = np.array([cme1,cme2,cme4])
L    = 5
ac   = cme[0]-L*np.sqrt(cme[1]+np.sqrt(cme[2]))
bc   = cme[0]+L*np.sqrt(cme[1]+np.sqrt(cme[2]))
xc   = np.linspace(ac,bc,500)
time1 = time.time()
fx2  = COS.density_2D_dis(xc, x, cfx, cme)
time2 = time.time()-time1
print(time2)

cf3 = lambda u: (np.exp(1j*u*(X0[0]+(r-0.5*s**2)*T)-u**2*T*s**2/2)
                 *esep.cf_integral(u,x,T,Q0,vectorize=True))
time1 = time.time()
fx3 = COS.density(xc,cf3,cme)
time2 = time.time()-time1
print(time2)

# u = np.arange(2**6)/(b-a)*np.pi
# pfun = lambda t: esep.cf_integral(u,id,t,Q0,vectorize=True)
# cProfile.run('prof_func(pfun,T,5000)')

## Plot
id  = 9
plt.figure()
plt.hist(Se[0,Qe==id],xc,density=True)
plt.plot(xc,fx2[id,:]/fx[id])
plt.plot(xc,fx3[Q0,id,:]/fx[id])
# plt.hist(Se[0,:],xc,density=True)
# plt.plot(xc,fx2.sum(0))
plt.legend(('COS method','Simulation'))
plt.show()

## %% Price European put
Kv    = S0[1]*np.arange(0.8,1.25,0.05)
Tv    = np.array([0.1,0.25,0.5,0.75,1,2])
nK    = Kv.size
nT    = Tv.size
P     = np.zeros((nT,nK))
Ppoi  = np.zeros((nT,nK))
Ph    = np.zeros((nT,nK))

ejd  = ajd.GBMJD(S0[1],r,s,esep)
hjd  = ajd.GBMJD(S0[1],r,s,hawkes)
poi  = Poisson(a,b,hb,Q0,cm,eJ,fu)
pjd  = ajd.GBMJD(S0[1],r,s,poi)
for i in range(nT):

    for j in range(nK):
        # ESEP prices
        cfe = lambda u: ejd.cf(u,0,Tv[i],1,Q0)

        # Cumulants
        cm1 = ejd.mean(Tv[i])-np.log(Kv[j])
        cm2 = ejd.var(Tv[i])
        cm4 = ejd.cumulant4(Tv[i])

        # COS method calls y puts
        P[i,j]  = COS.vanilla(S0[1],Kv[j],Tv[i],r,cfe,[cm1,cm2,cm4],-1)

        # Poisson jump-diffusion
        cfp  = lambda u: pjd.cf(u,0,Tv[i],1,Q0)

        cm1  = pjd.mean(Tv[i])-np.log(Kv[j])
        cm2  = pjd.var(Tv[i])
        cm4  = pjd.cumulant4(Tv[i])

        Ppoi[i,j]  = COS.vanilla(S0[1],Kv[j],Tv[i],r,cfp,[cm1,cm2,cm4],-1)

        # Hawkes jump-diffusion 
        cfh = lambda u: hjd.cf(u,0,Tv[i],1,Q0)

        # Cumulants
        cm1 = hjd.mean(Tv[i])-np.log(Kv[j])
        cm2 = hjd.var(Tv[i])
        cm4 = hjd.cumulant4(Tv[i])

        # COS method calls y puts
        Ph[i,j]  = COS.vanilla(S0[1],Kv[j],Tv[i],r,cfh,[cm1,cm2,cm4],-1)

## %% Plot the results
plt.figure()
plt.plot(Tv,P[:,4])
plt.plot(Tv,Ph[:,4])
plt.plot(Tv,Ppoi[:,4])
plt.xlabel('Maturity',fontsize=16)
plt.ylabel('Price',fontsize=16)
plt.legend(('Queue-H','Hawkes','Poisson'),fontsize=12)
plt.title(r'Set '+st,fontsize=20)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation Industry Week/figures/europeanT'+st+'.png')
plt.show()

plt.figure()
plt.plot(Kv,P[-2,:])
plt.plot(Kv,Ph[-2,:])
plt.plot(Kv,Ppoi[-2,:])
plt.xlabel('Strike',fontsize=16)
plt.ylabel('Price',fontsize=16)
plt.legend(('Queue-H','Hawkes','Poisson'),fontsize=12)
plt.title(r'Set '+st,fontsize=20)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation Industry Week/figures/europeanK'+st+'.png')
plt.show()

## %% Price Bermudan put
T = 1
K = S0[1]
M = np.arange(1,21)
nM = M.size
P2     = np.zeros((nM,))
Ppoi2  = np.zeros((nM,))
Ph2    = np.zeros((nM,))
for i in range(nM):
    # ESEP prices
    cfe = lambda u,v,t,Q: ejd.cf(u,v,t,1,Q)

    # Cumulants
    cm1 = ejd.mean(T)-np.log(K)
    cm2 = ejd.var(T)
    cm4 = ejd.cumulant4(T)
    cms = np.array([cm1,cm2,cm4])

    # COS method calls y puts
    Qe,Pe = COS.bermudan_put_nolevy_v2(S0[1],K,T,r,cms,0,0,cfe,M[i],jump=True)
    P2[i]  = Pe[Q0]

    # Poisson jump-diffusion
    cfp  = lambda u,v,t,Q: np.tile(pjd.cf(u,0,t,1,Q0),(Q.size,1,v.size))

    cm1  = pjd.mean(T)-np.log(K)
    cm2  = pjd.var(T)
    cm4  = pjd.cumulant4(T)
    cms = np.array([cm1,cm2,cm4])

    Qe,Pe = COS.bermudan_put_nolevy_v2(S0[1],K,T,r,cms,0,0,cfp,M[i],jump=True)
    Ppoi2[i] = Pe[Q0]

    # Hawkes jump-diffusion 
    cfh = lambda u,v,t,Q: hjd.cf(u,v,t,1,Q)

    # Cumulants
    cm1 = hjd.mean(T)-np.log(K)
    cm2 = hjd.var(T)
    cm4 = hjd.cumulant4(T)
    cms = np.array([cm1,cm2,cm4])

    # COS method calls y puts
    ah  = 0
    bh  = 2**5
    Qh,Pbh = COS.bermudan_put_nolevy_v2(S0[1],K,T,r,cms,ah,bh,cfh,M[i],jump=False)
    Pbh = CubicSpline(Qh,Pbh,axis=0)
    Ph2[i]  = Pbh(Q0)

## %% Plot the results
plt.figure()
plt.plot(M,P2)
plt.plot(M,Ph2)
plt.plot(M,Ppoi2)
plt.xlabel('Exercise dates',fontsize=16)
plt.ylabel('Price',fontsize=16)
plt.legend(('Queue-H','Hawkes','Poisson'),fontsize=12)
plt.title(r'Set '+st,fontsize=20)
plt.savefig('/home/luis_souto/Thesis/ESEP/Presentation Industry Week/figures/bermudan'+st+'.png')
plt.show()

## %% Price Bermudan Put

M   = 10  # Number of exercise times
cf  = lambda u: ejd.cf(u,0,T,1,Q0)
P   = COS.put(S0,K,T,r,cf,cme)
Pmc = K*np.exp(-r*T)*np.maximum(1-np.exp(Se),0).mean(-1)
print(P)
print(Pmc)

timestart = time.time()
cfe   = lambda u,v,t,Q: ejd.cf(u,v,t,1,Q)
Qe,Pe = COS.bermudan_put_nolevy_v2(S0,K,T,r,cme,0,0,cfe,M,jump=True)
time1 = time.time()-timestart

plt.figure()
plt.plot(Qe,Pe)
plt.xlabel('Intensity')
plt.ylabel('Bermudan price')
plt.show()

print('Value: ',Pe[Q0])
print('Time: ',time1)

## %% Same using Hawkes and Poisson

hjd  = ajd.GBMJD(S0,r,s,hawkes)
cmh1 = hjd.mean(T).mean()-np.log(K)
cmh2 = hjd.var(T)
cmh4 = 0
cmh  = np.array([cmh1,cmh2,cmh4])
cf2  = lambda u: hjd.cf(u,0,T,1,Q0)
P2   = COS.put(S0,K,T,r,cf2,cmh)
print(P2)

ah  = 0
bh  = 2**5
cfh = lambda u,v,t,Q: hjd.cf(u,v,t,1,Q)
timestart = time.time()
Qh,Ph = COS.bermudan_put_nolevy_v2(S0,K,T,r,cmh,ah,bh,cfh,M,jump=False)
time2 = time.time()-timestart

Pbh = CubicSpline(Qh,Ph,axis=0)

plt.figure()
plt.plot(Qh,Pbh(Qh))
plt.xlabel('Intensity')
plt.ylabel('Bermudan price')
plt.show()

print('Value: ',Pbh(Q0))
print('Time: ',time2)
