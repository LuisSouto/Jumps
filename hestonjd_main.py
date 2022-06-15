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

## %% Load modules

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
from hestonjd_class import HestonJD
from poisson_class import Poisson

np.set_printoptions(threshold=sys.maxsize)

## %% Initialize parameters
st = 'E'                   # Scenario
T  = 1                     # Maturity
nT = 300
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
a  = 2.9                   # Clustering rate
b  = 3.                    # Expiration rate
hb = 1.1                   # Baseline intensity
Q0 = 2                     # Initial activation
h0 = hb + a*Q0             # Initial intensity
mj = 0.3                   # Mean of log-jump size
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
heston_e = HestonJD(heston,qhawk)
heston_h = HestonJD(heston,hawk)
heston_p = HestonJD(heston,poi)

# Simulate the processes via Monte Carlo. This was done in order to validate
# the results obtained with the COS method. Now it can be safely removed.
# J     = np.random.standard_normal((N,))
# Tx,B  = qhawk.simul(N,T)
# Th,Bh = hawk.simul(N,T)
# Ne,Qe = qhawk.compute_intensity(T,Tx,B)
# Nh,Ih = hawk.compute_intensity(T,Th,Bh)
# Nh    = Nh.squeeze()
# S,V   = heston.simul(N,nT,T)
# S    -= np.log(K)
# Mq    = Ne*mj+sj*J*np.sqrt(Ne)-eJ*a*((Tx<T)*((T-Tx)*B)).sum(0)-eJ*h0*T
# Sq    = S[:,-1]+Mq
# Mh    = (-eJ*hb*T+eJ/b*(h0-hb)*(np.exp(-b*T)-1)
#          +eJ*a/b*(Bh*(Th<T)*(np.exp(-b*(T-Th))-1)).sum(0)+Nh*mj+sj*J*np.sqrt(Nh))
# Sh    = S[:,-1]+Mh

## Compute the density with the COS method
x0 = np.array([X0[0],V0,Q0])

# Heston-Q-Hawkes
cfx  = lambda u: heston_e.cf(u,0,0,T,x0)
cme  = heston_e.cumulants_cos(T)
cme[0] -= np.log(K)
L    = 5
ac   = cme[0]-L*np.sqrt(cme[1]+np.sqrt(cme[2]))
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
x0  = np.array([0,V0,Q0])
Kv  = S0[0]*np.linspace(0.8,1.2,21,endpoint=True)  # Strike grid
Tv  = np.linspace(0.1,2,20,endpoint=True)          # Maturity grid
nK  = Kv.size
nT  = Tv.size
Pq  = np.zeros((nT,nK))
Pp  = np.zeros_like(Pq)
Ph  = np.zeros_like(Pq)
IVe = np.zeros_like(Pq)
IVh = np.zeros_like(Pq)
IVp = np.zeros_like(Pq)
Delta_e = np.zeros_like(Pq)
Delta_h = np.zeros_like(Pq)
Delta_p = np.zeros_like(Pq)

time_e = 0
time_h = 0
time_p = 0
nrep   = 1    # Number of repetitions for runtime comparison purposes
for k in range(nrep):
    for i in range(nT):
        for j in range(nK):
            # Heston-Q-Hawkes
            cfe = lambda u: heston_e.cf(u,0,0,Tv[i],x0)
            cme = heston_e.cumulants_cos(Tv[i])
            cme[0] -= np.log(Kv[j])

            time1 = time.time()
            Pq[i,j],Delta_e[i,j] = COS.vanilla(S0[0],Kv[j],Tv[i],r,cfe,cme,
                                               alpha=-1,compute_delta=True)
            time_e += (time.time()-time1)
            IVe[i,j] = bs.implied_volatility(Pq[i,j],S0[0],Kv[j],Tv[i],r,'p')

            # Bates model
            cfp  = lambda u: heston_p.cf(u,0,0,Tv[i],x0)
            cmp = heston_p.cumulants_cos(Tv[i])
            cmp[0] -= np.log(Kv[j])

            time1 = time.time()
            Pp[i,j],Delta_p[i,j] = COS.vanilla(S0[0],Kv[j],Tv[i],r,cfp,cmp,
                                               alpha=-1,compute_delta=True)
            time_p += (time.time()-time1)
            IVp[i,j] = bs.implied_volatility(Pp[i,j],S0[0],Kv[j],Tv[i],r,'p')

            # Heston-Hawkes
            cfh = lambda u: heston_h.cf(u,0,0,Tv[i],x0)
            cmh = heston_h.cumulants_cos(Tv[i])
            cmh[0] -= np.log(Kv[j])

            time1 = time.time()
            Ph[i,j],Delta_h[i,j] = COS.vanilla(S0[0],Kv[j],Tv[i],r,cfh,cmh,
                                               alpha=-1,compute_delta=True)
            time_h += (time.time()-time1)
            IVh[i,j] = bs.implied_volatility(Ph[i,j],S0[0],Kv[j],Tv[i],r,'p')

print(time_e/nrep,time_h/nrep,time_p/nrep)
## Plot the results
idK = 10
plt.figure()
plt.plot(Tv,IVe[:,idK],'r*')
plt.plot(Tv,IVh[:,idK],'bx')
plt.plot(Tv,IVp[:,idK],'g^')
plt.xlabel('Maturity',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12,loc='lower right')
plt.show()

plt.figure()
plt.plot(Tv,Delta_e[:,idK],'r*')
plt.plot(Tv,Delta_h[:,idK],'bx')
plt.plot(Tv,Delta_p[:,idK],'g^')
plt.xlabel('Maturity',fontsize=16)
plt.ylabel(r'Delta $\Delta$',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12,loc='lower right')
plt.show()

idT = 9
plt.figure()
plt.plot(Kv,IVe[idT,:],'r*')
plt.plot(Kv,IVh[idT,:],'bx')
plt.plot(Kv,IVp[idT,:],'g^')
plt.xlabel('Strike',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

plt.figure()
plt.plot(Kv,Delta_e[idT,:],'r*')
plt.plot(Kv,Delta_h[idT,:],'bx')
plt.plot(Kv,Delta_p[idT,:],'g^')
plt.xlabel('Strike',fontsize=16)
plt.ylabel(r'Delta $\Delta$',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.show()

## Price Bermudan put options
cfV   = lambda u,t,vt,vs: heston.cfcondv_dens(u,t,0,vt,vs)
cfPoi = lambda u,t,vt,vs: cfV(u,t,vt,vs)*poi.cf_cj(u,0,t,Q0)
cfQ   = lambda u,t,qt,qs: qhawk.cf_integral(u,qt,t,qs,vectorize=True)
cfH   = lambda u,t,qt,qs: hawk.cf_cossum(u,t,qt,qs)

av,bv = heston.logvar_bounds(Tv[idT])

M  = np.arange(1,11) # Exercise dates
nM = M.size
Pq2  = np.zeros((nM,))
Pp2  = np.zeros_like(Pq2)
Ph2  = np.zeros_like(Pq2)
IVbe = np.zeros_like(Pq2)
IVbh = np.zeros_like(Pq2)
IVbp = np.zeros_like(Pq2)

time_e = 0
time_h = 0
time_p = 0
nrep   = 1
for k in range(nrep):
    for i in range(nM):
        # Heston-Q-Hawkes
        cme = heston_e.cumulants_cos(Tv[idT]/M[i])
        cme[0] -= np.log(Kv[idK])
        time1   = time.time()
        Vb,Qb,Pb = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cme,av,bv,cfV,
                                       cfQ,M[i],N1=2**6,nV=2**5,nQ=2**5)
        time_e += (time.time()-time1)
        Pq2[i]  = CubicSpline(Vb,Pb[:,Q0,:],axis=0)(V0)
        IVbe[i] = bs.implied_volatility(Pq2[i],S0[0],Kv[idK],Tv[idT],r,'p')

        # Bates
        cmp = heston_p.cumulants_cos(Tv[idT]/M[i])
        cmp[0] -= np.log(Kv[idK])
        time1   = time.time()
        Vbp,Pbp = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cmp,av,bv,cfPoi,
                                      cfQ,M[i],N1=2**6,nV=2**5,dim3=False)
        time_p += (time.time()-time1)
        Pp2[i]  = CubicSpline(Vbp,Pbp,axis=0)(V0)
        IVbp[i] = bs.implied_volatility(Pp2[i],S0[0],Kv[idK],Tv[idT],r,'p')

        # Heston-Hawkes
        cmh = heston_h.cumulants_cos(Tv[idT]/M[i])
        cmh[0] -= np.log(Kv[idK])
        time1   = time.time()
        Vbh,Qbh,Pbh = COS.bermudan_put_3D(S0[0],Kv[idK],Tv[idT],r,cmh,av,bv,cfV,
                                          cfH,M[i],N1=2**6,nV=2**5,nQ=2**5,
                                          jump=False)
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
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_a'+st+'.eps',format='eps')
plt.show()

## Sensitivity analysis beta
bv  = a*np.linspace(1.01,10,20,endpoint=True)
nb  = bv.size

Pbe,IVbe = heston_e.sens_an(S0[0],T,S0[0],bv,'b')
Pbh,IVbh = heston_h.sens_an(S0[0],T,S0[0],bv,'b')

## %% Plot the results
plt.figure()
plt.plot(bv,IVbe,'r*')
plt.plot(bv,IVbh,'bx')
plt.xlabel(r'Expiration rate $\beta$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_b'+st+'.eps',format='eps')
plt.show()

## Sensitivity analysis hb
hbv  = np.linspace(0.05,5,20,endpoint=True)
nhb  = hbv.size

Phbe,IVhbe = heston_e.sens_an(S0[0],T,S0[0],hbv,['hb'])
Phbh,IVhbh = heston_h.sens_an(S0[0],T,S0[0],hbv,['hb'])

## %% Plot the results
plt.figure()
plt.plot(hbv,IVhbe,'r*')
plt.plot(hbv,IVhbh,'bx')
plt.xlabel(r'Baseline intensity $\lambda^*$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_hb'+st+'.eps',format='eps')
plt.show()

## Sensitivity analysis Q0
Q0v  = np.linspace(0.05,5,20,endpoint=True)
nQ0  = Q0v.size

PQ0e,IVQ0e = heston_e.sens_an(S0[0],T,S0[0],Q0v,['Q0'])
PQ0h,IVQ0h = heston_h.sens_an(S0[0],T,S0[0],Q0v,['Q0'])

## %% Plot the results
plt.figure()
plt.plot(Q0v,IVQ0e,'r*')
plt.plot(Q0v,IVQ0h,'bx')
plt.xlabel(r'Initial value $Q_0$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_Q0'+st+'.eps',format='eps')
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

## %% Plot the results
plt.figure()
plt.plot(mjv,IVmje,'r*')
plt.plot(mjv,IVmjh,'bx')
plt.xlabel(r'Jump size expectation $\mu_Y$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_mj'+st+'.eps',format='eps')
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

## %% Plot the results
plt.figure()
plt.plot(sjv,IVsje,'r*')
plt.plot(sjv,IVsjh,'bx')
plt.xlabel(r'Jump size volatility $\sigma_Y$',fontsize=16)
plt.ylabel('Implied volatility',fontsize=16)
plt.legend(('Q-Hawkes','Hawkes','Poisson'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/sensitivity_sj'+st+'.eps',format='eps')
plt.show()


##

