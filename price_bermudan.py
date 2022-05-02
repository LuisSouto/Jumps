#%% Numerical testing of the COS method for bermudan options with the Heston model.


import sys
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import CubicSpline


import cos_method as COS
import quad_rules as Quad
from heston_class import Heston

np.set_printoptions(threshold=sys.maxsize)

#%% Initialize data

# Simple setting (Test No. 4 in Fang)
# S0   = np.array([8,9,10,11,12])
# K    = 10
# r    = 0.1
# V0   = 0.0625
# rho  = 0.1
# eta  = 0.9     # Vol of vol
# lamb = 5       # Speed of mean-reversion
# nu   = 0.16    # Long-term variance
# T    = 0.25
# M    = 1      # Number of exercise times

# Medium setting (Test No. 5 in Fang)
# S0   = np.array([90,100,110])
# K    = 100
# r    = 0.04
# V0   = 0.0348
# rho  = -0.64
# eta  = 0.39
# lamb = 1.15
# nu   = 0.0348
# T    = 0.25
# M    = 20

# Hard setting (Set C in Ruijter)
S0   = np.array([90,100,110])
K    = 100
r    = 0.04
V0   = 0.04
rho  = -0.9
eta  = 0.5
lamb = 0.5
nu   = 0.04
T    = 1
M    = 10

#%% Compute European put for reference
heston = Heston(S0,V0,r,lamb,nu,eta,rho)
cm     = [heston.mean(T).mean()-np.log(K),heston.var(T),0]
cf     = lambda u: heston.cf(u,0,T,0,V0)
P      = COS.put(S0,K,T,r,cf,cm)

print(P)

#%% Determine truncation interval for variance

tola = 1e-8      # Bound on left-tail
tolb = 1e-6      # Bound on right-tail
mV   = np.log(V0*np.exp(-lamb*T)+nu*(1-np.exp(-lamb*T)))
q    = 2*lamb*nu/eta**2-1
c    = 2*lamb/((1-np.exp(-lamb*T))*eta**2)
av   = mV - 5/(1+q)
# bv   = mV + 1/(1+q)
bv   = mV + 2.1 # The rule of thumb is pretty bad for Set C. This works better

# Density of log-variance and its first derivative
fv   = lambda v: heston.dens_logvar(T,np.exp(v),V0)
fv_v = lambda v: heston.dens_logvar_dv(T,np.exp(v),V0)

while(np.abs(fv(av))>tola):
    av -= fv(av)/fv_v(av)

while(np.abs(fv(bv))>tolb):
    bv -= fv(bv)/fv_v(bv)  
    
#%% Price Bermudan put using Fang

# This is Equation (30) in Fang (2011)
cfb = lambda u,t,vt,vs: heston.cfcondv_dens(u,t,0,vt,vs)

timestart = time.time()
Vs,PVs    = COS.bermudan_put_nolevy(S0,K,T,r,cm,av,bv,cfb,M)
time1     = time.time()-timestart
print('Elapsed time: ',time1)

Pb = CubicSpline(Vs,PVs,axis=0)

plt.figure()
plt.plot(Vs,Pb(Vs))
plt.show()

print(Pb(V0))

#%% Price with the 2D COS method (using quadrature)

cf2 = lambda u,v,t,vs: heston.cf(u,v,t,0,vs)

timestart = time.time()
Vs,PVs    = COS.bermudan_put_nolevy_v2(S0,K,T,r,cm,np.exp(av),np.exp(bv),cf2,M)
time2     = time.time()-timestart
print('Elapsed time: ',time2)

Pb2 = CubicSpline(Vs,PVs,axis=0)

plt.figure()
plt.plot(Vs,Pb2(Vs))
plt.show()

plt.figure()
plt.plot(Vs,Pb2(Vs)-Pb(Vs))
plt.show()

print(Pb2(V0))
# %% Simulate Heston (for debugging)

N  = 8000
nT = 1000
dt = T/(nT-1)

S = np.zeros((N,nT))
V = np.zeros((N,nT))
S[:,0] = np.log(S0[0])
V[:,0] = V0

dWS = np.random.normal(0,np.sqrt(dt),(N,nT-1))
dWV = rho*dWS + np.sqrt(1-rho**2)*np.random.normal(0,np.sqrt(dt),(N,nT-1))
for i in range(1,nT):
    Vp = np.maximum(V[:,i-1],0)
    V[:,i] = V[:,i-1] + lamb*(nu-Vp)*dt + np.sqrt(Vp)*eta*dWV[:,i-1]
    S[:,i] = S[:,i-1] + (r-0.5*Vp)*dt + np.sqrt(Vp)*dWS[:,i-1]
    
u1 = 0.5
u2 = 1.9
print(np.maximum(K-np.exp(S[:,-1]),0).mean())
print(np.exp(1j*(u1*S[:,-1]+u2*V[:,-1])).mean())
print(heston.cf(u1,u2,T,np.log(S0[0]),V0))

#%% Compare COS method with DCT

J   = 2**6
k   = np.arange(J)*np.pi/J
eav = np.exp(av)
ebv = np.exp(bv)
vi,wi = Quad.trapezoidal(eav, ebv, J)

cos_term = np.cos(k[:,np.newaxis]*(np.arange(J)+0.5))
cos_term[0,:] *= 0.5
fdct = 2/J*np.real(np.dot((np.exp(1j*k*(0.5-eav*(J-1)/(ebv-eav)))
                           *heston.cf(0,k*(J-1)/(ebv-eav),T,1,V0))
                          ,cos_term))

k = np.arange(J)*np.pi/(ebv-eav)
cos_term = np.cos(k[:,np.newaxis]*(vi-eav))
cos_term[0,:] *= 0.5
fcos = 2/(ebv-eav)*np.real(np.dot((np.exp(-1j*k*eav)*heston.cf(0,k,T,1,V0))
                                  ,cos_term))

cfd = lambda u: (heston.cf(0,u*(J-1)/(ebv-eav),T,1,V0)
                 *np.exp(-1j*u*(J-1)*eav/(ebv-eav)))

fdis2 = COS.density_dis(np.arange(J), cfd)
plt.figure()
plt.plot(vi,fdct)
plt.plot(vi,fcos*wi)
plt.plot(vi,fdis2)
# plt.plot(vi,fv(np.log(vi))/vi*wi)
plt.show()