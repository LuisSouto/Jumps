#%% In this script we perform a sensitivity analysis for the paramters of the ESEP process. The results are compared with the Poisson and Hawkes jump-diffusions for similar configurations.

# %reload_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import py_vollib.black_scholes.implied_volatility as bs

import cos_method as COS
import gbmjd_class as ajd
from poisson_class import Poisson
from esep_class import ESEP
from hawkes_class import Hawkes

#%% Initialize parameters
a  = 1.9              # Intensity jump size
b  = 3.             # Memory kernel rate
hb = 6.             # Intesity baseline
Q0 = 0.             # Initial memory
h0 = hb + a*Q0      # Initial intensity
T  = 1              # Maturity
N  = 100000         # MC paths
S0 = 8              # Asset's initial value
K  = 10             # Strike
X0 = np.log(S0/K)   # Log-stock normalized by the strike
r  = 0.1            # Interest rate
s  = 0.3            # Volatility brownian motion
m  = -0.1           # Mean of jump size
s2 = 0.1            # Volatility of jump size
eJ = np.exp(m+s2**2/2)-1
fu = lambda u: np.exp(1j*u*m-s2**2*u**2/2);
cm = [m,m**2+s2**2,m**3+3*m*s2**2,m**4+6*m**2*s2**2+3*s2**4]

#%% Simulate the processes

## ESEP jump times 
esep = ESEP(a,b,hb,Q0,cm,eJ,fu)
Tx,B = esep.simul(N,T)

# Hawkes jump times 
hawk  = Hawkes(a,b,hb,Q0,cm,eJ,fu)
Th,Bh = hawk.simul(N,T)

# Poisson counts
hp = (b*hb/(a-b)+h0)*np.exp((a-b)*T)-b*hb/(a-b)
Np = np.random.poisson(hp*T,(N,))

## Brownian motion and jump sizes
W  = np.random.standard_normal((N,))
J  = np.random.standard_normal((N,))

Ne = ((Tx<=T)*(B>0)).sum(0)
Qe = (Ne-((Tx<=T)*(B<0)).sum(0))
Ie = h0+a*Qe
Se = (X0+(r-eJ*h0-s**2/2)*T+s*np.sqrt(T)*W-eJ*a*((Tx<T)*((T-Tx)*B)).sum(0)+Ne*m
      +s2*J*np.sqrt(Ne))
Nh = ((Th<=T)*Bh).sum(0)
Sh = (X0+(r-eJ*hb-s**2/2)*T+s*np.sqrt(T)*W+eJ/b*(h0-hb)*(np.exp(-b*T)-1)
      +eJ*a/b*(Bh*(Th<T)*(np.exp(-b*(T-Th))-1)).sum(0)+Nh*m+s2*J*np.sqrt(Nh))

Sp = X0+(r-eJ*hp-s**2/2)*T+s*np.sqrt(T)*W+Np*m+s2*J*np.sqrt(Np)    

#%% Plot histograms for the jump-difussions
x  = np.linspace(-4,4,300)
ex = np.exp(x)

ejd  = ajd.GBMJD(S0,r,s,esep)
cfe  = lambda u: ejd.cf(u,0,T,1,Q0)

cme1 = ejd.mean(T)
cme2 = ejd.var(T)
cme4 = ejd.cumulant4(T)

hjd  = ajd.GBMJD(S0,r,s,hawk)
cfh  = lambda u: hjd.cf(u,0,T,1,Q0)

cmh1 = hjd.mean(T)
cmh2 = hjd.var(T)
cmh4 = hjd.cumulant4(T)

poi  = Poisson(a,b,hb,Q0,cm,eJ,fu)
pjd  = ajd.GBMJD(S0,r,s,poi)
cfp  = lambda u: pjd.cf(u,0,T,1,Q0)

cmp1  = pjd.mean(T)
cmp2  = pjd.var(T)
cmp4  = pjd.cumulant4(T)


plt.figure()
plt.hist(Ie,hb+a*np.arange(10),density=True)
plt.xlim([0,10])
plt.show()

plt.figure()
plt.hist(np.exp(Se-X0),ex,density=True)
plt.plot(ex,COS.density(x,cfe,[cme1,cme2,cme4])/ex)
plt.xlim([0,10])
plt.show()

plt.figure()
plt.hist(np.exp(Sh-X0),ex,density=True)
plt.plot(ex,COS.density(x,cfh,[cmh1,cmh2,cmh4])/ex)
plt.xlim([0,10])
plt.show()

plt.figure()
plt.hist(np.exp(Sp-X0),ex,density=True)
plt.plot(ex,COS.density(x,cfp,[cmp1,cmp2,cmp4])/ex)
plt.xlim([0,10])
plt.show()

plt.figure()
plt.plot(ex,COS.density(x,cfe,[cme1,cme2,cme4])/ex)
plt.plot(ex,COS.density(x,cfh,[cmh1,cmh2,cmh4])/ex)
plt.plot(ex,COS.density(x,cfp,[cmp1,cmp2,cmp4])/ex)
plt.xlim([0,10])
plt.legend(('ESEP','Hawkes','Poisson'))
plt.show()

#%% Plot variance, skewness and kurtosis
t = np.linspace(0.1,2,400)

# Variance
plt.figure()
plt.plot(t,esep.var_cj(t))
plt.plot(t,hawk.var_cj(t))
plt.plot(t,poi.var_cj(t))
plt.legend(('ESEP','Hawkes','Poisson'))
plt.show()

# Skewness
plt.figure()
plt.plot(t,esep.skewness_cj(t))
plt.plot(t,hawk.skewness_cj(t))
plt.plot(t,poi.skewness_cj(t))
plt.legend(('ESEP','Hawkes','Poisson'))
plt.show()

# Kurtosis
plt.figure()
plt.plot(t,esep.kurtosis_cj(t))
plt.plot(t,hawk.kurtosis_cj(t))
plt.plot(t,poi.kurtosis_cj(t))
plt.legend(('ESEP','Hawkes','Poisson'))
plt.show()

#%% Pricing loop model comparison
Kv    = S0*np.arange(0.8,1.25,0.05)
Tv    = np.array([0.1,0.25,0.5,0.75,1,2])
nK    = Kv.size
nT    = Tv.size
P     = np.zeros((nT,nK))
C     = np.zeros((nT,nK))
Ppoi  = np.zeros((nT,nK))
Cpoi  = np.zeros((nT,nK))
Pmc   = np.zeros((nT,nK))
Cmc   = np.zeros((nT,nK))
Ph    = np.zeros((nT,nK))
Ch    = np.zeros((nT,nK))
IV    = np.zeros((nT,nK))
IVmc  = np.zeros((nT,nK))
IVh   = np.zeros((nT,nK))
IVpoi = np.zeros((nT,nK))

for i in range(nT):  

    for j in range(nK):
        # ESEP prices
        cfe = lambda u: ejd.cf(u,0,Tv[i],1,Q0)

        # Cumulants
        cm1 = ejd.mean(Tv[i])-np.log(Kv[j])
        cm2 = ejd.var(Tv[i])
        cm4 = ejd.cumulant4(Tv[i])

        # COS method calls y puts
        C[i,j]  = COS.call(S0,Kv[j],Tv[i],r,cfe,[cm1,cm2,cm4])
        P[i,j]  = COS.put(S0,Kv[j],Tv[i],r,cfe,[cm1,cm2,cm4])
        IV[i,j] = bs.implied_volatility(P[i,j],S0,Kv[j],Tv[i],r,'p') 

        # Monte Carlo calls y puts
        Cmc[i,j]  = np.exp(-r*Tv[i])*np.maximum(np.exp(Se)-Kv[j],0).mean()
        Pmc[i,j]  = np.exp(-r*Tv[i])*np.maximum(Kv[j]-np.exp(Se),0).mean()
        IVmc[i,j] = bs.implied_volatility(Pmc[i,j],S0,Kv[j],Tv[i],r,'p') 

        # Poisson jump-diffusion
        
        cfp  = lambda u: pjd.cf(u,0,Tv[i],1,Q0)

        cm1  = pjd.mean(Tv[i])-np.log(Kv[j])
        cm2  = pjd.var(Tv[i])
        cm4  = pjd.cumulant4(Tv[i])

        Cpoi[i,j]  = COS.call(S0,Kv[j],Tv[i],r,cfp,[cm1,cm2,cm4])
        Ppoi[i,j]  = COS.put(S0,Kv[j],Tv[i],r,cfp,[cm1,cm2,cm4])
        IVpoi[i,j] = bs.implied_volatility(Ppoi[i,j],S0,Kv[j],Tv[i],r,'p') 
        
        # Hawkes jump-diffusion 
        cfh = lambda u: hjd.cf(u,0,Tv[i],1,Q0)

        # Cumulants
        cm1 = hjd.mean(Tv[i])-np.log(Kv[j])
        cm2 = hjd.var(Tv[i])
        cm4 = hjd.cumulant4(Tv[i])

        # COS method calls y puts
        Ch[i,j]  = COS.call(S0,Kv[j],Tv[i],r,cfh,[cm1,cm2,cm4])
        Ph[i,j]  = COS.put(S0,Kv[j],Tv[i],r,cfh,[cm1,cm2,cm4])
        IVh[i,j] = bs.implied_volatility(Ph[i,j],S0,Kv[j],Tv[i],r,'p') 

#%% Plot results

# As a function of the strike
fig, axs = plt.subplots(nT, 1, figsize=(4,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Strike',fontsize=18,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nT):
     axs[i].plot(Kv,IV[i,:],label='ESEP')
     axs[i].plot(Kv,IVh[i,:],label='Hawkes')
     axs[i].plot(Kv,IVpoi[i,:],label='Poisson')
axs[-1].legend(fontsize=10)

plt.show()

# As a function of the maturity

fig, axs = plt.subplots(nK, 1, figsize=(4,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Maturity',fontsize=18,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nK):
     axs[i].plot(Tv,IV[:,i],label='ESEP')
     axs[i].plot(Tv,IVh[:,i],label='Hawkes')
     axs[i].plot(Tv,IVpoi[:,i],label='Poisson')
axs[-1].legend(fontsize=10)

plt.show()

#%% Sensitivity analysis: parameter "a"

Ta   = np.array([0.1,0.5,1,2])
Ka   = S0*np.arange(0.8,1.25,0.05)
av   = b*np.array([0.1,0.25,0.5,0.75,0.9,0.99,0.999])
na   = av.size 
nTa  = Ta.size
nKa  = Ka.size

Cea,Pea,IVea = ejd.sens_an(Ta,Ka,av,'a')
Cha,Pha,IVha = hjd.sens_an(Ta,Ka,av,'a')
Cpa,Ppa,IVpa = pjd.sens_an(Ta,Ka,av,'a')
            
ejd.jp.set_param(a=a)
hjd.jp.set_param(a=a)
pjd.jp.set_param(a=a)

#%% Plot results
fig, axs = plt.subplots(nTa, nKa, figsize=(25,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Intensity jump size $\alpha$',fontsize=22,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nTa):
     for j in range(nKa):
          axs[i,j].plot(av,IVea[:,i,j],label='ESEP')
          axs[i,j].plot(av,IVha[:,i,j],label='Hawkes')
          axs[i,j].plot(av,IVpa[:,i,j],label='Poisson')
axs[0,-1].legend(fontsize=16)

plt.show()

#%% Sensitivity analysis: parameter "b"

Tb   = np.array([0.1,0.5,1.0,2.])
Kb   = S0*np.arange(0.8,1.25,0.05)
bv   = a*np.array([1.001,1.01,1.11,1.5,2,4,7,10,20])
nb   = bv.size
nTb  = Tb.size 
nKb  = Kb.size

Ceb,Peb,IVeb = ejd.sens_an(Tb,Kb,bv,'b')
Chb,Phb,IVhb = hjd.sens_an(Tb,Kb,bv,'b')
Cpb,Ppb,IVpb = pjd.sens_an(Tb,Kb,bv,'b')            

ejd.jp.set_param(b=b)
hjd.jp.set_param(b=b)
pjd.jp.set_param(b=b)

#%% Plot results
fig, axs = plt.subplots(nTb, nKb, figsize=(25,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Expiration rate $\beta$',fontsize=22,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nTb):
     for j in range(nKb):
          axs[i,j].plot(bv,IVeb[:,i,j],label='ESEP')
          axs[i,j].plot(bv,IVhb[:,i,j],label='Hawkes')
          axs[i,j].plot(bv,IVpb[:,i,j],label='Poisson')
axs[0,-1].legend(fontsize=16)

plt.show()

#%% Sensitivity analysis: parameter "hb"

Thb   = np.array([0.1,0.5,1.0,2.])
Khb   = S0*np.arange(0.8,1.25,0.05)
hbv   = np.array([0.25,0.5,1,2,5,10])
nhb   = hbv.size 
nThb  = Thb.size
nKhb  = Khb.size

Cehb,Pehb,IVehb = ejd.sens_an(Thb,Khb,hbv,'hb')
Chhb,Phhb,IVhhb = hjd.sens_an(Thb,Khb,hbv,'hb')
Cphb,Pphb,IVphb = pjd.sens_an(Thb,Khb,hbv,'hb')            

ejd.jp.set_param(hb=hb)
hjd.jp.set_param(hb=hb)
pjd.jp.set_param(hb=hb)

#%% Plot results
fig, axs = plt.subplots(nThb, nKhb, figsize=(25,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Baseline intensity $\lambda^*$',fontsize=22,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nThb):
     for j in range(nKhb):
          axs[i,j].plot(hbv,IVehb[:,i,j],label='ESEP')
          axs[i,j].plot(hbv,IVhhb[:,i,j],label='Hawkes')
          axs[i,j].plot(hbv,IVphb[:,i,j],label='Poisson')
axs[0,-1].legend(fontsize=16)

plt.show()

#%% Sensitivity analysis: parameter "Q0"

TQ0   = np.array([0.1,0.5,1.0,2.])
KQ0   = S0*np.arange(0.8,1.25,0.05)
Q0v   = np.array([0,0.5,1,2,5,10])
nQ0   = Q0v.size 
nTQ0  = TQ0.size
nKQ0  = KQ0.size

CeQ0,PeQ0,IVeQ0 = ejd.sens_an(TQ0,KQ0,Q0v,'Q0')
ChQ0,PhQ0,IVhQ0 = hjd.sens_an(TQ0,KQ0,Q0v,'Q0')
CpQ0,PpQ0,IVpQ0 = pjd.sens_an(TQ0,KQ0,Q0v,'Q0')
            

ejd.jp.set_param(Q0=Q0)
hjd.jp.set_param(Q0=Q0)
pjd.jp.set_param(Q0=Q0)

#%% Plot results
fig, axs = plt.subplots(nTQ0, nKQ0, figsize=(25,15), sharex=True, sharey=True)
fig.tight_layout()

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)

plt.xlabel(r'Initial activation number $Q_0$',fontsize=22,labelpad=7)
plt.ylabel('Implied volatility',fontsize=22,labelpad=12)  
for i in range(nTQ0):
     for j in range(nKQ0):
          axs[i,j].plot(Q0v,IVeQ0[:,i,j],label='ESEP')
          axs[i,j].plot(Q0v,IVhQ0[:,i,j],label='Hawkes')
          axs[i,j].plot(Q0v,IVpQ0[:,i,j],label='Poisson')
axs[0,-1].legend(fontsize=16)

plt.show()
# %%
