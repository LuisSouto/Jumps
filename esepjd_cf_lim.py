#%% Look for asymptotic behaviour of the characteristic function for the ESEP jump-diffusion.

%reload_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import cProfile

import esep_class as es
import cos_method as COS
import black_scholes as bs
import hawkes_class as hawkes

#%% Initialize parameters
a  = 1.5          # Intensity jump size
b  = 3.          # Memory kernel rate
hb = 3.           # Intesity baseline
Q0 = 5.           # Initial memory
h0 = hb + a*Q0    # Initial intensity
T  = 300.           # Maturity
N  = 100000       # MC paths
S0 = 1            # Asset's initial value
K  = 1            # Strike
r  = 0.03         # Interest rate
s  = 0.3          # Volatility brownian motion
m  = 1.0          # Mean of jump size
s2 = 0.0          # Volatility of jump size
eJ = np.exp(m+s2**2/2)-1
fu = lambda u: np.exp(1j*u*m-s2**2*u**2/2);

esep = es.ESEP(a,b,hb,Q0)
u    = np.linspace(0,6*np.pi,300)
cfu  = esep.cf_jd(u,T,eJ,fu)
cfu2 = esep.cf_jd_asy(u,T,eJ,fu)

#%% Plot the results
plt.figure()
plt.plot(u,np.abs(cfu))
plt.plot(u,np.abs(cfu2))
plt.xlabel('u')
plt.ylabel('cfu')
plt.title('Characteristic function (real part)')
plt.legend(('Transient','Asymptotic'))
plt.show()