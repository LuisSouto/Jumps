## Load libraries

import numpy as np
import matplotlib.pyplot as plt
import cos_method as COS

from scipy.special import factorial
from scipy.stats import poisson

def cf_uni(u,M):
    f = (1-np.exp(1j*(M+1)*u))/((M+1)*(1-np.exp(1j*u)))
    f[u%(2*np.pi)==0] = 1
    return f

def cf_poi(u,lamb):
    return np.exp(lamb*(np.exp(1j*u)-1))

## Compute error
xU0 = 24
xP0 = 12
kU = np.arange(20,150)
kP = np.arange(20,40)
M = 500
lamb = 15

fU = lambda u: cf_uni(u,M)
fP = lambda u: cf_poi(u,lamb)

COS_errU = np.zeros((kU.size,))
COS_errP = np.zeros((kP.size,))
fUref = 1/(M+1)
fPref = poisson.pmf(xP0,lamb)
for i in range(kU.size):
    fxU = COS.density_dis(xU0,fU,kU[i])
    COS_errU[i] = np.abs(fxU-fUref)

for i in range(kP.size):
    fxP = COS.density_dis(xP0,fP,kP[i])
    COS_errP[i] = np.abs(fxP-fPref)

## Plot results
plt.figure()
plt.plot(kU,COS_errU,'r*')
plt.plot(kU,COS_errU[0]/kU*kU[0],'b-')
plt.xlabel('N',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.legend((r'$\hat{p}_X(n)-p_X(n)$',r'$1/N$'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/cos_error_uniform.eps',format='eps')
plt.show()

plt.figure()
plt.plot(kP,COS_errP,'r*')
plt.plot(kP,COS_errP[0]*np.exp(-kP/lamb*(kP-kP[0])),'b-')
plt.yscale("log")
plt.xlabel('N',fontsize=16)
plt.ylabel('Error',fontsize=16)
plt.legend((r'$\hat{p}_X(n)-p_X(n)$',r'exp(-$N^2 / \lambda$)'),fontsize=12)
plt.savefig('/home/luis_souto/Thesis/ESEP/Paper/figures/cos_error_poisson.eps',format='eps')
plt.show()

##

