#%% Poisson bridge.

%reload_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt

#%%

X1 = 10
r  = 1
T  = 1
n  = 5000
N  = 1000
dt = T/(n-1)

t = np.linspace(0,T,n,endpoint=True)
X = np.zeros((N,n))
# %%

for i in range(n-1):
    X[:,i+1] = X[:,i] + np.random.poisson(r*(np.maximum(X1-X[:,i],0))/(T-t[i])*dt)
# %%

plt.figure()
for i in range(5):
    plt.plot(t,X[i,:])

plt.show()
# %%
