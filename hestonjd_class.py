#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:55:53 2022

@author: luis_souto

Class which combines a Heston diffusion with a jump process.
"""

import numpy as np

import cos_method as COS
import py_vollib.black_scholes.implied_volatility as bs

class HestonJD():
    def __init__(self,dp,jp):
        self.dp  = dp
        self.jp  = jp

    # Characteristic function 
    def cf(self,u,v,w,t,X):        
        return self.dp.cf(u,v,t,X[:-1])*self.jp.cf_cj(u,w,t,X[-1])

    def mean(self,t):
        return self.dp.mean(t)+self.jp.mean_cj(t)

    def var(self,t):
        return self.dp.var(t)+self.jp.var_cj(t)

    def cumulant4(self,t):
        return self.dp.cumulant4(t)+self.jp.cumulant4_cj(t)

    def cumulants_cos(self,t):
        return np.array([self.mean(t).mean(),self.var(t),0])
    
    def simul(self,N,n,T):
        dt = T/(n-1)        
        
        # Simulate the jumps
        Tx,B  = self.jp.simul(N,T)
        Q  = np.zeros((N,n))
        Nj = np.zeros((N,n))
        Nj[:,0] = 0
        Q[:,0]  = self.jp.Q0
        for i in range(1,n):
            Nj[:,i],Q[:,i] = self.jp.compute_intensity(dt*i,Tx,B)
        I = self.jp.hb + self.jp.a*Q            
        
        # Simulate the diffusion
        X = np.zeros((N,n))
        V = np.zeros((N,n))
        X[:,0] = self.dp.X0
        V[:,0] = self.dp.V0

        lamb = self.dp.lamb
        nu   = self.dp.nu
        eta  = self.dp.eta
        rho  = self.dp.rho
        r    = self.dp.r
        s    = np.sqrt(self.jp.mJ[1]-self.jp.mJ[0]**2)

        dWS = np.random.normal(0,np.sqrt(dt),(N,n-1))
        dWV = rho*dWS+np.sqrt(1-rho**2)*np.random.normal(0,np.sqrt(dt),(N,n-1))
        J   = np.random.normal(self.jp.mJ[0],s,(N,n-1))
        for i in range(1,n):
            Vp = np.maximum(V[:,i-1],0)
            V[:,i] = V[:,i-1] + lamb*(nu-Vp)*dt + np.sqrt(Vp)*eta*dWV[:,i-1]
            X[:,i] = (X[:,i-1] + (r-0.5*Vp)*dt + np.sqrt(Vp)*dWS[:,i-1] 
                      +J[:,i-1]*(Nj[:,i]-Nj[:,i-1])-self.jp.eJ*I[:,i-1]*dt)
            
        return X,V,Q

    def sens_an(self,S,T,K,v,param=['a']):
        if np.isscalar(T): T = np.array([T])
        if np.isscalar(K): K = np.array([K])
        if np.isscalar(v[0]): v = np.array([v]).T

        r  = self.dp.r
        n  = v.shape[0]
        nT = T.size
        nK = K.size
        P  = np.zeros((n,nT,nK))
        IV = np.zeros((n,nT,nK))
        for i in range(n):
            param_dict = dict(zip(param,v[i]))
            self.jp.set_param(**param_dict)
            x0 = np.array([0,self.dp.V0,self.jp.Q0])
            for j in range(nT):
                # Characteristic function
                cfs = lambda u: self.cf(u,0,0,T[j],x0)
                for k in range(nK):
                    # Cumulants
                    cm = self.cumulants_cos(T[j])
                    cm[0] -= np.log(K[k])

                    # COS method puts
                    P[i,j,k]  = COS.vanilla(S,K[k],T[j],r,cfs,cm,-1)
                    IV[i,j,k] = bs.implied_volatility(P[i,j,k],S,K[k],T[j],r,'p')

        return P.squeeze(),IV.squeeze()