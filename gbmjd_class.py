#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mond Feb 21 15:01:42 2022

@author: luis_souto

Class for classical jump-diffusions where the diffusion component is given by a standar brownian motion with constant drift and volatility. The jump component is externally provided and is an attribute of the jump-diffusion class.

This process corresponds to the log-stock in the risk neutral measure.
"""

import numpy as np

import cos_method as COS
import py_vollib.black_scholes.implied_volatility as bs

class GBMJD():
    def __init__(self,S0,r,sig,jp):
        self.S0  = S0
        self.r   = r
        self.sig = sig
        self.jp  = jp


    # Characteristic function 
    def cf(self,u,v,t,S,Q):
        r = self.r
        s = self.sig
        
        return (np.exp(1j*u*(np.log(S)+(r-0.5*s**2)*t)-u**2*t*s**2/2)
                *self.jp.cf_cj(u,v,t,Q))

    def mean(self,t):
        S0 = self.S0
        r  = self.r
        s  = self.sig
        return np.log(S0) + (r-0.5*s**2)*t + self.jp.mean_cj(t)

    def var(self,t):
        return self.sig**2*t + self.jp.var_cj(t)

    def cumulant4(self,t):
        return self.jp.cumulant4_cj(t)

    def sens_an(self,T,K,v,param='a'):
        S0 = self.S0
        r  = self.r
        n  = v.size 
        nT = T.size
        nK = K.size
        P  = np.zeros((n,nT,nK))
        C  = np.zeros((n,nT,nK))
        IV = np.zeros((n,nT,nK))

        for i in range(n):
            for j in range(nT):
                for k in range(nK):
                    # ESEP prices
                    self.jp.set_param(**{param:v[i]})
                    cfs = lambda u: self.cf(u,0,T[j],1,self.jp.Q0)

                    # Cumulants
                    cm1   = self.mean(T[j])-np.log(K[k])
                    cm2   = self.var(T[j])
                    cm4   = 0 #esep.cumulant4_cj(T[j],cm,eJ)

                    # COS method calls y puts
                    C[i,j,k]  = COS.call(S0,K[k],T[j],r,cfs,[cm1,cm2,cm4])
                    P[i,j,k]  = COS.put(S0,K[k],T[j],r,cfs,[cm1,cm2,cm4])
                    IV[i,j,k] = bs.implied_volatility(P[i,j,k],S0,K[k],T[j],r,'p') 

        return C,P,IV