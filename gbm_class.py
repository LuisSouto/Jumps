#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mond Feb 21 15:01:42 2022

@author: luis_souto

Class for classical Geometric Brownian motion. Meaning of the attributes are:
     - S0:  Initial stock value
     - r:   Risk-free interest rate
     - sig: Volatility
"""

import numpy as np

class GBM():
    def __init__(self,S0,r,sig):
        self.S0  = S0
        self.r   = r
        self.sig = sig

    # Characteristic function 
    def cf(self,u,v,t,S):
        r = self.r
        s = self.sig
        
        return np.exp(1j*u*(np.log(S)+(r-0.5*s**2)*t)-u**2*t*s**2/2)

    def mean(self,t):
        S0 = self.S0
        r  = self.r
        s  = self.sig
        return np.log(S0)+(r-0.5*s**2)*t

    def var(self,t):
        return self.sig**2*t

    def cumulant4(self,t):
        return 0