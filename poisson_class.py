#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:07:42 2022

@author: luis_souto

class related to the ESEP process. It contains attributes related to the 
parameters of the model, as well as methods for computing the characteristic
function and performing simulations via the thinning algorithm.
"""

from re import A
import numpy as np

from pp_class import PointP

class Poisson(PointP):
        
    def rate(self,t):
        a  = self.a
        b  = self.b
        hb = self.hb
        h0 = self.h0

        return  (b*hb/(a-b)+h0)*np.exp((a-b)*t)-b*hb/(a-b)

    def mean_cj(self,t):
        m  = self.mJ[0]
        eJ = self.eJ

        return self.rate(t)*t*(m-eJ)

    def var_cj(self,t):
        m  = self.mJ[1]

        return self.rate(t)*t*m

    def cumulant3_cj(self,t):
        return self.rate(t)*t*self.mJ[2]

    def cumulant4_cj(self,t):
        return self.rate(t)*t*self.mJ[3]

    def cf_cj(self,u,v,t,Q):
        eu = self.cfJ(u)

        return np.exp(self.rate(1)*t*(eu-1-1j*u*self.eJ))

                    