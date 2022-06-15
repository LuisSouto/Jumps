#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:07:42 2022

@author: Luis Antonio Souto Arias

@Software: PyCharm

Class for the Poisson jump process. Notice from the PointP class that it
takes the same parameters as the Q-Hawkes process. This is mostly to have
everything as a single implementation and to compute the Poisson intensity
as a function of the Q-Hawkes parameters.
"""

import numpy as np

from pp_class import PointP

class Poisson(PointP):
        
    def rate(self,t):
        a  = self.a
        b  = self.b
        hb = self.hb
        h0 = self.h0

        return (b*hb/(a-b)+h0)*np.exp((a-b)*t)-b*hb/(a-b)

    def mean_cj(self,t):
        return self.rate(t)*t*(self.mJ[0]-self.eJ)

    def var_cj(self,t):
        return self.rate(t)*t*self.mJ[1]

    def cumulant3_cj(self,t):
        return self.rate(t)*t*self.mJ[2]

    def cumulant4_cj(self,t):
        return self.rate(t)*t*self.mJ[3]

    def cf_cj(self,u,v,t,Q):
        return np.exp(self.rate(1)*t*(self.cfJ(u)-1-1j*u*self.eJ))

                    