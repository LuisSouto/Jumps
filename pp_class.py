#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 2022

@author: Luis Antonio Souto Arias

Basis for the point process class. It contains the basic methods that
the three classes Poisson, Hawkes and QHawkes should have in common.

The notation and meaning of the parameters is extracted from Souto, Cirillo and
Oosterlee (2022): A new self-exciting jump-diffusion model for option pricing.
https://doi.org/10.48550/arXiv.2205.13321
"""

class PointP():
    def __init__(self,a,b,hb,Q0,mJ,eJ,cfJ):
        self.set_param(a,b,hb,Q0,mJ,eJ,cfJ)

    def set_param(self,a=None,b=None,hb=None,Q0=None,mJ=None,eJ=None,cfJ=None):
        """ Selects a new set of parameters.

        Input
        -----
        a: float
           Clustering rate.
        b: float
           Expiration rate.
        hb: float
           Baseline intensity.
        Q0: int
           Initial value of the activation number.
        mJ: ndarray(dtype=float,shape=(4,))
           First four moments of the log-jump size distribution.
        eJ: float
           E[exp(Y)]-1, where Y is the log-jump size.
        cfJ: callable
           Characteristic function of Y.
        """

        # Set to old values in case new ones are not specified
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if hb is None:
            hb = self.hb
        if Q0 is None:
            Q0 = self.Q0
        if mJ is None:
            mJ = self.mJ
        if eJ is None:
            eJ = self.eJ
        if cfJ is None:
            cfJ = self.cfJ

        self.a  = a
        self.b  = b
        self.hb = hb
        self.Q0 = Q0
        self.h0 = hb + a*Q0
        self.mJ = mJ
        self.eJ = eJ
        self.cfJ = cfJ

    def simul(self,N,T):
        """ Monte Carlo simulation on the interval [0,T]

        Input
        -----
        N: int
           Number of Monte Carlo paths.
        T: float
           Final time of the simulation.

        Output
        ------
        Trajectories
        """
        pass

    def mean_cj(self,t):
        pass

    def var_cj(self,t):
        pass    

    def cumulant3_cj(self,t):
        pass    

    def cumulant4_cj(self,t):
        pass    

    def skewness_cj(self,t):
        return self.cumulant3_cj(t)/self.var_cj(t)**(1.5)

    def kurtosis_cj(self,t):
        return self.cumulant4_cj(t)/self.var_cj(t)**2+3    

    def cf_cj(self,u,v,t,Q):
        pass