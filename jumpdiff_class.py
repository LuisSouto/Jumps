#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 23:55:53 2022

@author: Luis Antonio Souto Arias

@Software: PyCharm

Class which combines a diffusion and a jump process into a jump-diffusion model
for the log-asset price. The jump process affects only the asset price, and it
is independent of the diffusion processes. Examples of tested diffusion processes
are the Black-Scholes and Heston models. Examples of tested jump processes are
the Poisson, Hawkes and Q-Hawkes processes.
"""

import numpy as np

import cos_method as COS
import py_vollib.black_scholes.implied_volatility as bs
from pricing_functions import price_vanilla

class JumpDiffusion():
    def __init__(self,dp,jp):
        self.dp  = dp   # Diffusion class
        self.jp  = jp   # Jump class

    def cf(self,u,v,w,t,X):
        """ Joint characteristic function (CF) of all processes.

        Input
        -----
        u: ndarray(dtype=float)
           Argument of the CF related to the log-asset price
        v: ndarray(dtype=float)
           Argument of the CF related to the variance in the Heston model.
           It does not play any role for the GBM.
        u: ndarray(dtype=float)
           Argument of the CF related to the jump process.
        t: float
           Time at which the CF is evaluated
        X: ndarray(dtype=float,shape=(3,))
           Initial values of the log-asset price, variance and activation
           number, respectively.

        Output
        ------
        cf: ndarray(dtype=complex)
           Characteristic function. The shape depends on the shape of the
           inputs.
        """
        return self.dp.cf(u,v,t,X[:-1])*self.jp.cf_cj(u,w,t,X[-1])

    def cf_asset(self,u,t,X):
        return self.dp.cf(u,0,t,X[:-1])*self.jp.cf_cj(u,0,t,X[-1])

    def mean(self,t):
        return self.dp.mean(t)+self.jp.mean_cj(t)

    def var(self,t):
        return self.dp.var(t)+self.jp.var_cj(t)

    def cumulant4(self,t):
        return self.dp.cumulant4(t)+self.jp.cumulant4_cj(t)

    def cumulants_cos(self,t):
        """ Returns the cumulants for the rule of thumb in the COS method.
        The fourth cumulant is assumed to be zero.
        """
        return np.array([self.mean(t).mean(),self.var(t),0])

    def sens_an(self,S,T,K,v,param='a'):
        """ Sensitivity analysis of the implied volatility curves
         from European puts with respect to the Q-Hawkes parameters.

        Input
        -----
        S: float
           Asset price.
        T: ndarray(dtype=float)
           Maturity of the option.
        K: ndarray(dtype=float)
           Strike of the option.
        v: ndarray(dtype=object)
           Values of the parameters to be analyzed.
        param: list(dtype=str)
           List of the parameters that are modified.

        Output
        ------
        P: ndarray(dtype=float)
          Price of the put options for each strike, maturity and
          parameter configuration.
        IV: ndarray(dtype=float)
          Implied volatilities obtained from P.
        """
        if np.isscalar(T): T = np.array([T])
        if np.isscalar(K): K = np.array([K])
        if np.isscalar(v[0]): v = np.array([v]).T

        old_vars  = [vars(self.jp)[key] for key in param]
        old_param = dict(zip(param,old_vars))   # Save old configuration

        n  = v.shape[0]
        nT = T.size
        nK = K.size
        P  = np.zeros((n,nT,nK))
        IV = np.zeros((n,nT,nK))
        for i in range(n):
            param_dict = dict(zip(param,v[i]))
            self.jp.set_param(**param_dict)
            try:
                x0 = np.array([0,self.dp.V0,self.jp.Q0])
            except:
                x0 = np.array([0,self.jp.Q0])
            P[i],IV[i] = price_vanilla(S,T,K,self.dp.r,x0)

        self.jp.set_param(**old_param)
        return P.squeeze(),IV.squeeze()
