#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3/7/22 17:28

@Author: Luis Antonio Souto Arias

@Software: PyCharm

Contains pricing functions for European options. The pricing algorithm is
the COS method.
"""

import numpy as np
import cos_method as COS
import py_vollib.black_scholes.implied_volatility as bs

def price_vanilla(model,S,T,K,r,x0,alpha=-1):
    nT = T.size
    nK = K.size
    P  = np.zeros((nT,nK))
    PS = np.zeros_like(P)
    IV = np.zeros_like(P)
    option = (alpha==1)*'c'+(alpha==-1)*'p'
    for j in range(nT):
        # Characteristic function
        cfs = lambda u: model.cf_asset(u,T[j],x0)
        for k in range(nK):
            # Cumulants
            cm = model.cumulants_cos(T[j])
            cm[0] -= np.log(K[k])

            # Vanilla pricing with COS
            P[j,k],PS[j,k] = COS.vanilla(S,K[k],T[j],r,cfs,cm,alpha=alpha)
            IV[j,k] = bs.implied_volatility(P[j,k],S,K[k],T[j],r,option)

    return P,IV,PS