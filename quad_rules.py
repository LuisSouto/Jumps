#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 00:09:36 2022

Some cuadrature rules for numerical integration.

@author: luis_souto
"""

import numpy as np

def trapezoidal(a,b,N):
    xi = a + np.arange(N)*(b-a)/(N-1)
    wi = (b-a)/(2*(N-1))*np.ones((N,))
    wi[1:-1] *= 2
    
    return xi,wi

def gauss_legendre(a,b,N):
    xi,wi = np.polynomial.legendre.leggauss(N) # Nodes in [-1,1]
    xi    = a + (xi+1)/2*(b-a)                 # Equidistant mapping to [a,b]
    wi   *= (b-a)/2                            # Normalize the weights
    
    return xi,wi

# Trapezoidal rule adapted to a non-uniform grid concentrated on the initial 
# value "a". Larger values of U yield more non-uniformity.
def nonuni_trap(a,b,N,U):
    k  = np.arange(N)/(N-1)
    g  = U+np.sqrt(U*U+1)
    xi = a+(b-a)/(2*U)*(g**k-(1/g)**k)
    wi = (b-a)/(4*U*(N-1))*np.log(g)*(g**k+(1/g)**k)
    wi[1:-1] *=2
    
    return xi,wi