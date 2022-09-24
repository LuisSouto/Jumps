#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luis Antonio Souto Arias

@Software: PyCharm

This module contains functions related to the COS method of Fang and
Oosterlee (2008): A novel pricing method for European options based
on Fourier-cosine based series expansions. SIAM Journal on Scientific Computing.

The module contains the following functions:
  * density: Density estimation for continuous random variables.
  * density_dis: Density estimation for discrete random variables.
  * density_2D_dis: Same for 2D, with one continuous and one discrete variable.
  * vanilla: European call and put valuation.
  * bermudan_put: Bermudan put valuation for Levy processes.
  * bermudan_put_3D: Bermudan put valuation for processes up to 3D.
  * american_put: American put valuation based on bermudan_put.
  * bermudan_coeff: Auxiliary function for bermudan_put(_3D).
  * Cvalue: Auxiliary function for bermudan_coeff.
  * newton_bermudan: Auxiliary function for bermudan_coeff.
  * vanilla_coeff: Cosine coefficients of vanilla call and put options.
  * trunc_inter: Truncation interval using rule of thumb.

"""

from scipy.fftpack import fft,ifft
from scipy.special import erf

import numpy as np
import quad_rules as quad

def density(x,cf,cm,N=2**6):
    """ Density estimation for continuous random variables.

    Input
    ----------
      x: ndarray(dtype=float,shape=(x.size,))
         Value of the random variable.
     cf: callable
         Characteristic function E[e^(i k x)].
     cm: ndarray(dtype=float,shape=(3,))
         First, second and fourth cumulants of x.
      N: int, optional, default=2**6
         Number of terms in the cosine expansion.

    Output
    ------
     fx: ndarray_like(x)
         Value of the density function at x.
    """

    L   = 6
    a,b = trunc_interval(cm,L)
    k   = np.arange(N)/(b-a)*np.pi

    # Compute characteristic function
    char_fun = np.real(cf(k)*np.exp(-1j*a*k))

    cos_term = np.cos((x-a)*k[:,np.newaxis])
    cos_term[0,:] *= 0.5

    fx = 2/(b-a)*np.dot(char_fun,cos_term)
    return fx

def density_dis(x,cf,N=2**6):
    """ Density estimation for discrete random variables.

    Input
    -----
    x: ndarray(dtype=int,shape=(x.size,))
       Value of the random variable.
    cf: callable
       Characteristic function E[e^(i k x)].
    N: int, optional, default=2**6
       Number of terms in the cosine expansion. It also determines the
       truncation range of x. Thus, x is assumed to live on the integers
       between 0 and N-1.

    Output
    ------
    fx: ndarray_like(x)
       Value of the probability mass function at x.
    """

    k = np.arange(N)/N*np.pi

    # Compute characteristic function
    char_fun = np.real(cf(k)*np.exp(1j*0.5*k))
    char_fun[0] /= 2    

    cos_term = np.cos((x+0.5)*k[:,np.newaxis])
    fx = 2/N*np.dot(char_fun,cos_term)
    return fx

def density_2D_dis(x1,x2,cf,cm,N1=2**6,N2=2**6):
    """ Probability density with one continuous (x1) and one discrete (x2) variable.

    Input
    -----
    x1: ndarray(dtype=float,shape=(x1.size,))
        Value of the continuous random variable.
    x2: ndarray(dtype=int,shape=(x2.size,))
        Value of the discrete random variable.
    cf: callable
        Characteristic function E[e^(i k1 x1 + i k2 x2)].
    N1: int, optional, default=2**6
        Number of COS terms for x1.
    N2: int, optional, default=2**6
        Number of COS terms for x2. It also determines the
        truncation range of x2. Thus, x2 is assumed to live on the integers
        between 0 and N2-1.

    Output
    ------
     fx: ndarray_like(x1+x2)
        Value of the density at (x1,x2).
    """

    L   = 12
    a,b = trunc_interval(cm,L)
    k1  = np.arange(N1)/(b-a)*np.pi
    k2  = np.arange(N2)/N2*np.pi

    # Compute characteristic function
    cfu = np.real(np.exp(-1j*a*k1[:,np.newaxis])\
                  *(cf(k1[:,np.newaxis],k2)*np.exp(1j*0.5*k2)\
                  +cf(k1[:,np.newaxis],-k2)*np.exp(-1j*0.5*k2)))
    cfu[0,:] *= 0.5
    cfu[:,0] *= 0.5
    
    fx = np.dot(np.dot(cfu,np.cos((x2+0.5)*k2[:,np.newaxis])).T[:,:-1]
                      ,np.cos((x1-a)*k1[:-1,np.newaxis]))
    fx *= 2/((b-a)*N2)
    return fx

def vanilla(S,K,T,r,cf,cm,N=2**6,alpha=-1):
    """ European call and put valuation

    Input
    -----
    S: ndarray(dtype=float,shape=(S.size,))
       Asset price.
    K: float
       Strike.
    T: float
       Maturity (in years).
    r: float
       Interest rate.
    cf: callable
       Characteristic function E[e^(i k log(S/S0))].
    cm: ndarray(dtype=float,shape=(3,))
       First, second and fourth cumulants of log(S/K).
    N: int, optional, default=2**6
       Number of COS terms.
    alpha: int, optional, default=-1
       Indicator for European calls (alpha=1) or puts (alpha=-1).

    Output
    ------
    P: ndarray_like(S)
       Price of the option.
    P_delta: ndarray_like(S)
       Delta of the option.
    """

    L   = 6
    a,b = trunc_interval(cm,L)
    k   = (np.arange(0,N)/(b-a)*np.pi)
    x   = np.log(S/K)

    char_fun = cf(k[:,np.newaxis])*np.exp(1j*(x-a)*k[:,np.newaxis])
    char_fun[0,:] /= 2
    cos_term = vanilla_coeff(a,b,k,alpha)

    P       = K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term,np.real(char_fun))
    P_delta = -K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term*k,np.imag(char_fun))
    return P,P_delta


# Bermudan put valuation for Levy processes. Adaption of Marion's Matlab code
def bermudan_put(S,K,T,r,cm,cfS,M,N=2**7):
    # Parameters
    x = np.log(S/K)

    if not np.isscalar(x):
        x = x[:,np.newaxis]

    # Time step
    dt = T/M
    
    # Interval [a,b]
    L = 6
    a,b = trunc_interval(cm,L)

    # Number of Fourier cosine coefficients
    k = np.arange(0,N)*np.pi/(b-a)

    # Fourier cosine coefficients payoff function
    Vk = vanilla_coeff(a,b,k,-1)
    Vk[0] *= 0.5
    
    # Characteristic function
    cf = cfS(k,dt)

    # xs is the early-exercise point where c = g,
    xs = 0  # initial value

    for m in range(M-1,0,-1):
        Bk = Vk*cf
        xs = newton_bermudan(a,b,k,xs,Bk,K,r,dt).squeeze()
        Vk = bermudan_coeff(a,b,k,N,xs,Bk,dt,r)

    # Fourier cosine coefficients density function
    Bk = Vk*cf
    ccom = (Bk*(np.exp(1j*k*(x-a)))).sum(0)
    # Option value
    return np.exp(-r*dt)*2/(b-a)*K*np.real(ccom)

def bermudan_put_3D(S,K,T,r,cm,M,cfV=None,cfQ=None,N1=2**6,n1=2**6,
                    n2=2**6):
    """ Bermudan put valuation using the COS method. It accepts a two-dimensional
    model like Heston (dim3=False) or a three-dimensional jump-diffusion with
    a two-dimensional diffusion (dim3=True) where the diffusion and jump contributions
    to the characteristic function are independent.

    Input
    -----
    S: ndarray(dtype=float,shape=(S.size,))
       Initial asset value.
    K: float
       Strike of the option.
    T: float
       Maturity of the option.
    r: float
      Risk-free interest rate.
    cm: ndarray(dtype=float,shape=(3,))
       First, second and fourth cumulants of log(S/K).
    cfV: callable
       Characteristic function of the two-dimensional model.
    cfQ: callable
       Characteristic function of the jump contribution.
    M: int
       Number of exercise dates.
    N1: int
       Number of COS terms in the asset dimension.
    n1: int
       Number of quadrature nodes for the second process (e.g., Heston variance).
    n2: int
       Number of quadrature nodes for the third process (e.g., Q-Hawkes).

    Output
    ------
    P: ndarray(dtype=float,shape=(nV,nQ,S.size))
       Price of the Bermudan put option.
    """

    # Parameters
    x     = np.log(S/K)
    dt    = T/M
    L     = 6.5                   # Determines domain size in the COS method. May require tuning.
    a1,b1 = trunc_interval(cm,L)
    k1    = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]

    cfvNone = cfV is None
    cfQNone = cfQ is None

    # Compute characteristic functions
    if not cfvNone:
        cf1,v1 = cfV(k1,dt,n1)                     # Characteristic function and nodes
        if not cfQNone:
            k1        = k1[:,np.newaxis]
            cf2,v2    = cfQ(k1,dt,n2)              # Characteristic function and nodes
            tile_size = (1,n1,n2)
        else:
            v2 = None
            tile_size = (1,n1)
    else:
        v1,v2 = None,None

    # Fourier cosine coefficients payoff function
    Vk     = vanilla_coeff(a1,b1,k1,-1)
    Vk[0] *= 0.5
    Vk     = np.tile(Vk,tile_size)

    # Apply Newton-Raphson to find exercise value
    xs = np.zeros_like(Vk[0])  # initial value
    Bk = np.zeros_like(Vk,dtype=complex)
    for m in range(M-1,0,-1):
        # Summations in V and Q for the continuation value
        if (not cfvNone) and (not cfQNone):
            for i in range(N1):
                Bk[i]  = np.dot(cf1[:,i,:],np.dot(Vk[i],cf2[i]))
        elif not cfvNone:
            Bk = (Vk*cf1).sum(-1).T
        xs = newton_bermudan(a1,b1,k1,xs,Bk,K,r,dt)
        Vk = bermudan_coeff(a1,b1,k1,N1,xs,Bk,dt,r)

    # Fourier cosine coefficients density function
    if (not cfvNone) and (not cfQNone):
        for i in range(N1):
            Bk[i] = np.dot(cf1[:,i,:],np.dot(Vk[i],cf2[i]))
        ccom = np.dot(Bk.transpose((1,2,0)),np.exp(1j*k1.squeeze(axis=-1)*(x-a1)))
    elif not cfvNone:
        Bk   = (Vk*cf1).sum(2)[:,:,np.newaxis]
        ccom = (Bk*np.exp(1j*k1*(x-a1))).sum(1)

    return np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom),v1,v2

def american_put(S,K,T,r,cm,cfS):
    # American put valuation using the COS method

    MRc    = 0

    if np.isscalar(S):
        U_Rich = np.zeros((4,1))
    else:
        U_Rich = np.zeros((4,S.size))

    for M in 2**(np.arange(2,6)):
        U_Rich[MRc,:] = bermudan_put(S,K,T,r,cm,cfS,M)
        MRc += 1 

    # 4-point Richardson extrapolation scheme (with k0=1, k1=2, k2=3)
    return 1/21*(64*U_Rich[3:,:]-56*U_Rich[2:-1,:]+14*U_Rich[1:-2,:]-U_Rich[:-3,:])

def bermudan_coeff(a, b, k, N, x, Bk, dt, r):
    """ Similar to vanilla_coeff but for Bermudan call and put options.

    Input
    -----
    a,b: float, float
       Truncation range.
    k: ndarray(dtype=float)
       Argument of the characteristic function. If dim3=False,
       k.shape=(N1,1), else k.shape=(N1,1,1)).
    N: int
       Number of COS terms.
    x: ndarray(dtype=float)
       Roots returned by newton_bermudan. If dim3=False, x.shape=(nV,),
       else x.shape=(nV,nQ)).
    Bk: ndarray(dtype=complex)
       Product of the characteristic function and the option coefficients Vk.
       If dim3=False, Bk.shape=(N1,nV), else Bk.shape=(N1,nV,nQ)).
    dt: float
       Time step.
    r: float
       Interest rate.

    Output
    ------
      Vk: ndarray(dtype=float,shape=Bk.shape)
         Coefficients of the Bermudan put or call option in the COS method.
    """

    # Fourier cosine coefficients payoff function G
    kxa = k*(x-a)
    sin_uxa = np.sin(kxa)
    cos_uxa = np.cos(kxa)
    G = -(cos_uxa*np.exp(x)-np.exp(a)+k*sin_uxa*np.exp(x))/(1+k**2)
    G[1:] += sin_uxa[1:]/k[1:]
    G[0]  += x-a

    # Fourier cosine coefficients continuation value C
    C = Cvalue(x,b,N,a,b,k,Bk,dt,r)
    Vk = C+G
    Vk[0] *= 0.5
    return Vk

def Cvalue(x1, x2, N, a, b, k, uj, dt, r):
    """ Continuation value

    Input
    -----
    x1: ndarray(dtype=float)
       Lower bound integral.
    x2: ndarray(dtype=float)
       Upper bound integral
    N: int
       Number of COS terms
    a,b: float, float
       Truncation range.
    k: ndarray(dtype=float)
       Argument of the characteristic function. If dim3=False,
       k.shape=(N1,1), else k.shape=(N1,1,1)).
    Bk: ndarray(dtype=complex)
       Product of the characteristic function and the option coefficients Vk.
       If dim3=False, Bk.shape=(N1,nV), else Bk.shape=(N1,nV,nQ)).
    dt: float
       Time step.
    r: float
       Interest rate.

    Output
    ------
    C: ndarray(dtype=float,shape=Bk.shape)
       Contribution of the continuation value to the cosine coefficients Vk.
    """

    factor = (b-a)/np.pi
    kN     = k+N/factor

    mj = (np.exp(1j*(x2-a)*k)-np.exp(1j*(x1-a)*k))/(k*factor)
    mj[0] = 1j*(x2-x1)/factor
    mj_minus = -np.conj(mj)
    mj_add = (np.exp(1j*(x2-a)*kN)-np.exp(1j*(x1-a)*kN))/kN

    mc = np.concatenate((mj_add[::-1],mj[::-1]),axis=0)
    ms = np.concatenate((mj_minus,[np.zeros_like(uj[0])],mj[:0:-1]),axis=0)
    us = np.concatenate((uj,np.zeros_like(uj)),axis=0)

    # Matrix-vector multiplication M_s*u with the help of FFT algorithm
    fftu = fft(us,axis=0)
    Msu  = ifft(fft(ms,axis=0)*fftu,axis=0)
    Msu  = Msu[:N]

    # Matrix-vector multiplication M_c*u with the help of FFT algorithm
    sgnvector = np.ones_like(us)
    sgnvector[1::2] = -1
    Mcu = ifft(fft(mc,axis=0)*sgnvector*fftu,axis=0)
    Mcu = Mcu[:N]
    Mcu = Mcu[::-1]

    # Fourier cosine coefficients C
    return np.exp(-r*dt)/np.pi*np.imag(Msu+Mcu)

def newton_bermudan(a, b, k, x, Bk, K, r, dt):
    """ Newton's method for Bermudan option pricing. It finds the
    point where the continuation value and the payoff coincide. The
    shapes of the ndarrays are written in terms of the inputs of
    bermudan_put_3D.

    Input
    -----
    a,b: float,float
       Truncation range.
    k: ndarray(dtype=float)
       Argument of the characteristic function. If dim3=False,
       k.shape=(N1,1), else k.shape=(N1,1,1)).
    x: ndarray(dtype=float)
       Initial guess. If dim3=False, x.shape=(nV,),
       else x.shape=(nV,nQ)).
    Bk: ndarray(dtype=complex)
       Product of the characteristic function and the option coefficients Vk.
       If dim3=False, Bk.shape=(N1,nV), else Bk.shape=(N1,nV,nQ)).
    K: float
       Strike.
    r: float
       Interest rate.
    dt: float
       Time step.

    Output
    ------
    xs: ndarray_like(x)
       Value of the asset where the continuation value C of the option and the
       payoff G coincide.
    """

    if np.isscalar(x):
        xs = np.array([x])
    else:
        xs = x

    for NR in range(5):
        # Fourier cosine coefficients density function
        cre = np.real((Bk*np.exp(1j*k*(xs-a))).sum(0))
        cim = -(k*np.imag((Bk*np.exp(1j*k*(xs-a))))).sum(0)

        # Continuation and payoff value and derivatives
        c   = np.exp(-r*dt)*2/(b-a)*K*cre
        c_x = np.exp(-r*dt)*2/(b-a)*K*cim
        g   = np.zeros_like(xs)
        g_x = np.zeros_like(xs)

        # Check for negative values
        idx = xs<=0
        g[idx]   = K*(1-np.exp(xs[idx]))
        g_x[idx] = -K*np.exp(xs[idx])

        f   = c-g
        f_x = c_x-g_x

        # Next approximation to the root
        xs = xs-f/f_x

    # If outside the boundaries
    xs[x<a] = a
    xs[x>b] = b

    if np.isscalar(x):
        return xs[0]
    else:
        return xs

def vanilla_coeff(a, b, k, alpha=-1):
    """ Cosine coefficients Vk for European call and put options
    Input
    -----
    a,b: float,float
       Truncation range.
    k: ndarray(dtype=float,shape=(k.size,))
       Argument of the characteristic function.
    alpha: int, optional, default=-1
       Indicator for European calls (alpha=1) or puts (alpha=-1).

    Output
    ------
    Vk: ndarray_like(k)
       Coefficients for European call and put optional.
    """

    if alpha==1:
        esgn = np.empty_like(k,int)
        esgn[::2]  = 1
        esgn[1::2] = -1
        Vk = np.exp(b)*esgn/(1+k**2)
    elif alpha==-1:
        Vk = np.exp(a)/(1+k**2)

    Vk += (-np.cos(a*k)+k*np.sin(a*k))/(1+k**2)
    Vk[0]  -= (alpha==-1)*a+(alpha==1)*b
    Vk[1:] -= np.sin(a*k[1:])/k[1:]
    return Vk

def trunc_interval(cm, L):
    """ Truncation interval, rule of thumb from Fang and Oosterlee (2008).

    Input
    -----
    cm: ndarray(dtype=float,shape=(3,))
       First, second and fourth cumulants.
    L: float
       Constant for integration range (see original paper).

    Output
    ------
    a,b: float,float
       Truncation range.
    """

    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    return a,b