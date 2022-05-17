#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 16:08:26 2022

@author: luis_souto

COS method implementation.
"""

from scipy.fftpack import fft,ifft

import numpy as np
import matplotlib.pyplot as plt
import quad_rules as quad

# Truncation interval, rule of thumb
def trunc_interval(cm,L):
    # Input:
    #  cm: First, second and fourth cumulants.
    #  L: Constant for integration range (see original paper)
    #
    # Output:
    #  Truncation range [a,b]

    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    return a,b

# Cosine coefficients for call and put options
def vanilla_coeff(a,b,u,alpha=1):
    if alpha==1:
        esgn = np.empty_like(u,int)
        esgn[::2]  = 1
        esgn[1::2] = -1
        vcoeff = esgn/(1+u**2)
    elif alpha==-1:
        vcoeff = np.exp(a)/(1+u**2)

    vcoeff     += (-np.cos(a*u)+u*np.sin(a*u))/(1+u**2)
    vcoeff[0]  -= (alpha==-1)*a+(alpha==1)*b
    vcoeff[1:] -= np.sin(a*u[1:])/u[1:]
    return vcoeff

# Newton's method for our particular application in Bermudan options
def newton_bermudan(a,b,k,x,Bk,K,r,dt):
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

        idx = xs<=0
        g[idx]   = K*(1-np.exp(xs[idx]))
        g_x[idx] = -K*np.exp(xs[idx])

        f   = c-g
        f_x = c_x-g_x

        # Next approximation to the root
        xs = xs-f/f_x

    xs[x<a] = a
    xs[x>b] = b
    # xs[:] = a1
    if np.isscalar(x):
        return xs[0]
    else:
        return xs

def bermudan_coeff(a,b,k,N,x,Bk,dt,r):
    # Fourier cosine coefficients payoff function Gk(a,x2)
    omegaxsmina     = k*(x-a)
    sin_omegaxsmina = np.sin(omegaxsmina)
    cos_omegaxsmina = np.cos(omegaxsmina)
    G        = -(cos_omegaxsmina*np.exp(x)-np.exp(a)
                 +k*sin_omegaxsmina*np.exp(x))/(1+k**2)
    G[1:] += sin_omegaxsmina[1:]/k[1:]
    G[0]  += x-a

    # Fourier cosine coefficients continuation value Ck(t_m,xs,b)
    C = Cvalue(x,b,N,a,b,k[1:],Bk,dt,r)
    Vk = C+G
    Vk[0] *= 0.5
    return Vk

# Density estimation for continuous random variables
def density(x,cf,cm):
    # Input:
    #  x: Value of the random variable
    #  cf: Characteristic function E[e^(i u X)], with argument u.
    #  cm: First, second and fourth cumulant. Usually the fourth is set to zero.
    #
    # Output:
    #  Value of the density function at x.
    
    # Method parameters
    L = 8
    a,b = trunc_interval(cm,L)
    n = 2**6    # Number of COS terms
    u = np.arange(n)/(b-a)*np.pi

    # Compute characteristic function
    char_fun = np.real(cf(u)*np.exp(-1j*a*u))

    cos_term = np.cos((x-a)*u[:,np.newaxis])
    cos_term[0,:] *= 0.5

    return 2/(b-a)*np.dot(char_fun,cos_term)

# Density estimation for discrete random variables
def density_dis(x,cf,N=2**6):
    
    # Method parameters
    u   = np.arange(N)/N*np.pi

    # Compute characteristic function
    char_fun = np.real(cf(u)*np.exp(1j*0.5*u))
    char_fun[0] /= 2    

    cos_term = np.cos((x+0.5)*u[:,np.newaxis])
    
    return 2/N*np.dot(char_fun,cos_term)

# Bivariate density with one discrete variable
def density_2D_dis(x1,x2,cf,cm):
    
    # Method parameters
    L = 12
    a,b = trunc_interval(cm,L)

    # Auxiliary vectors
    N1  = 2**8
    N2  = 2**8
    u1  = np.arange(N1)/(b-a)*np.pi
    u2  = np.arange(N2)/N2*np.pi

    # Compute characteristic function
    cfu = np.real(np.exp(-1j*a*u1[:,np.newaxis])
                  *(cf(u1[:,np.newaxis],u2)*np.exp(1j*0.5*u2)
                    +cf(u1[:,np.newaxis],-u2)*np.exp(-1j*0.5*u2)))
    cfu[0,:] *= 0.5
    cfu[:,0] *= 0.5
    
    cos_term = np.dot(np.dot(cfu,np.cos((x2+0.5)*u2[:,np.newaxis])).T[:,:-1]
                      ,np.cos((x1-a)*u1[:-1,np.newaxis]))

    return 2/((b-a)*N2)*cos_term    

# Call and put valuation
def vanilla(S,K,T,r,cf,cm,alpha=-1,compute_delta=False):
    
    # Method parameters
    L = 6
    a,b = trunc_interval(cm,L)
    N = 2**6
    u = (np.arange(0,N)/(b-a)*np.pi)
    x = np.log(S/K)
    
    # Compute characteristic function
    char_fun = cf(u[:,np.newaxis])*np.exp(1j*(x-a)*u[:,np.newaxis])
    char_fun[0,:] /= 2

    # Compute European put term
    cos_term = vanilla_coeff(a,b,u,alpha)
    P = K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term,np.real(char_fun))

    if compute_delta:
        P_S = -K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term*u,np.imag(char_fun))
        return P,P_S
    else:
        return P


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

# Bermudan put valuation using the 2D COS method (quadrature)
def bermudan_put_nolevy_v2(S,K,T,r,cm,a2,b2,cfS,M,jump=False,
                           N1=2**6,N2=2**6,J=2**6):
    # Parameters
    x  = np.log(S/K)
    dt = T/M
    L  = 7
    a1,b1 = trunc_interval(cm,L)

    if jump:
        a2 = -0.5
        b2 = N2-0.5
        vi = np.arange(N2)
        wi = np.ones((N2,))
    elif not jump:
        vi,wi = quad.gauss_legendre(a2,b2,J)   # Quadrature nodes

    k1  = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]
    k2  = (np.arange(N2)*np.pi/(b2-a2))
    wi /= (b2-a2)
    
    # Fourier cosine coefficients payoff function
    Vk = vanilla_coeff(a1,b1,k1,-1)
    Vk[0]  *= 0.5

    # Summation in N2
    Vk2       = np.cos(k2*(vi[:,np.newaxis]-a2))*wi[:,np.newaxis]
    Vk2[:,0] *= 0.5
    Vk        = np.dot(np.tile(Vk,(1,J)),Vk2)

    # Characteristic function 2D
    cf = (cfS(k1,k2,dt,vi[:,np.newaxis,np.newaxis])*np.exp(-1j*k2*a2)
          +cfS(k1,-k2,dt,vi[:,np.newaxis,np.newaxis])*np.exp(1j*k2*a2))

    # xs is the early-exercise point where c = g,
    xs = np.zeros((J,)) # initial value

    for m in range(M-1,0,-1):
        Bk = (Vk*cf).sum(-1).T
        xs = newton_bermudan(a1,b1,k1,xs,Bk,K,r,dt)
        Vk = bermudan_coeff(a1,b1,k1,N1,xs,Bk,dt,r)
        Vk = np.dot(Vk,Vk2)

    # Fourier cosine coefficients density function
    Bk   = (Vk*cf).sum(2)[:,:,np.newaxis]
    ccom = (Bk*np.exp(1j*k1*(x-a1))).sum(1)

    # Option value
    return vi,np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom)    

# Bermudan put valuation using the COS method. It accepts a two dimensional
# diffusion like Heston (dim3=False) or a three dimensional jump-diffusion with a
# two dimensional diffusion (dim3=True).
def bermudan_put_3D(S,K,T,r,cm,aV,bV,cfV,cfQ,M,N1=2**6,nV=2**6,nQ=2**4,dim3=True,jump=True):
    # Parameters
    x  = np.log(S/K)
    dt = T/M
    L  = 10
    a1,b1 = trunc_interval(cm,L)

    vV,wV = quad.gauss_legendre(aV,bV,nV)
    vV = np.exp(vV)
    k1 = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]
    if dim3:
        k1 = k1[:,np.newaxis]
    
    # Fourier cosine coefficients payoff function
    Vk = vanilla_coeff(a1,b1,k1,-1)
    Vk[0] *= 0.5
    if dim3:
        if jump:
            vQ = np.arange(nQ)
        if not jump:
            vQ,wQ = quad.gauss_legendre(0,nQ-1,nQ)

        Vk = np.tile(Vk,(1,nV,nQ))
        Bk = np.zeros_like(Vk, dtype=complex)
        # Integrated characteristic function
        cf1 = cfV(k1,dt,vV,vV[:,np.newaxis])*wV
        cf2 = cfQ(k1.squeeze(),dt,nQ-1,nQ-1)
        cf2 = cf2.transpose(2,1,0)
    else:
        Vk = np.tile(Vk,(1,nV))
        cf1 = cfV(k1,dt,vV,vV[:,np.newaxis,np.newaxis])*wV

    # xs is the early-exercise point where c = g,
    xs = np.zeros_like(Vk[0])  # initial value

    for m in range(M-1,0,-1):
        # Summations in V and Q for the continuation value
        if dim3:
            for i in range(N1):
                Bk[i]  = np.dot(cf1[i],np.dot(Vk[i],cf2[i]))
        else:
            Bk = (Vk*cf1).sum(-1).T  # Should be size N x J
        xs = newton_bermudan(a1,b1,k1,xs,Bk,K,r,dt)
        Vk = bermudan_coeff(a1,b1,k1,N1,xs,Bk,dt,r)

    # Fourier cosine coefficients density function
    if dim3:
        for i in range(N1):
            Bk[i] = np.dot(cf1[i],np.dot(Vk[i],cf2[i]))
        ccom = np.dot(Bk.transpose((1,2,0)),np.exp(1j*k1.squeeze(axis=-1)*(x-a1)))

        # Option value
        return vV,vQ,np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom)
    else:
        Bk   = (Vk*cf1).sum(2)[:,:,np.newaxis]
        ccom = (Bk*np.exp(1j*k1*(x-a1))).sum(1)

        return vV,np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom)

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

# Continuation value (vector)
def Cvalue(x1, x2, N, a, b, temp, uj, dt, r):
    #
    # Input:    x1      - Lower bound integral
    #           x2      - Upper bound integral
    #           N       - Number of terms series expansion
    #           a       - Left bound computational domain
    #           b       - Right bound computational domain
    #           temp    - Vector with values [1:N-1]*pi/(b-a)
    #           cf      - Characteristic function
    #           V       - Fourier cosine coefficients Vk(t_{m+1})
    #           dt      - Time-step
    #           r       - Risk-free interest rate
    #
    # Output:   C       - Fourier cosine coefficients Ck(t_m,x1,x2)

    k = temp*(b-a)/np.pi

    # Elements m_j
    mj = (np.exp(1j*(x2-a)*temp)-np.exp(1j*(x1-a)*temp))/k
    mj_0 = 1j*np.pi*(x2-x1)/(b-a)
    mj_N = (np.exp(1j*N*np.pi*(x2-a)/(b-a))-np.exp(1j*N*np.pi*(x1-a)/(b-a)))/N
    mj_minus = -np.conj(mj)
    mfactor1 = np.exp(1j*N*np.pi*(x2-a)/(b-a))
    mfactor2 = np.exp(1j*N*np.pi*(x1-a)/(b-a))
    mj_add = ((mfactor1*np.exp(1j*(x2-a)*temp)-mfactor2*np.exp(1j*(x1-a)*temp))
              /(k+N))

    mc = np.concatenate((mj_add[::-1],[mj_N],mj[::-1],[mj_0]),axis=0)
    ms = np.concatenate(([mj_0],mj_minus,[np.zeros_like(uj[0])],mj[::-1]),axis=0)
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

    # Fourier cosine coefficients Ck(t_m,x1,x2)
    return (np.exp(-r*dt)/np.pi)*np.imag(Msu+Mcu)
##

