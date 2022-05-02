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

# Density estimation for continuous random variables
def density(x,cf,cm):
    
    # Method parameters
    L = 8
    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    n = 2**6
    u = np.arange(n)/(b-a)*np.pi

    # Spectral filter
    #spu = np.exp(np.log(eps)*((0:n-1)/n).^10)
    #spu = spu(ones(lx,1),:)

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
    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))

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
    
    cos_term = np.dot((np.dot(cfu,np.cos((x2+0.5)*u2[:,np.newaxis])).T)[:,:-1]
                      ,np.cos((x1-a)*u1[:-1,np.newaxis]))

    return 2/((b-a)*N2)*cos_term    

# European call valuation
def call(S,K,T,r,cf,cm):
    
    # Method parameters
    L = 6
    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    N = 2**8
    u = np.arange(0,N)/(b-a)*np.pi
    x = np.log(S/K)
    
    # Compute characteristic function
    char_fun = np.real(cf(u[:,np.newaxis])*np.exp(1j*(x-a)*u[:,np.newaxis]))
    char_fun[0,:] /= 2
    
    # Compute European call term
    cos_term = (((-1)**np.arange(0,N)*np.exp(b)-np.cos(a*u)+u*np.sin(a*u))
                /(1+u**2))
    cos_term[0]  -= b
    cos_term[1:] -= np.sin(a*u[1:])/u[1:]
    
    return K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term,char_fun)


# European put valuation
def put(S,K,T,r,cf,cm):
    
    # Method parameters
    L = 6
    a = cm[0]-L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0]+L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    N = 2**9
    u = (np.arange(0,N)/(b-a)*np.pi)
    x = np.log(S/K)
    
    # Compute characteristic function
    char_fun = np.real(cf(u[:,np.newaxis])*np.exp(1j*(x-a)*u[:,np.newaxis]))
    char_fun[0,:] /= 2

    # Compute European call term
    cos_term = -(np.cos(a*u)-np.exp(a)-u*np.sin(a*u))/(1+u**2)
    cos_term[0]  -= a
    cos_term[1:] -= np.sin(a*u[1:])/u[1:]
    
    return K*2/(b-a)*np.exp(-r*T)*np.dot(cos_term,char_fun)


# Bermudan put valuation for Levy processes. Adaption of Marion's Matlab code
def bermudan_put(S,K,T,r,cm,cfS,M):

    # Parameters
    x = np.log(S/K)

    if not np.isscalar(x):
        x = x[:,np.newaxis]

    # Time step
    dt = T/M
    
    # Interval [a,b]
    L = 10
    a = cm[0] - L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0] + L*np.sqrt(cm[1]+np.sqrt(cm[2]))

    # Number of Fourier cosine coefficients
    N = 2**9
    k = np.arange(0,N)*np.pi/(b-a)

    # Fourier cosine coefficients payoff function
    sin_ka = -np.sin(k*a)
    cos_ka = np.cos(k*a)
    chi    = (cos_ka-np.exp(a)+k*sin_ka)/(1+k**2)
    psi    = sin_ka/k
    psi[0] = -a    
    Vk     = (-chi+psi)
    Vk[0] *= 0.5
    
    # Characteristic function
    cf = cfS(k,dt)

    # xs is the early-exercise point where c = g,
    xs = 0 # initial value

    for m in range(M-1,0,-1):

        # Newton-Raphson iterations to find the early-exercise point
        # where f = c-g = 0
        for NR in range(5):

            # Fourier cosine coefficients density function
            Recf   = np.real(cf*np.exp(1j*k*(xs-a)))
            Recf_x = -k*np.imag(cf*np.exp(1j*k*(xs-a)))
            
            # Continuation and payoff value and derivatives
            c   = np.exp(-r*dt)*2/(b-a)*K*(np.dot(Vk,Recf))
            c_x = np.exp(-r*dt)*2/(b-a)*K*(np.dot(Vk,Recf_x))
            g   = 0
            g_x = 0

            if (xs <= 0):
                g   = K*(1-np.exp(xs))  # x = log(S/K),S = Kexp(x)
                g_x = -K*np.exp(xs)

            f   = c-g
            f_x = c_x-g_x

            # Next approximation to the root
            xs = xs-f/f_x

        if (xs < a):
            xs = a
        elif (xs > b):
            xs = b

        # Fourier cosine coefficients payoff function Gk(a,x2)            
        omegaxsmina = k*(xs-a)
        sin_omegaxsmina = np.sin(omegaxsmina)
        cos_omegaxsmina = np.cos(omegaxsmina)
        chi    = (cos_omegaxsmina*np.exp(xs)-np.exp(a)
                  +k*sin_omegaxsmina*np.exp(xs))/(1+k**2)
        psi    = sin_omegaxsmina/k
        psi[0] = xs-a
        G = (-chi+psi)
        
        # Fourier cosine coefficients continuation value Ck(t_m,xs,b)
        C = Cvalue(xs,b,N,a,b,k[1:],cf,Vk,dt,r)
        
        # Fourier cosine coefficients option value Vk(t_m)
        Vk     = C+G
        Vk[0] *= 0.5

    # Fourier cosine coefficients density function
    Recf = np.real(cf*(np.exp(1j*k*(x-a)))).T

    # Option value
    return np.exp(-r*dt)*2/(b-a)*K*(np.dot(Vk,Recf))

# Bermudan put valuation for the Heston model (Fang)
def bermudan_put_nolevy(S,K,T,r,cm,a2,b2,cfS,M):

    # Parameters
    x  = np.log(S/K)
    dt = T/M
    
    # Interval [a,b]
    L = 12
    a = cm[0] - L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b = cm[0] + L*np.sqrt(cm[1]+np.sqrt(cm[2]))

    # Number of Fourier cosine coefficients
    N = 2**7
    k = (np.arange(0,N)*np.pi/(b-a))[:,np.newaxis]

    # Fourier cosine coefficients payoff function
    sin_ka    = -np.sin(k*a)
    cos_ka    = np.cos(k*a)
    Vk        = -(cos_ka-np.exp(a)+k*sin_ka)/(1+k**2)
    Vk[1:,:] += sin_ka[1:,:]/k[1:,:]
    Vk[0,:]  -= a    
    Vk[0,:]  *= 0.5
    
    # Get the quadrature points
    J     = 2**7
    vi,wi = quad.gauss_legendre(a2,b2,J)
    vi    = np.exp(vi)

    # Product of the characteristic function and the density evaluated on the
    # quadrature nodes
    cf  = cfS(k,dt,vi,vi[:,np.newaxis,np.newaxis])*wi

    # xs is the early-exercise point where c = g,
    xs = np.zeros((J,)) # initial value

    for m in range(M-1,0,-1):

        # Newton-Raphson iterations to find the early-exercise point
        # where f = c-g = 0
        Bk  = (Vk*cf).sum(-1).T # Should be size N x J
        for NR in range(5):

            # Fourier cosine coefficients density function
            cre = np.real((Bk*np.exp(1j*k*(xs-a))).sum(0))
            cim = -(k*np.imag((Bk*np.exp(1j*k*(xs-a))))).sum(0)
            
            # Continuation and payoff value and derivatives
            c   = np.exp(-r*dt)*2/(b-a)*K*cre
            c_x = np.exp(-r*dt)*2/(b-a)*K*cim
            g   = np.zeros((J,))
            g_x = np.zeros((J,))

            g[xs<=0]   = K*(1-np.exp(xs[xs<=0]))           
            g_x[xs<=0] = -K*np.exp(xs[xs<=0])

            f   = c-g
            f_x = c_x-g_x

            # Next approximation to the root
            xs = xs-f/f_x

        xs[xs<a] = a
        xs[xs>b] = b

        # Uncomment if you wanna plot exercise region
        # plt.figure()
        # plt.plot(vi,xs)
        # plt.show()
        
        # Fourier cosine coefficients payoff function Gk(a,x2)            
        omegaxsmina     = k*(xs-a)
        sin_omegaxsmina = np.sin(omegaxsmina)
        cos_omegaxsmina = np.cos(omegaxsmina)
        G        = -(cos_omegaxsmina*np.exp(xs)-np.exp(a)
                     +k*sin_omegaxsmina*np.exp(xs))/(1+k**2)
        G[1:,:] += sin_omegaxsmina[1:,:]/k[1:,:]
        G[0,:]  += xs-a

        # Fourier cosine coefficients continuation value Ck(t_m,xs,b)
        C = Cvalue_par(xs,b,N,J,a,b,k[1:],Bk,dt,r)

        # Fourier cosine coefficients option value Vk(t_m)
        Vk       = C+G
        Vk[0,:] *= 0.5

    # Fourier cosine coefficients density function
    Bk   = (Vk*cf).sum(2)[:,:,np.newaxis]
    ccom = (Bk*np.exp(1j*k*(x-a))).sum(1)    

    # Option value
    return vi,np.exp(-r*dt)*2/(b-a)*K*np.real(ccom)

# Bermudan put valuation using the 2D COS method (quadrature)
def bermudan_put_nolevy_v2(S,K,T,r,cm,a2,b2,cfS,M,jump=False):

    # Parameters
    x = np.log(S/K)

    # Time step
    dt = T/M
    
    # Interval [a,b]
    L = 7
    a1 = cm[0] - L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b1 = cm[0] + L*np.sqrt(cm[1]+np.sqrt(cm[2]))

    # Number of Fourier cosine coefficients
    N1 = 2**8
    N2 = 2**7
    J  = 2**7

    if jump==True:
        a2 = -0.5
        b2 = N2-0.5
        vi = np.arange(N2)
        wi = np.ones((N2,))
    elif jump==False:
        vi,wi = quad.gauss_legendre(a2,b2,J)   # Quadrature nodes
        # vi,wi = quad.trapezoidal(a2,b2,J)
        # vi,wi = quad.nonuni_trap(a2,b2,J,1)        

    k1  = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis]
    k2  = (np.arange(N2)*np.pi/(b2-a2))
    wi /= (b2-a2)
    
    # Fourier cosine coefficients payoff function
    sin_ka    = -np.sin(k1*a1)
    cos_ka    = np.cos(k1*a1)
    Vk        = -(cos_ka-np.exp(a1)+k1*sin_ka)/(1+k1**2)
    Vk[1:,:] += sin_ka[1:,:]/k1[1:,:]
    Vk[0,:]  -= a1    
    Vk[0,:]  *= 0.5

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

        # Newton-Raphson iterations to find the early-exercise point
        # where f = c-g = 0
        Bk  = (Vk*cf).sum(-1).T
        for NR in range(5):

            # Fourier cosine coefficients density function
            cre = np.real((Bk*np.exp(1j*k1*(xs-a1))).sum(0))
            cim = -(k1*np.imag((Bk*np.exp(1j*k1*(xs-a1))))).sum(0)
            
            # Continuation and payoff value and derivatives
            c   = np.exp(-r*dt)*2/(b1-a1)*K*cre
            c_x = np.exp(-r*dt)*2/(b1-a1)*K*cim
            g   = np.zeros((J,))
            g_x = np.zeros((J,))

            g[xs<=0]   = K*(1-np.exp(xs[xs<=0]))         
            g_x[xs<=0] = -K*np.exp(xs[xs<=0])

            f   = c-g
            f_x = c_x-g_x

            # Next approximation to the root
            xs = xs-f/f_x

        xs[xs<a1] = a1
        xs[xs>b1] = b1
        
        # Uncomment to plot exercise region
        # plt.figure()
        # plt.plot(vi,xs)
        # plt.show()

        # Fourier cosine coefficients payoff function Gk(a,x2)            
        omegaxsmina     = k1*(xs-a1)
        sin_omegaxsmina = np.sin(omegaxsmina)
        cos_omegaxsmina = np.cos(omegaxsmina)
        G        = -(cos_omegaxsmina*np.exp(xs)-np.exp(a1)
                     +k1*sin_omegaxsmina*np.exp(xs))/(1+k1**2)
        G[1:,:] += sin_omegaxsmina[1:,:]/k1[1:,:]
        G[0,:]  += xs-a1

        # Fourier cosine coefficients continuation value Ck(t_m,xs,b)
        C = Cvalue_par(xs,b1,N1,J,a1,b1,k1[1:],Bk,dt,r)

        # Fourier cosine coefficients option value Vk(t_m)
        Vk       = C+G
        Vk[0,:] *= 0.5
        Vk       = np.dot(Vk,Vk2)

    # Fourier cosine coefficients density function
    Bk   = (Vk*cf).sum(2)[:,:,np.newaxis]
    ccom = (Bk*np.exp(1j*k1*(x-a1))).sum(1)

    # Option value
    return vi,np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom)    

# Bermudan put valuation using the 3D COS method (quadrature)
def bermudan_put_3D(S,K,T,r,cm,aV,bV,cfV,cfQ,M,nQ=2**5):
    # Parameters
    x  = np.log(S/K)
    dt = T/M

    # Interval [a,b]
    L = 8
    a1 = cm[0] - L*np.sqrt(cm[1]+np.sqrt(cm[2]))
    b1 = cm[0] + L*np.sqrt(cm[1]+np.sqrt(cm[2]))

    # Number of Fourier cosine coefficients
    N1 = 2**7
    nV = 2**7

    # vi,wi = quad.trapezoidal(a2,b2,J)
    vV,wV = quad.gauss_legendre(aV,bV,nV)
    vV = np.exp(vV)
    vQ = np.arange(nQ)
    wQ = np.ones((nQ,))

    k1  = (np.arange(N1)*np.pi/(b1-a1))[:,np.newaxis,np.newaxis]
    wQ /= nQ
    
    # Fourier cosine coefficients payoff function
    sin_ka  = -np.sin(k1*a1)
    cos_ka  = np.cos(k1*a1)
    Vk      = -(cos_ka-np.exp(a1)+k1*sin_ka)/(1+k1**2)
    Vk[1:] += sin_ka[1:]/k1[1:]
    Vk[0]  -= a1
    Vk[0]  *= 0.5
    Vk      = np.tile(Vk,(1,nV,nQ))

    # Integrated characteristic function
    cf1 = cfV(k1,dt,vV,vV[:,np.newaxis])*wV
    cf2 = cfQ(k1.squeeze(),dt,nQ-1,nQ-1)
    cf2 = cf2.transpose(2,1,0)

    # xs is the early-exercise point where c = g,
    xs = np.zeros((nV,nQ)) # initial value
    Bk = np.zeros((N1,nV,nQ),dtype=complex)

    for m in range(M-1,0,-1):
        # Summations in V and Q for the continuation value
        for i in range(N1):
            Bk[i]  = np.dot(cf1[i],np.dot(Vk[i],cf2[i]))

        # Newton-Raphson iterations to find the early-exercise point
        # where f = c-g = 0
        for NR in range(5):
            # Fourier cosine coefficients density function
            cre = np.real((Bk*np.exp(1j*k1*(xs-a1))).sum(0))
            cim = -(k1*np.imag((Bk*np.exp(1j*k1*(xs-a1))))).sum(0)
            
            # Continuation and payoff value and derivatives
            c   = np.exp(-r*dt)*2/(b1-a1)*K*cre
            c_x = np.exp(-r*dt)*2/(b1-a1)*K*cim
            g   = np.zeros((nV,nQ))
            g_x = np.zeros((nV,nQ))

            g[xs<=0]   = K*(1-np.exp(xs[xs<=0]))         
            g_x[xs<=0] = -K*np.exp(xs[xs<=0])

            f   = c-g
            f_x = c_x-g_x

            # Next approximation to the root
            xs = xs-f/f_x

        xs[xs<a1] = a1
        xs[xs>b1] = b1
        xs[:] = a1
        
        # Uncomment to plot exercise region
        # plt.figure()
        # plt.plot(vi,xs)
        # plt.show()

        # Fourier cosine coefficients payoff function Gk(a,x2)            
        omegaxsmina     = k1*(xs-a1)
        sin_omegaxsmina = np.sin(omegaxsmina)
        cos_omegaxsmina = np.cos(omegaxsmina)
        G        = -(cos_omegaxsmina*np.exp(xs)-np.exp(a1)
                     +k1*sin_omegaxsmina*np.exp(xs))/(1+k1**2)
        G[1:] += sin_omegaxsmina[1:]/k1[1:]
        G[0]  += xs-a1

        # Fourier cosine coefficients continuation value Ck(t_m,xs,b)
        C = Cvalue_3D(xs,b1,N1,nV,nQ,a1,b1,k1[1:],Bk,dt,r)

        # Fourier cosine coefficients option value Vk(t_m)
        Vk     = C+G
        Vk[0] *= 0.5

    # Fourier cosine coefficients density function
    for i in range(N1):
        Bk[i] = np.dot(cf1[i],np.dot(Vk[i],cf2[i]))
    ccom = np.dot(Bk.transpose((1,2,0)),np.exp(1j*k1.squeeze(axis=-1)*(x-a1)))

    # Option value
    return vV,vQ,np.exp(-r*dt)*2/(b1-a1)*K*np.real(ccom)


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


# Continuation value (scalar)
def Cvalue(x1, x2, N, a, b, temp, cf, U, dt, r):
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

    k = np.arange(1,N)
            
    # Elements m_j
    mj   = (np.exp(1j*(x2-a)*temp)-np.exp(1j*(x1-a)*temp))/k        
    mj_0 = 1j*np.pi*(x2-x1)/(b-a)        
    mj_N = (np.exp(1j*N*np.pi*(x2-a)/(b-a))-np.exp(1j*N*np.pi*(x1-a)/(b-a)))/N  
    mj_minus = -np.conj(mj) 
    mfactor1 = np.exp(1j*N*np.pi*(x2-a)/(b-a))        
    mfactor2 = np.exp(1j*N*np.pi*(x1-a)/(b-a))        
    mj_add   = ((mfactor1*np.exp(1j*(x2-a)*temp)-mfactor2*np.exp(1j*(x1-a)*temp))
                /(k+N))

    ms = np.concatenate(([mj_0],mj_minus,[0],mj[::-1]))
    mc = np.concatenate((mj_add[::-1],[mj_N],mj[::-1],[mj_0]))
    uj = cf*U
    us = np.concatenate((uj,np.zeros((N,))))
            
    # Matrix-vector mulitplication M_s*u with the help of FFT algorithm
    fftu = fft(us)
    Msu  = ifft(fft(ms)*fftu)
    Msu  = Msu[:N]
    
    # Matrix-vector mulitplication M_c*u with the help of FFT algorithm
    sgnvector = np.ones((2*N,))
    sgnvector[1::2] = -1
    Mcu = ifft(fft(mc)*sgnvector*fftu)
    Mcu = Mcu[:N]
    Mcu = Mcu[::-1]

    # Fourier cosine coefficients Ck(t_m,x1,x2)
    return (np.exp(-r*dt)/np.pi)*(Msu+Mcu).imag

# Continuation value (vector)
def Cvalue_par(x1, x2, N, J, a, b, temp, uj, dt, r):
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

    k = np.arange(1,N)[:,np.newaxis]
            
    # Elements m_j
    mj   = (np.exp(1j*(x2-a)*temp)-np.exp(1j*(x1-a)*temp))/k        
    mj_0 = 1j*np.pi*(x2-x1)/(b-a)        
    mj_N = (np.exp(1j*N*np.pi*(x2-a)/(b-a))-np.exp(1j*N*np.pi*(x1-a)/(b-a)))/N  
    mj_minus = -np.conj(mj) 
    mfactor1 = np.exp(1j*N*np.pi*(x2-a)/(b-a))        
    mfactor2 = np.exp(1j*N*np.pi*(x1-a)/(b-a))        
    mj_add   = ((mfactor1*np.exp(1j*(x2-a)*temp)-mfactor2*np.exp(1j*(x1-a)*temp))
                /(k+N))

    ms = np.vstack(([mj_0],mj_minus,np.zeros((1,J)),mj[::-1,:]))
    mc = np.vstack((mj_add[::-1,:],[mj_N],mj[::-1,:],[mj_0]))
    us = np.vstack((uj,np.zeros((N,J))))
            
    # Matrix-vector mulitplication M_s*u with the help of FFT algorithm
    fftu = fft(us,axis=0)
    Msu  = ifft(fft(ms,axis=0)*fftu,axis=0)
    Msu  = Msu[:N,:]
    
    # Matrix-vector mulitplication M_c*u with the help of FFT algorithm
    sgnvector = np.ones((2*N,J))
    sgnvector[1::2,:] = -1
    Mcu = ifft(fft(mc,axis=0)*sgnvector*fftu,axis=0)
    Mcu = Mcu[:N,:]
    Mcu = Mcu[::-1,:]

    # Fourier cosine coefficients Ck(t_m,x1,x2)
    return (np.exp(-r*dt)/np.pi)*np.imag(Msu+Mcu)


def Cvalue_3D(x1, x2, N, nV, nQ, a, b, temp, uj, dt, r):
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

    k = np.arange(1,N)[:,np.newaxis,np.newaxis]

    # Elements m_j
    mj = (np.exp(1j*(x2-a)*temp)-np.exp(1j*(x1-a)*temp))/k
    mj_0 = 1j*np.pi*(x2-x1)/(b-a)
    mj_N = (np.exp(1j*N*np.pi*(x2-a)/(b-a))-np.exp(1j*N*np.pi*(x1-a)/(b-a)))/N
    mj_minus = -np.conj(mj)
    mfactor1 = np.exp(1j*N*np.pi*(x2-a)/(b-a))
    mfactor2 = np.exp(1j*N*np.pi*(x1-a)/(b-a))
    mj_add = ((mfactor1*np.exp(1j*(x2-a)*temp)-mfactor2*np.exp(1j*(x1-a)*temp))
              /(k+N))

    ms = np.vstack(([mj_0],mj_minus,np.zeros((1,nV,nQ)),mj[::-1]))
    mc = np.vstack((mj_add[::-1],[mj_N],mj[::-1],[mj_0]))
    us = np.vstack((uj,np.zeros((N,nV,nQ))))

    # Matrix-vector multiplication M_s*u with the help of FFT algorithm
    fftu = fft(us,axis=0)
    Msu  = ifft(fft(ms,axis=0)*fftu,axis=0)
    Msu  = Msu[:N]

    # Matrix-vector multiplication M_c*u with the help of FFT algorithm
    sgnvector = np.ones((2*N,nV,nQ))
    sgnvector[1::2] = -1
    Mcu = ifft(fft(mc,axis=0)*sgnvector*fftu, axis=0)
    Mcu = Mcu[:N]
    Mcu = Mcu[::-1]

    # Fourier cosine coefficients Ck(t_m,x1,x2)
    return (np.exp(-r*dt)/np.pi)*np.imag(Msu+Mcu)
##

