#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:07:42 2022

@author: Luis Antonio Souto Arias

Class related to the Q-Hawkes process. It contains attributes related to the
parameters of the model, as well as methods for computing the characteristic
function and performing simulations via the thinning algorithm.

The notation and meaning of the parameters is extracted from [1]: Souto, Cirillo
and Oosterlee (2022): A new self-exciting jump-diffusion model for option pricing.
https://doi.org/10.48550/arXiv.2205.13321
"""

from scipy.stats import nbinom
from scipy.special import binom
import numpy as np

from pp_class import PointP

class QHawkes(PointP):

    def cf(self,u,t,Q):
        """ Characteristic function of the activation number Qt.

        Input
        -----
        u: ndarray(dtype=float,shape=(u.size,))
           Argument of the characteristic function.
        t: float
           Difference between initial and current times.
        Q: ndarray(dtype=int,shape=(Q.size,1))
           Initial value of the process

        Output
        ------
        cf: ndarray(dtype=complex,shape=(Q.size,u.size))
           Characteristic function.
        """
        a  = self.a
        b  = self.b
        eu = np.exp(1j*u)
        et = np.exp((a-b)*t)

        tq = (b-a*eu-b*(1-eu)*et)/(b-a*eu-a*(1-eu)*et)
        th = (b-a)/(b-a*(et-eu*(et-1)))

        return (tq**Q)*(th**(self.hb/a))


    def dens_int(self,Q,t):
        """ PMF of the activation number

        Input
        -----
        Q: ndarray(dtype=float,shape=(Q.size,))
           Point at which the PMF is evaluated.
        t: float
           Difference between initial and current times.

        Output
        ------
        f: ndarray_like(Q)
          PMF evaluated at Q.
        """
        a  = self.a
        b  = self.b
        Q0 = self.Q0
        et = np.exp((a-b)*t)
        p  = (b-a)/(b-a*et)
        g  = b/(b-a)*(1-et)

        nbh = nbinom(self.h0/a,p)

        # Extension to allow vectorization
        mx = np.max(Q)
        xi = np.tile(np.arange(0,-Q0-1,-1),(mx+1,1))+np.arange(mx+1)[:,np.newaxis]
        qi = np.arange(Q0+1)

        # Check for obvious result
        if Q0<=0: return nbh.pmf(Q)

        th = nbh.pmf(xi)
        qh = binom(Q0,qi)*g**(Q0-qi)*(1-g)**qi

        return np.sum(th*qh,1)


    def simul(self,N,T):
        """ Thinning algorithm for the Q-Hawkes process.

        Input
        -----
        N: int
           Number of trajectories.
        T: float
           Final simulation time.

        Output
        ------
        Tx: ndarray(dtype=float,shape=(n,N))
           Jump arrival times.
        B: ndarray(dtype=bool,shape=(n,N))
           Indicator of whether the jump is upwards or downwards.
        """

        n  = 600   # Maximum number of allowed jumps per trajectory.
        X  = np.zeros((N,))
        Tx = np.zeros((n,N))
        B  = np.zeros((n,N))
        s  = np.zeros((N,))
        tr = np.ones((N,),int)
        kx = np.zeros((N,),int)

        X[:] = self.h0

        U    = np.random.uniform(0,1,(n,N))
        iter = 0
        while tr.sum()>0:  # As long as not all trajectories have reached T
          dt  = -np.log(U[2*iter,:])/(X*(1+self.b/self.a)-self.b/self.a*self.hb)
          s   = s+dt
          tr  = s<T   # Active trajectories
          B[kx[tr],tr]  = 2*(U[2*iter+1,tr]<=(X[tr]/(X[tr]*(1+self.b/self.a)-self.b/self.a*self.hb)))-1
          X[tr]        += B[kx[tr],tr]*self.a
          Tx[kx[tr],tr] = s[tr]
          iter += 1
          kx[tr] = iter

        return Tx,B

    def compute_intensity(self,T,Tx,B):
        """ Computes the activation and counting processes from the arrival times.

        Input
        -----
        T: ndarray(dtype=float,shape=(T.size,))
           Time at which we evaluate the activation and counting processes. It must
           be smaller or equal than the final time of the trajectories.
        Tx,B: Output from method simul.

        Output
        ------
        Ne: ndarray(dtype=int,shape=(N,T.size))
           Counting process.
        Qe: ndarray_like(Ne)
           Activation number.
        """
        if np.isscalar(T): T = np.array([T])

        Ne = np.zeros((Tx.shape[1],np.size(T)))
        Qe = np.zeros((Tx.shape[1],np.size(T)))
        for i in range(np.size(T)):
            idx = (Tx<=T[i])
            Ne[:,i] = (idx*(B>0)).sum(0)
            Qe[:,i] = self.Q0+(Ne[:,i]-(idx*(B<0)).sum(0))

        return Ne.squeeze(),Qe.squeeze()

    def inter_time_dist(self,t,Q):
        a = self.a
        b = self.b
        return 1-np.exp(-self.hb*t)*((a*np.exp(-(a+b)*t)+b)/(a+b))**Q

    def inter_time_pdf(self,t,Q):
        a = self.a
        b = self.b
        return ((a*np.exp(-(a+b)*t)+b)/(a+b))**(Q-1)*np.exp(-self.hb*t)\
               *(self.hb*(a*np.exp(-(a+b)*t)+b)/(a+b)+Q*a*np.exp(-(a+b)*t))

    def cf_cj(self,u,v,t,Q):
        """ Characteristic function of the compensated jump term Mt. This is
        exactly Equation (17) in [1].

        Input
        -----
        u: ndarray(dtype=float)
           Argument of the activation number
        v: ndarray(dtype=float)
           Argument of the compensated jump Mt.
        t: float
           Time between initial and current times.
        Q: ndarray(dtype=int)
           Initial values of the activation number

        Output
        ------
        cf: ndarray(dtype=complex)
           Characteristic function of the activation and compensated jump processes.
           The shape is some combination of the inputs, e.g. (u.size,v.size,Q.size).
        """
        a  = self.a
        b  = self.b
        iu = 1j*u*self.eJ
        eu = self.cfJ(u)
        ev = np.exp(1j*v)
        fu = np.sqrt((b+a*(1+iu))**2-4*a*b*eu)
        gu = (b+a*(1+iu-2*eu*ev))

        efu = np.exp(-t*fu)
        au1 = (fu+gu)
        au2 = efu*(fu-gu)
        f1  = 2*fu/(au1+au2)
        f2  = ((1-efu)*(2*b-ev*(b+a*(1+iu)))+ev*fu*(1+efu))/(au1+au2)

        return np.exp(self.hb*t/(2*a)*(b-a-a*iu-fu))*f1**(self.hb/a)*f2**Q

    def cf_integral(self,v,x,t,Q,vectorize=True):
        """ Integral of the characteristic function with respect to u. This is
        Equation (20) in [1].

        Input
        -----
        v: ndarray(dtype=float)
           Argument of the compensated jump Mt.
        x: int
           Argument of the inverse Fourier transform.
        t: float
           Time between initial and current times.
        Q: int
           Initial value of the activation number
        vectorize: bool
           If true, computes the integral for every combination of (0,...,x)
           and (0,...,Q). If false, it computed the integral for the vector (0,...,x)
           given Q.

        Output
        ------
        res: ndarray(dtype=complex)
           Inverse Fourier transform of the characteristic function with respect
           to u. If vectorize=True, shape=(Q+1,x+1,v.size). Else, shape=(x+1,v.size).
        """
        if not np.isscalar(x): x = x[-1]
        if not np.isscalar(Q): Q = Q[-1]

        # Auxiliary variables
        a  = self.a
        b  = self.b
        iu = 1j*v*self.eJ
        eu = self.cfJ(v)
        fu = np.sqrt((b+a*(1+iu))**2-4*a*b*eu)
        efu = np.exp(-fu*t)
        hu = (1+efu)*fu+(1-efu)*(b+a*(1+iu))
        h2 = ((1+efu)*fu-(1-efu)*(b+a*(1+iu)))
        p  = (1-efu)*a*2*eu/hu
        g  = 2*b*(1-efu)

        # Check for obvious result
        if np.min(Q)<=0: return (np.exp(self.hb*t/(2*a)*(b-a-fu-a*iu))
                                 *(2*fu/hu)**(self.hb/a)*binom(x+self.hb/a-1,x)*p**x)

        # Indices for all x and Q
        mxQ = np.minimum(x,Q)
        xi  = (np.arange(0,-mxQ-1,-1)[:,np.newaxis,np.newaxis]
               +np.arange(x+1)[:,np.newaxis])

        if not vectorize:
            qi  = np.arange(mxQ+1)[:,np.newaxis,np.newaxis]

            # Compute summation terms
            th = (np.exp(self.hb*t/(2*a)*(b-a-fu-a*iu))*(2*fu/hu)**(self.hb/a)
                  *binom(xi+self.hb/a+Q-1,xi)*p**xi)
            th[xi[:,:,0]<0,:] = 0
            qh = hu**(-Q)*binom(Q,qi)*g**(Q-qi)*h2**qi

            return np.sum(th*qh,0)

        elif vectorize:
            qi  = np.arange(1,Q+1)
            qi2 = qi[:,np.newaxis]
            qi3 = np.arange(Q+1)[:,np.newaxis,np.newaxis]

            # Compute summation terms
            qh = hu**(-qi2)*binom(qi2,qi3)*g**(qi2-qi3)*h2**qi3
            bn = binom(xi+self.hb/a+qi-1,xi)
            th = np.zeros((Q+1,x+1,v.size),dtype=complex)
            th[xi[:,:,0]>=0,:] = p**xi[xi[:,:,0]>=0,:]

            res = np.zeros((Q+1,x+1,v.size),dtype=complex)
            res[0] = th[0]*binom(xi[0]+self.hb/a-1,xi[0])
            for i in qi:
                res[i] = np.sum(th[:i+1]*bn[:i+1,:,i-1:i]*qh[:i+1,i-1:i,:],0)

            return res*(2*fu/hu)**(self.hb/a)*np.exp(self.hb*t/(2*a)*(b-a-fu-a*iu))

    def mean_cj(self,t):
        """ Mean of the compensated jump term Mt (see [1])

        Input
        -----
        t: float
           Time at which we evaluate the mean.

        Output
        ------
        mt: float
           Mean of the compensated jump term Mt.
        """
        a  = self.a
        b  = self.b
        m  = self.mJ[0]
        eJ = self.eJ

        return (-a/(a-b)*(-1+np.exp((a-b)*t))*(eJ-m)*self.Q0-1/2*eJ*self.hb*t
               +1/2/(a-b)**2*self.hb*(a**2*eJ*t-b**2*(eJ-2*m)*t-2*a*
               np.exp((a-b)*t)*(eJ-m)+(-eJ+m+b*m*t)))

    def var_cj(self,t):
        a  = self.a
        b  = self.b
        et = np.exp((a-b)*t)
        m  = self.mJ[0:2]
        eJ = self.eJ

        f1 = self.hb/(b-a)**4*(2*a**4*et*eJ*(eJ-m[0])*t-b**4*m[1]*t+a*b**2*(et*(2*eJ*m[0]-
             2*m[0]**2-m[1])+m[1]*(1+3*b*t)+2*m[0]**2*(1-b*t)+2*eJ*m[0]*(b*t-1))-a**2*
             b*(8*eJ*m[0]+2*eJ**2*(b*t-2)-2*m[0]**2*(2+b*t)+m[1]*(2+3*b*t)+2*et*(-m[1]+
             2*m[0]**2*(1+b*t)+eJ**2*(2+b*t)-eJ*m[0]*(4+3*b*t)))+a**3*(-et**2*(eJ-m[0])**2-
             m[0]**2+m[1]-2*b*eJ*m[0]*t+m[1]*b*t+eJ**2*(1+2*b*t)-et*(m[1]-2*m[0]**2*(1+2*b*t)+
             2*eJ*m[0]*(1+2*b*t))))

        f2 = a*self.Q0/(b-a)**3*(-b**2*m[1]*(1-et)-2*a**3*et*eJ*(eJ-m[0])*t-a**2*(-et**2*(eJ-
             m[0])**2+eJ**2-m[0]**2+m[1]-et*(m[1]+2*(eJ-m[0])*m[0]*(1+2*b*t)))-a*b*(-
             et**2*(eJ-m[0])**2+eJ**2-4*eJ*m[0]+3*m[0]**2-2*m[1]-2*et*(-m[1]+(eJ-m[0])*
             (-m[0]+b*(eJ-2*m[0])*t))))

        return -(f1+f2)

    def cumulant3_cj(self,t):
        a  = self.a
        b  = self.b
        hb = self.hb
        Q0 = self.Q0
        m  = self.mJ[0:-1]
        eJ = self.eJ

        return(
            (a-b)**(-6)*np.exp(-3*b*t)*(a*(a-b)*Q0*(-b**4*(np.exp(3*b*t)
            -np.exp((a+2*b)*t))*m[2]+(-3*a**6*np.exp((a+2*b)*t)*eJ**2*
            (eJ-m[0])*t**2+(3*a**5*np.exp((a+b)*t)*eJ*t*(2*np.exp(a*t)
            *(eJ-m[0])**2+np.exp(b*t)*(m[1]+(-2*(m[0])**2*(1+2*b*t)+2*eJ*
            (m[0]+2*b*m[0]*t))))+(a*b**3*(-3*np.exp((2*a+b)*t)*(eJ-m[0])*m[1]
            +(np.exp(3*b*t)*(6*eJ*m[1]+(-9*m[0]*m[1]+4*m[2]))+np.exp((a+2*b)*t)
            *(-4*m[2]+(6*m[0]*(m[1]+2*b*m[1]*t)-3*eJ*(m[1]+3*b*m[1]*t)))))+
            (a**2*b**2*(-2*np.exp(3*a*t)*(eJ-m[0])**3+(np.exp(3*b*t)*(2*eJ**3+
            (-18*eJ**2*m[0]+(36*eJ*(m[0])**2+(-20*(m[0])**3+(-12*eJ*m[1]+
            (21*m[0]*m[1]-6*m[2]))))))+(-3*np.exp((2*a+b)*t)*(eJ-m[0])*
            (-m[1]+(2*b*eJ**2*t+(4*(m[0])**2*(1+b*t)-2*eJ*m[0]*(2+3*b*t))))-
            3*np.exp((a+2*b)*t)*(-2*m[2]+((b)**2*eJ**3*t**2+(-b*eJ**2*m[0]*
            t*(6+5*b*t)+(-eJ*m[1]*(3+8*b*t)+(6*m[0]*(m[1]+2*b*m[1]*t)+
            (-2*(m[0])**3*(1+(4*b*t+2*b**2*t**2))+2*eJ*(m[0])**2*(1+(7*b*t+
            4*b**2*t**2)))))))))))+(a**3*b*(-2*np.exp(3*a*t)*(eJ-m[0])**3+
            (np.exp(3*b*t)*(8*eJ**3+(-18*eJ**2*m[0]+(10*(m[0])**3+(6*eJ*m[1]+
            (-15*m[0]*m[1]+4*m[2])))))+(-3*np.exp((2*a+b)*t)*(eJ-m[0])*
            (2*(m[0])**2+(-m[1]+(2*eJ**2*(2+b*t)-2*eJ*m[0]*(3+b*t))))-
            np.exp((a+2*b)*t)*(4*m[2]+(6*eJ**3*(-1+2*b*t)+(9*eJ*(m[1]+2*b*m[1]*t)+
            (-18*m[0]*(m[1]+2*b*m[1]*t)+(6*eJ**2*m[0]*(3+(-b*t+2*b**2*t**2))+
            (6*(m[0])**3*(3+(6*b*t+4*b**2*t**2))-6*eJ*(m[0])**2*(5+(7*b*t+
            6*b**2*t**2)))))))))))+a**4*(-2*np.exp(3*a*t)*(eJ-m[0])**3+(np.exp(3*b*t)
            *(2*eJ**3+(-2*(m[0])**3+(3*m[0]*m[1]-m[2])))+(3*np.exp((2*a+b)*t)*
            (eJ-m[0])*(-m[1]+(2*b*eJ**2*t+((m[0])**2*(2+4*b*t)-2*eJ*(m[0]+3*b*
            m[0]*t))))+np.exp((a+2*b)*t)*(m[2]+(6*b*eJ**3*t*(2+b*t)+(-6*b*eJ**2
            *m[0]*t*(5+3*b*t)+(-6*m[0]*(m[1]+2*b*m[1]*t)+(6*(m[0])**3*(1+
            (2*b*t+2*b**2*t**2))+3*eJ*(m[1]+2*(m[0])**2*(-1+b*t))))))))))))))))
            -hb*(-b**6*np.exp(3*b*t)*m[2]*t+(3*a**7*np.exp((a+2*b)*t)*eJ**2*
            (eJ-m[0])*t**2+(a*b**4*(np.exp((a+2*b)*t)*(3*eJ*m[1]+(-6*m[0]*m[1]-
            m[2]))+np.exp(3*b*t)*(6*m[0]*m[1]+(m[2]+(-6*b*m[0]*m[1]*t+(5*b*m[2]*
            t+3*eJ*m[1]*(-1+b*t))))))+(-3*a**6*np.exp((a+b)*t)*eJ*t*(2*np.exp(a*t)
            *(eJ-m[0])**2+np.exp(b*t)*(m[1]+(-2*(m[0])**2*(1+2*b*t)+2*eJ*(m[0]+
            2*b*m[0]*t))))+(a**3*b**2*(-3*np.exp((2*a+b)*t)*(eJ-m[0])*(2*eJ*m[0]+
            (-2*(m[0])**2-m[1]))+(np.exp(3*b*t)*(eJ*(-96*(m[0])**2+21*m[1])+
            (6*eJ**2*m[0]*(14-3*b*t)+(6*eJ**3*(-3+b*t)+(6*(m[0])**3*(5+2*b*t)+
            (2*m[2]*(3+5*b*t)-3*m[0]*m[1]*(7+6*b*t))))))+3*np.exp((a+2*b)*t)*
            (-2*m[2]+(4*m[0]*m[1]*(2+3*b*t)+(-8*eJ*(m[1]+b*m[1]*t)+(-4*(m[0])**3
            *(2+(2*b*t+b**2*t**2))+((eJ)**3*(6+(4*b*t+b**2*t**2))+(4*eJ*(m[0])**2
            *(7+(6*b*t+2*b**2*t**2))-eJ**2*m[0]*(26+(20*b*t+5*b**2*t**2))))))))))+
            (-a**2*b**3*(2*np.exp(3*b*t)*(-9*b*m[0]*m[1]*t+(3*eJ**2*m[0]*(-2+b*t)+
            (6*(m[0])**3*(-2+b*t)+(m[2]*(2+5*b*t)+3*eJ*(m[1]+(b*m[1]*t+(m[0])**2
            *(6-3*b*t)))))))+np.exp((a+2*b)*t)*(6*eJ**2*m[0]*(2+b*t)+(4*(-m[2]+
            (3*b*m[0]*m[1]*t+3*(m[0])**3*(2+b*t)))-3*eJ*(6*(m[0])**2*(2+b*t)+m[1]
            *(2+3*b*t)))))+(a**4*b*(6*np.exp((2*a+b)*t)*(eJ-m[0])*(-m[1]+(2*
            (m[0])**2*(1+b*t)+((eJ)**2*(2+b*t)-eJ*m[0]*(4+3*b*t))))+(np.exp(3*b*t)
            *(-24*eJ**3+(-12*(m[0])**3+(18*eJ**2*m[0]*(2+b*t)+(6*m[0]*m[1]*(3+b*t)+
            (-m[2]*(4+5*b*t)+6*eJ*(-3*b*(m[0])**2*t+m[1]*(-2+b*t)))))))+2*np.exp(
            (a+2*b)*t)*(6*eJ**3*(1+b*t)+(6*b*eJ**2*m[0]*t*(2+b*t)+(9*eJ*(m[1]+
            b*m[1]*t)+(-6*eJ*(m[0])**2*(3+(7*b*t+3*b**2*t**2))+2*(m[2]+(6*(m[0])**3
            *(1+b*t)**2-3*m[0]*m[1]*(2+3*b*t)))))))))+a**5*(2*np.exp(3*a*t)*
            (eJ-m[0])**3+(np.exp(3*b*t)*(2*(m[0])**3+(-3*m[0]*m[1]+(m[2]+(6*b*eJ**2
            *m[0]*t+(-3*b*eJ*m[1]*t+(b*m[2]*t-2*eJ**3*(1+3*b*t)))))))+(3*np.exp(
            (2*a+b)*t)*(eJ-m[0])*(m[1]+(-2*(m[0])**2*(1+2*b*t)+2*eJ*(m[0]+2*b*m[0]*t)
            ))-np.exp((a+2*b)*t)*(m[2]+(6*b*eJ**3*t*(4+b*t)+(-6*b*eJ**2*m[0]*t*
            (8+3*b*t)+(-6*m[0]*(m[1]+2*b*m[1]*t)+(6*(m[0])**3*(1+(2*b*t+2*b**2*t**2))
            +3*eJ*(m[1]+(m[0])**2*(-2+4*b*t))))))))))))))))))
        )

    def cumulant4_cj(self,t):
        a  = self.a
        b  = self.b
        hb = self.hb
        Q0 = self.Q0
        m  = self.mJ
        eJ = self.eJ

        return 1 / 8 * np.exp((-a + b) * t) * Q0 * (24 * a ** 2 * (a-b) ** (-4) * np.exp((-a + b) * t) * ((-a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (2 * a * (b-a) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) + (2 * a * (-1 + np.exp(2 * (a - b) * t)) * (eJ - m[0]) * m[0] + (2 * (a - b) * np.exp((a - b) * t) * m[1] + (-a * (-1 + np.exp(2 * (a - b) * t)) * m[1] + (np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) + (a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (-2 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) * t + (4 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * t + ((a - b) * np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2))))))))))) ** 2 + (8 * a * (a-b) ** (-2) * (15 * a ** 3 * (a-b) ** (-6) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 + (12 * a ** 3 * (a-b) ** (-5) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - 2 * m[0]) + (12 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * m[1] + (6 * a * (-1 + np.exp(2 * (a - b) * t)) * (m[1]) ** 2 + (-18 * a ** 2 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) + (12 * a ** 2 * (b-a) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) * (a * eJ ** 2 - 2 * b * m[1]) + (12 * a * (b-a) ** (-1) * np.exp((a - b) * t) * m[1] * (a * eJ ** 2 - 2 * b * m[1]) + (3 * a * (a-b) ** (-2) * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 + (-8 * a * (a-b) ** (-2) * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] + (8 * a * (b-a) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] + (8 * a * b * (b-a) ** (-1) * np.exp((a - b) * t) * (eJ - 2 * m[0]) * m[2] + (-4 * a * (-1 + np.exp(2 * (a - b) * t)) * (eJ - m[0]) * m[2] + (4 * a * (-1 + np.exp(2 * (a - b) * t)) * m[0] * m[2] + (2 * b * np.exp((a - b) * t) * m[3] + (2 * (-a + b) * np.exp((a - b) * t) * m[3] + (a * (-1 + np.exp(2 * (a - b) * t)) * m[3] + (-15 * a ** 3 * (a-b) ** (-5) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t + (-12 * a ** 3 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - 2 * m[0]) * t + (-24 * a ** 4 * (a-b) ** (-5) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - m[0]) * t + (24 * a ** 3 * (a-b) ** (-3) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - m[0]) * m[0] * t + (-12 * a ** 2 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * m[1] * t + (12 * a ** 3 * (b-a) ** (-3) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * m[1] * t + (24 * a ** 2 * (b-a) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * m[1] * t + (24 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[0] * m[1] * t + (18 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) * t + (12 * a ** 2 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t + (24 * a ** 3 * (a-b) ** (-3) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t + (24 * a ** 2 * (b-a) ** (-1) * np.exp(2 * (a - b) * t) * (eJ - m[0]) * m[0] * (a * eJ ** 2 - 2 * b * m[1]) * t + (-12 * a * np.exp((a - b) * t) * m[1] * (a * eJ ** 2 - 2 * b * m[1]) * t + (12 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * m[1] * (a * eJ ** 2 - 2 * b * m[1]) * t + (-3 * a * (a-b) ** (-1) * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 * t + (-8 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t + (8 * a * (a-b) ** (-1) * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t + (8 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t + (-8 * a * b * np.exp((a - b) * t) * (eJ - 2 * m[0]) * m[2] * t + (16 * a ** 2 * (a-b) ** (-1) * b * np.exp(2 * (a - b) * t) * (eJ - m[0]) * m[2] * t + (2 * (a - b) * b * np.exp((a - b) * t) * m[3] * t + (3 * a ** 3 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** 2 + (48 * a ** 4 * (a-b) ** (-4) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - m[0]) * t ** 2 + (-48 * a ** 3 * (a-b) ** (-2) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - m[0]) * m[0] * t ** 2 + (12 * a ** 2 * (b-a) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * m[1] * t ** 2 + (24 * a ** 3 * (a-b) ** (-2) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * m[1] * t ** 2 + (12 * a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (-48 * a ** 3 * (a-b) ** (-2) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (-3 * a * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 * t ** 2 + (8 * a * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t ** 2 + (2 * a ** 3 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** 3 + (4 * a ** 3 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - 2 * m[0]) * t ** 3 + (32 * a ** 4 * (b-a) ** (-3) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * (eJ - m[0]) * t ** 3 + (-6 * a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) * t ** 3 - a ** 3 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** (4))))))))))))))))))))))))))))))))))))))))))))))))))) + (-48 * a ** 4 * (a-b) ** (-8) * np.exp(3 * (-a + b) * t) * ((a ** 2 * np.exp((a - b) * t) * eJ * t + (a * ((-1 + np.exp((a - b) * t)) * eJ + (m[0] + np.exp((a - b) * t) * m[0] * (1 - 2 * b * t))) + b * np.exp((a - b) * t) * (-m[0] * (3 + (np.exp((a - b) * t) - 2 * b * t)) + eJ * (1 + (np.exp((a - b) * t) - b * t)))))) ** 4 + (32 * a ** 2 * (a-b) ** (-4) * np.exp((-a + b) * t) * (3 * a ** 2 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 + (3 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - 2 * m[0]) + (6 * a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[1] + (3 * a * (-1 + np.exp(2 * (a - b) * t)) * (eJ - m[0]) * m[1] + (-3 * a * (-1 + np.exp(2 * (a - b) * t)) * m[0] * m[1] + (-3 * a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) + (3 * a * (b-a) ** (-1) * np.exp((a - b) * t) * (eJ - 2 * m[0]) * (a * eJ ** 2 - 2 * b * m[1]) + (2 * (a - b) * np.exp((a - b) * t) * m[2] + (-2 * b * np.exp((a - b) * t) * m[2] + (-a * (-1 + np.exp(2 * (a - b) * t)) * m[2] + (-3 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t + (-3 * a ** 2 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - 2 * m[0]) * t + (6 * a ** 3 * (b-a) ** (-3) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - m[0]) * t + (12 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * m[0] * t + (6 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[1] * t + (6 * a ** 2 * (b-a) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[1] * t + (3 * a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-3 * a * np.exp((a - b) * t) * (eJ - 2 * m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t + (6 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (eJ - m[0]) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-2 * (a - b) * b * np.exp((a - b) * t) * m[2] * t + (3 * a ** 2 * (b-a) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - 2 * m[0]) * t ** 2 + (12 * a ** 3 * (a-b) ** (-2) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (eJ - m[0]) * t ** 2 + (3 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3))))))))))))))))))))))) * (a * (-1 + np.exp(2 * (a - b) * t)) * eJ + (a ** 2 * np.exp((a - b) * t) * eJ * t + (-b * np.exp((a - b) * t) * (eJ - 2 * m[0]) * (-2 + b * t) - a * m[0] * (-1 + (np.exp(2 * (a - b) * t) + 2 * np.exp((a - b) * t) * (-1 + b * t)))))) + (-96 * a ** 3 * (a-b) ** (-6) * np.exp(2 * (-a + b) * t) * (-a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (2 * a * (b-a) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) + (2 * a * (-1 + np.exp(2 * (a - b) * t)) * (eJ - m[0]) * m[0] + (2 * (a - b) * np.exp((a - b) * t) * m[1] + (-a * (-1 + np.exp(2 * (a - b) * t)) * m[1] + (np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) + (a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (-2 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - 2 * m[0]) * t + (4 * a ** 2 * (a-b) ** (-1) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (eJ - m[0]) * t + ((a - b) * np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2)))))))))) * ((a * (-1 + np.exp(2 * (a - b) * t)) * eJ + (a ** 2 * np.exp((a - b) * t) * eJ * t + (-b * np.exp((a - b) * t) * (eJ - 2 * m[0]) * (-2 + b * t) - a * m[0] * (-1 + (np.exp(2 * (a - b) * t) + 2 * np.exp((a - b) * t) * (-1 + b * t))))))) ** 2 + (48 * a ** 4 * (a-b) ** (-8) * np.exp(5 * (a - b) * t) * ((a ** 2 * np.exp((-a + b) * t) * eJ * t + (-b * np.exp((-a + b) * t) * (eJ - 2 * m[0]) * (-2 + b * t) + a * (eJ + (-np.exp(2 * (-a + b) * t) * eJ + m[0] * (-1 + (np.exp(2 * (-a + b) * t) + np.exp((-a + b) * t) * (2 - 2 * b * t)))))))) ** 4 + (96 * a ** 3 * (a-b) ** (-6) * np.exp(2 * (-a + b) * t) * ((a ** 2 * np.exp((a - b) * t) * eJ * t + (a * ((-1 + np.exp((a - b) * t)) * eJ + (m[0] + np.exp((a - b) * t) * m[0] * (1 - 2 * b * t))) + b * np.exp((a - b) * t) * (-m[0] * (3 + (np.exp((a - b) * t) - 2 * b * t)) + eJ * (1 + (np.exp((a - b) * t) - b * t)))))) ** 2 * (2 * a * (a-b) ** (-1) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) + (-a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (-(1 + np.exp((a - b) * t)) * (-a + b * np.exp((a - b) * t)) * m[1] + (np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) + (2 * a * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * t + (a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + ((a - b) * np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + (a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 - 2 * a * (a-b) ** (-1) * np.exp(-2 * b * t) * (eJ - m[0]) * (-b ** 2 * (2 * np.exp(2 * a * t) + np.exp((a + b) * t)) * (eJ - 2 * m[0]) * t + (a * eJ * (np.exp(2 * b * t) + np.exp((a + b) * t) * (1 + a * t)) - b * (np.exp(2 * b * t) * m[0] + (np.exp(2 * a * t) * (-eJ + (m[0] + 2 * a * eJ * t)) + np.exp((a + b) * t) * (-eJ + (2 * m[0] + 2 * a * m[0] * t)))))))))))))) + (-24 * a ** 2 * (a-b) ** (-4) * np.exp((-a + b) * t) * ((2 * a * (a-b) ** (-1) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) + (-a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (-(1 + np.exp((a - b) * t)) * (-a + b * np.exp((a - b) * t)) * m[1] + (np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) + (2 * a * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * t + (a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + ((a - b) * np.exp((a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + (a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 - 2 * a * (a-b) ** (-1) * np.exp(-2 * b * t) * (eJ - m[0]) * (-b ** 2 * (2 * np.exp(2 * a * t) + np.exp((a + b) * t)) * (eJ - 2 * m[0]) * t + (a * eJ * (np.exp(2 * b * t) + np.exp((a + b) * t) * (1 + a * t)) - b * (np.exp(2 * b * t) * m[0] + (np.exp(2 * a * t) * (-eJ + (m[0] + 2 * a * eJ * t)) + np.exp((a + b) * t) * (-eJ + (2 * m[0] + 2 * a * m[0] * t))))))))))))))) ** 2 + (16 * a ** 2 * (a-b) ** (-4) * np.exp((-a + b) * t) * (-a ** 2 * np.exp((a - b) * t) * eJ * t + (-a * ((-1 + np.exp((a - b) * t)) * eJ + (m[0] + np.exp((a - b) * t) * m[0] * (1 - 2 * b * t))) - b * np.exp((a - b) * t) * (-m[0] * (3 + (np.exp((a - b) * t) - 2 * b * t)) + eJ * (1 + (np.exp((a - b) * t) - b * t))))) * (6 * a ** 2 * (b-a) ** (-3) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (6 * a ** 2 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 + (6 * a * (a-b) ** (-1) * np.exp((a - b) * t) * eJ * (a * eJ ** 2 - 2 * b * m[1]) + (-6 * a * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) + (-4 * b * np.exp((a - b) * t) * m[2] + (-2 * (1 + np.exp((a - b) * t)) * (-a + b * np.exp((a - b) * t)) * m[2] + (6 * a ** 2 * (a-b) ** (-2) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (-6 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t + (6 * a * np.exp((a - b) * t) * eJ * (a * eJ ** 2 - 2 * b * m[1]) * t + (6 * a * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-4 * (a - b) * b * np.exp((a - b) * t) * m[2] * t + (6 * a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 + (6 * a * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (2 * a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3 + (3 * a * (a-b) ** (-3) * (eJ - m[0]) * (a * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (-(a-b) ** 2 * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) + (4 * a * (a-b) ** 2 * np.exp(2 * (a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * t + (-2 * a * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (-2 * a * (a - b) * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (2 * (a-b) ** 2 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-2 * (a-b) ** 3 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-2 * a * (a-b) ** 2 * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 + (4 * a * (a - b) * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 - 2 * a * (a-b) ** 2 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2))))))))) + 6 * a * (a-b) ** (-1) * np.exp(-2 * b * t) * m[1] * (-b ** 2 * (2 * np.exp(2 * a * t) + np.exp((a + b) * t)) * (eJ - 2 * m[0]) * t + (a * eJ * (np.exp(2 * b * t) + np.exp((a + b) * t) * (1 + a * t)) - b * (np.exp(2 * b * t) * m[0] + (np.exp(2 * a * t) * (-eJ + (m[0] + 2 * a * eJ * t)) + np.exp((a + b) * t) * (-eJ + (2 * m[0] + 2 * a * m[0] * t))))))))))))))))))))) - 8 * a * (a-b) ** (-2) * (-12 * a ** 3 * (a-b) ** (-5) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 3 + (15 * a ** 3 * (a-b) ** (-6) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 + (12 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) + (-18 * a ** 2 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) + (3 * a * (a-b) ** (-2) * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 + (8 * a * (a-b) ** (-1) * b * np.exp((a - b) * t) * eJ * m[2] + (-8 * a * (a-b) ** (-2) * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] + (2 * b * np.exp((a - b) * t) * m[3] + ((1 + np.exp((a - b) * t)) * (-a + b * np.exp((a - b) * t)) * m[3] + (12 * a ** 3 * (a-b) ** (-4) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t + (-15 * a ** 3 * (a-b) ** (-5) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t + (-12 * a ** 2 * (a-b) ** (-2) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t + (18 * a ** 2 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) * t + (-3 * a * (a-b) ** (-1) * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 * t + (8 * a * b * np.exp((a - b) * t) * eJ * m[2] * t + (8 * a * (a-b) ** (-1) * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t + (2 * (a - b) * b * np.exp((a - b) * t) * m[3] * t + (3 * a ** 3 * (a-b) ** (-4) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** 2 + (12 * a ** 2 * (b-a) ** (-1) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (-3 * a * np.exp((a - b) * t) * ((a * eJ ** 2 - 2 * b * m[1])) ** 2 * t ** 2 + (8 * a * b * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * m[2] * t ** 2 + (-4 * a ** 3 * (a-b) ** (-2) * np.exp((a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3 + (2 * a ** 3 * (a-b) ** (-3) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** 3 + (-6 * a ** 2 * (a-b) ** (-1) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) * t ** 3 + (-a ** 3 * (a-b) ** (-2) * np.exp((a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 4 * t ** 4 + (3 * a * (a-b) ** (-3) * m[1] * (a * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ + b * (eJ - 2 * m[0])) ** 2 + (-(a-b) ** 2 * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ ** 2 - 2 * b * m[1]) + (4 * a * (a-b) ** 2 * np.exp(2 * (a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) * t + (-2 * a * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (-2 * a * (a - b) * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (2 * (a-b) ** 2 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-2 * (a-b) ** 3 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-2 * a * (a-b) ** 2 * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 + (4 * a * (a - b) * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 - 2 * a * (a-b) ** 2 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2))))))))) + (-2 * a * (a-b) ** (-5) * (eJ - m[0]) * (-3 * a ** 2 * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ + b * (eJ - 2 * m[0])) ** 3 + (3 * a * (a-b) ** 2 * ((1 + np.exp((a - b) * t))) ** 2 * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) + (2 * (a-b) ** 4 * b * ((1 + np.exp((a - b) * t))) ** 2 * m[2] + (-6 * a ** 2 * (a-b) ** 2 * np.exp(2 * (a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t + (6 * a ** 2 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t + (6 * a ** 2 * (a - b) * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t + (6 * a * (a-b) ** 4 * np.exp(2 * (a - b) * t) * eJ * (a * eJ ** 2 - 2 * b * m[1]) * t + (-6 * a * (a-b) ** 2 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-6 * a * (a-b) ** 3 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t + (-4 * (a-b) ** 4 * b * (a + b) * np.exp(2 * (a - b) * t) * m[2] * t + (4 * (a-b) ** 5 * b * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * m[2] * t + (12 * a ** 2 * (a-b) ** 3 * np.exp(2 * (a - b) * t) * eJ * (a * eJ + b * (eJ - 2 * m[0])) ** 2 * t ** 2 + (-12 * a ** 2 * (a - b) * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 2 + (-6 * a * (a-b) ** 4 * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (12 * a * (a-b) ** 3 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (-6 * a * (a-b) ** 4 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) * (a * eJ ** 2 - 2 * b * m[1]) * t ** 2 + (-6 * a ** 2 * (a-b) ** 3 * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3 + (8 * a ** 2 * (a-b) ** 2 * (a + b) * np.exp(2 * (a - b) * t) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3 - 2 * a ** 2 * (a-b) ** 3 * np.exp((a - b) * t) * (1 + np.exp((a - b) * t)) * (a * eJ + b * (eJ - 2 * m[0])) ** 3 * t ** 3)))))))))))))))))) - 4 * a * (a-b) ** (-1) * np.exp(-2 * b * t) * m[2] * (-b ** 2 * (2 * np.exp(2 * a * t) + np.exp((a + b) * t)) * (eJ - 2 * m[0]) * t + (a * eJ * (np.exp(2 * b * t) + np.exp((a + b) * t) * (1 + a * t)) - b * (np.exp(2 * b * t) * m[0] + (np.exp(2 * a * t) * (-eJ + (m[0] + 2 * a * eJ * t)) + np.exp((a + b) * t) * (-eJ + (2 * m[0] + 2 * a * m[0] * t)))))))))))))))))))))))))))))))))))))))))) - (a-b) ** (-8) * np.exp(-4 * b * t) * hb * (-b ** 8 * np.exp(4 * b * t) * m[3] * t + (4 * a ** 10 * np.exp((a + 3 * b) * t) * eJ ** 3 * (eJ - m[0]) * t ** 3 + (-a * b ** 6 * (np.exp((a + 3 * b) * t) * (6 * (m[1]) ** 2 + (-4 * eJ * m[2] + (8 * m[0] * m[2] + m[3]))) - np.exp(4 * b * t) * (8 * m[0] * m[2] + (m[3] + (-8 * b * m[0] * m[2] * t + (7 * b * m[3] * t + ((m[1]) ** 2 * (6 - 6 * b * t) + 4 * eJ * m[2] * (-1 + b * t))))))) + (-6 * a ** 9 * np.exp((a + 2 * b) * t) * eJ ** 2 * t ** 2 * (4 * np.exp(a * t) * (eJ - m[0]) ** 2 + np.exp(b * t) * (m[1] + (-2 * (m[0]) ** 2 * (1 + 2 * b * t) + 2 * eJ * (m[0] + 2 * b * m[0] * t)))) + (-a ** 2 * b ** 5 * (np.exp(4 * b * t) * (12 * (m[1]) ** 2 + (6 * m[3] + (-30 * b * (m[1]) ** 2 * t + (21 * b * m[3] * t + (12 * eJ ** 2 * m[1] * (-2 + b * t) + (72 * (m[0]) ** 2 * m[1] * (-2 + b * t) + (-8 * m[0] * m[2] * (-2 + 5 * b * t) + 8 * eJ * (2 * b * m[2] * t - 9 * m[0] * m[1] * (-2 + b * t))))))))) + 2 * np.exp((a + 3 * b) * t) * (-6 * (m[1]) ** 2 + (-3 * m[3] + (6 * b * (m[1]) ** 2 * t + (8 * m[0] * m[2] * (-1 + b * t) + (6 * eJ ** 2 * m[1] * (2 + b * t) + (36 * (m[0]) ** 2 * m[1] * (2 + b * t) - 6 * eJ * (b * m[2] * t + 6 * m[0] * m[1] * (2 + b * t))))))))) + (-4 * a ** 8 * eJ * t * (-6 * np.exp((3 * a + b) * t) * (eJ - m[0]) ** 3 + (-6 * np.exp(2 * (a + b) * t) * (eJ - m[0]) * (m[1] + (-2 * (m[0]) ** 2 * (1 + 2 * b * t) + 2 * eJ * (m[0] + 2 * b * m[0] * t))) + np.exp((a + 3 * b) * t) * (m[2] + (-9 * b * eJ ** 2 * m[0] * t * (4 + b * t) + (3 * b * eJ ** 3 * t * (6 + b * t) + (-6 * m[0] * (m[1] + 2 * b * m[1] * t) + (6 * (m[0]) ** 3 * (1 + (2 * b * t + 2 * b ** 2 * t ** 2)) + 3 * eJ * (m[1] + (b * m[1] * t - 2 * (m[0]) ** 2 * (1 + (-b * t + b ** 2 * t ** 2))))))))))) + (a ** 3 * b ** 4 * (-np.exp(2 * (a + b) * t) * (12 * (m[0]) ** 4 + (36 * (m[0]) ** 2 * m[1] + (3 * (m[1]) ** 2 + (12 * eJ ** 2 * ((m[0]) ** 2 + m[1]) + (4 * m[0] * m[2] - 4 * eJ * (6 * (m[0]) ** 3 + (12 * m[0] * m[1] + m[2]))))))) + (np.exp(4 * b * t) * ((m[0]) ** 4 * (348 - 120 * b * t) + (24 * eJ ** 3 * m[0] * (-3 + b * t) + (108 * (m[0]) ** 2 * m[1] * (-1 + 2 * b * t) + (-15 * (m[1]) ** 2 * (1 + 4 * b * t) + (5 * m[3] * (3 + 7 * b * t) + (-20 * m[0] * (m[2] + 4 * b * m[2] * t) + (4 * eJ * (5 * m[2] * (2 + b * t) + (-12 * m[0] * m[1] * (2 + 3 * b * t) + 6 * (m[0]) ** 3 * (-29 + 10 * b * t))) - 12 * eJ ** 2 * (m[1] * (-10 + b * t) + (m[0]) ** 2 * (-35 + 12 * b * t))))))))) + np.exp((a + 3 * b) * t) * (-15 * m[3] + (6 * (m[1]) ** 2 * (3 + 10 * b * t) + (8 * m[0] * m[2] * (3 + 10 * b * t) + (-72 * (m[0]) ** 2 * m[1] * (-2 + b ** 2 * t ** 2) + (12 * eJ ** 3 * m[0] * (6 + (4 * b * t + b ** 2 * t ** 2)) + (-48 * (m[0]) ** 4 * (7 + (5 * b * t + b ** 2 * t ** 2)) + (4 * eJ * (-m[2] * (11 + 14 * b * t) + (24 * (m[0]) ** 3 * (7 + (5 * b * t + b ** 2 * t ** 2)) + 12 * m[0] * m[1] * (1 + (3 * b * t + 2 * b ** 2 * t ** 2)))) - 6 * eJ ** 2 * (m[1] * (18 + (16 * b * t + 5 * b ** 2 * t ** 2)) + 2 * (m[0]) ** 2 * (34 + (24 * b * t + 5 * b ** 2 * t ** 2)))))))))))) + (a ** 6 * b * (-24 * np.exp((3 * a + b) * t) * (eJ - m[0]) ** 2 * (-m[1] + (2 * (m[0]) ** 2 * (1 + b * t) + ((eJ) ** 2 * (2 + b * t) - eJ * m[0] * (4 + 3 * b * t)))) + (np.exp(4 * b * t) * (48 * (m[0]) ** 4 + (-96 * (m[0]) ** 2 * m[1] + (24 * (m[1]) ** 2 + (-6 * m[3] + (6 * b * (m[1]) ** 2 * t + (-7 * b * m[3] * t + (48 * eJ ** 4 * (3 + b * t) + (8 * m[0] * m[2] * (4 + b * t) + (-192 * eJ ** 3 * (m[0] + b * m[0] * t) + (-8 * eJ * (9 * b * m[0] * m[1] * t + m[2] * (2 - 2 * b * t)) + 12 * eJ ** 2 * (12 * b * (m[0]) ** 2 * t + m[1] * (6 + b * t)))))))))))) + (-4 * np.exp(2 * (a + b) * t) * (-3 * (m[1]) ** 2 + (-4 * m[0] * m[2] + (12 * eJ ** 4 * (1 + b * t) + (24 * b * eJ ** 3 * m[0] * t * (2 + b * t) + (18 * (m[0]) ** 2 * m[1] * (2 + 3 * b * t) + (-12 * (m[0]) ** 4 * (3 + (7 * b * t + 4 * b ** 2 * t ** 2)) + (-6 * eJ ** 2 * (-m[1] * (4 + 3 * b * t) + 4 * (m[0]) ** 2 * (3 + (9 * b * t + 4 * b ** 2 * t ** 2))) + 4 * eJ * (m[2] + (-3 * m[0] * m[1] * (5 + 6 * b * t) + 6 * (m[0]) ** 3 * (4 + (10 * b * t + 5 * b ** 2 * t ** 2))))))))))) + 2 * np.exp((a + 3 * b) * t) * (3 * m[3] + (-6 * (m[1]) ** 2 * (3 + 5 * b * t) + (-8 * m[0] * m[2] * (3 + 5 * b * t) + (36 * (m[0]) ** 2 * m[1] * (3 + (6 * b * t + 4 * b ** 2 * t ** 2)) + (-6 * b * eJ ** 3 * m[0] * t * (74 + (44 * b * t + 5 * b ** 2 * t ** 2)) + (6 * eJ ** 4 * (-4 + (14 * b * t + (8 * b ** 2 * t ** 2 + b ** 3 * t ** 3))) + (-24 * (m[0]) ** 4 * (3 + (6 * b * t + (6 * b ** 2 * t ** 2 + 2 * b ** 3 * t ** 3))) + (6 * b * eJ ** 2 * t * (-m[1] * (13 + 6 * b * t) + (m[0]) ** 2 * (78 + (40 * b * t + 4 * b ** 2 * t ** 2))) + 2 * eJ * (m[2] * (8 + 5 * b * t) + (-6 * m[0] * m[1] * (8 + (9 * b * t + 4 * b ** 2 * t ** 2)) + 6 * (m[0]) ** 3 * (8 + (3 * b * t + (10 * b ** 2 * t ** 2 + 4 * b ** 3 * t ** 3)))))))))))))))) + (a ** 4 * b ** 3 * (4 * np.exp(2 * (a + b) * t) * (3 * (m[1]) ** 2 + (4 * m[0] * m[2] + (-18 * b * (m[0]) ** 2 * m[1] * t + (12 * eJ ** 3 * m[0] * (2 + b * t) + (-12 * (m[0]) ** 4 * (3 + 2 * b * t) + (-12 * eJ ** 2 * (m[1] + (b * m[1] * t + (m[0]) ** 2 * (7 + 4 * b * t))) + 2 * eJ * (-2 * m[2] + (3 * m[0] * m[1] * (2 + 5 * b * t) + 6 * (m[0]) ** 3 * (8 + 5 * b * t))))))))) + (np.exp(4 * b * t) * (-16 * eJ * (96 * (m[0]) ** 3 + (-39 * m[0] * m[1] + 5 * m[2])) + (-24 * eJ ** 4 * (-4 + b * t) + (60 * (m[1]) ** 2 * (1 + b * t) + (96 * eJ ** 3 * m[0] * (-9 + 2 * b * t) + (-72 * (m[0]) ** 2 * m[1] * (4 + 3 * b * t) + (24 * (m[0]) ** 4 * (14 + 5 * b * t) + (-5 * m[3] * (4 + 7 * b * t) + (80 * m[0] * (m[2] + b * m[2] * t) - 48 * eJ ** 2 * (m[1] * (5 - 2 * b * t) + (m[0]) ** 2 * (-41 + 6 * b * t)))))))))) - 4 * np.exp((a + 3 * b) * t) * (18 * (m[1]) ** 2 + (-5 * m[3] + (30 * b * (m[1]) ** 2 * t + (8 * m[0] * m[2] * (3 + 5 * b * t) + (-18 * (m[0]) ** 2 * m[1] * (4 + (7 * b * t + 4 * b ** 2 * t ** 2)) + ((eJ) ** 4 * (24 + (18 * b * t + (6 * b ** 2 * t ** 2 + b ** 3 * t ** 3))) + (4 * (m[0]) ** 4 * (12 + (9 * b * t + (6 * b ** 2 * t ** 2 + 2 * b ** 3 * t ** 3))) + (-eJ ** 3 * m[0] * (192 + (162 * b * t + (48 * b ** 2 * t ** 2 + 7 * b ** 3 * t ** 3))) + (3 * eJ ** 2 * (-m[1] * (24 + (23 * b * t + 7 * b ** 2 * t ** 2)) + 2 * (m[0]) ** 2 * (68 + (58 * b * t + (19 * b ** 2 * t ** 2 + 3 * b ** 3 * t ** 3)))) - eJ * (m[2] * (24 + 25 * b * t) + (-42 * m[0] * m[1] * (4 + (5 * b * t + 2 * b ** 2 * t ** 2)) + 4 * (m[0]) ** 3 * (72 + (60 * b * t + (24 * b ** 2 * t ** 2 + 5 * b ** 3 * t ** 3)))))))))))))))) + (a ** 7 * (-6 * np.exp(4 * a * t) * (eJ - m[0]) ** 4 + (np.exp(4 * b * t) * (-6 * (m[0]) ** 4 + (12 * (m[0]) ** 2 * m[1] + (-3 * (m[1]) ** 2 + (-4 * m[0] * m[2] + (m[3] + (-24 * b * eJ ** 3 * m[0] * t + (12 * b * eJ ** 2 * m[1] * t + (-4 * b * eJ * m[2] * t + (b * m[3] * t + 6 * eJ ** 4 * (1 + 4 * b * t)))))))))) + (-12 * np.exp((3 * a + b) * t) * (eJ - m[0]) ** 2 * (m[1] + (-2 * (m[0]) ** 2 * (1 + 2 * b * t) + 2 * eJ * (m[0] + 2 * b * m[0] * t))) + (np.exp(2 * (a + b) * t) * (-3 * (m[1]) ** 2 + (-4 * m[0] * m[2] + (48 * b * eJ ** 4 * t * (3 + b * t) + (-48 * b * eJ ** 3 * m[0] * t * (9 + 4 * b * t) + (36 * (m[0]) ** 2 * (m[1] + 2 * b * m[1] * t) + (-12 * (m[0]) ** 4 * (3 + (8 * b * t + 8 * b ** 2 * t ** 2)) + (4 * eJ * (m[2] + (-12 * m[0] * (m[1] + b * m[1] * t) + 6 * (m[0]) ** 3 * (3 + (2 * b * t + 4 * b ** 2 * t ** 2)))) + 12 * eJ ** 2 * (m[1] + (-2 * b * m[1] * t + (m[0]) ** 2 * (-3 + (28 * b * t + 12 * b ** 2 * t ** 2))))))))))) + np.exp((a + 3 * b) * t) * (6 * (m[1]) ** 2 + (-m[3] + (12 * b * (m[1]) ** 2 * t + (48 * b * eJ ** 4 * t * (2 + b * t) + (8 * m[0] * (m[2] + 2 * b * m[2] * t) + (-36 * (m[0]) ** 2 * m[1] * (1 + (2 * b * t + 2 * b ** 2 * t ** 2)) + (12 * b * eJ ** 3 * m[0] * t * (-4 + (15 * b * t + 4 * b ** 2 * t ** 2)) + (8 * (m[0]) ** 4 * (3 + (6 * b * t + (6 * b ** 2 * t ** 2 + 4 * b ** 3 * t ** 3))) + (-6 * b * eJ ** 2 * t * (-m[1] * (20 + 13 * b * t) + (m[0]) ** 2 * (32 + (86 * b * t + 24 * b ** 2 * t ** 2))) + 4 * eJ * (m[2] * (-1 + 2 * b * t) + (-6 * m[0] * m[1] * (-1 + (4 * b * t + 4 * b ** 2 * t ** 2)) + 2 * (m[0]) ** 3 * (-3 + (12 * b * t + (30 * b ** 2 * t ** 2 + 8 * b ** 3 * t ** 3)))))))))))))))))) - a ** 5 * b ** 2 * (-12 * np.exp((3 * a + b) * t) * (eJ - m[0]) ** 2 * (2 * eJ * m[0] + (-2 * (m[0]) ** 2 - m[1])) + (np.exp(4 * b * t) * (1104 * eJ ** 3 * m[0] + (168 * (m[0]) ** 4 + (30 * (m[1]) ** 2 * (2 + b * t) + (40 * m[0] * m[2] * (2 + b * t) + (24 * eJ ** 4 * (-13 + 2 * b * t) + (-24 * (m[0]) ** 2 * m[1] * (14 + 3 * b * t) + (-3 * m[3] * (5 + 7 * b * t) + (4 * eJ * (60 * b * (m[0]) ** 3 * t + (12 * m[0] * m[1] * (8 - 3 * b * t) + 5 * m[2] * (-3 + b * t))) - 24 * eJ ** 2 * (m[1] + (-4 * b * m[1] * t + 4 * (m[0]) ** 2 * (10 + 3 * b * t))))))))))) + (6 * np.exp(2 * (a + b) * t) * (3 * (m[1]) ** 2 + (4 * m[0] * m[2] + (-12 * (m[0]) ** 2 * m[1] * (2 + 3 * b * t) + (4 * eJ ** 4 * (5 + (4 * b * t + b ** 2 * t ** 2)) + (-8 * eJ ** 3 * m[0] * (12 + (11 * b * t + 3 * b ** 2 * t ** 2)) + (4 * (m[0]) ** 4 * (5 + (6 * b * t + 4 * b ** 2 * t ** 2)) + (-4 * eJ * (m[2] + (-2 * m[0] * m[1] * (6 + 7 * b * t) + 2 * (m[0]) ** 3 * (12 + (13 * b * t + 6 * b ** 2 * t ** 2)))) + 4 * eJ ** 2 * (-m[1] * (6 + 5 * b * t) + (m[0]) ** 2 * (38 + (38 * b * t + 13 * b ** 2 * t ** 2)))))))))) + np.exp((a + 3 * b) * t) * (15 * m[3] + (48 * eJ ** 4 * ((2 + b * t)) ** 2 + (-6 * (m[1]) ** 2 * (13 + 20 * b * t) + (-8 * m[0] * m[2] * (13 + 20 * b * t) + (36 * (m[0]) ** 2 * m[1] * (13 + (22 * b * t + 12 * b ** 2 * t ** 2)) + (12 * eJ ** 3 * m[0] * (-42 + (-20 * b * t + (-b ** 2 * t ** 2 + 2 * b ** 3 * t ** 3))) + (-24 * (m[0]) ** 4 * (13 + (26 * b * t + (16 * b ** 2 * t ** 2 + 4 * b ** 3 * t ** 3))) + (4 * eJ * (m[2] * (21 + 20 * b * t) + (-6 * m[0] * m[1] * (27 + (38 * b * t + 16 * b ** 2 * t ** 2)) + 6 * (m[0]) ** 3 * (27 + (66 * b * t + (38 * b ** 2 * t ** 2 + 8 * b ** 3 * t ** 3))))) - 6 * eJ ** 2 * (-m[1] * (26 + (20 * b * t + 7 * b ** 2 * t ** 2)) + 2 * (m[0]) ** 2 * (2 + (76 * b * t + (47 * b ** 2 * t ** 2 + 10 * b ** 3 * t ** 3)))))))))))))))))))))))))