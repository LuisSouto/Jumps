# Definition of the Heston class

import numpy as np
import scipy.special as sp

class Heston():
    def __init__(self,S0,V0,r,lamb,nu,eta,rho):
        self.S0   = S0
        self.X0   = np.log(S0)
        self.V0   = V0
        self.r    = r
        self.lamb = lamb
        self.nu   = nu
        self.eta  = eta
        self.rho  = rho

    def simul(self,N,n,T):
        
        # Simulate the diffusion
        dt = T/(n-1)

        X = np.zeros((N,n))
        V = np.zeros((N,n))
        X[:,0] = self.X0
        V[:,0] = self.V0

        lamb = self.lamb
        nu   = self.nu
        eta  = self.eta
        rho  = self.rho
        r    = self.r

        dWS = np.random.normal(0,np.sqrt(dt),(N,n-1))
        dWV = rho*dWS+np.sqrt(1-rho**2)*np.random.normal(0,np.sqrt(dt),(N,n-1))
        for i in range(1,n):
            Vp = np.maximum(V[:,i-1],0)
            V[:,i] = V[:,i-1] + lamb*(nu-Vp)*dt + np.sqrt(Vp)*eta*dWV[:,i-1]
            X[:,i] = X[:,i-1] + (r-0.5*Vp)*dt + np.sqrt(Vp)*dWS[:,i-1]
            
        return X,V

    # Characteristic function
    def cf(self,u,v,t,X):
        lamb = self.lamb
        nu   = self.nu
        eta  = self.eta
        rho  = self.rho

        b  = lamb-1j*u*eta*rho
        d  = np.sqrt(b**2+eta**2*u*(1j+u))
        g1 = (b-d)/(b+d)
        g2 = (b-d-1j*eta**2*v)/(b+d-1j*eta**2*v)
        f1 = (b+d)/eta**2*((g1-g2*np.exp(-d*t))/(1-g2*np.exp(-d*t)))
        f2 = (1j*u*self.r*t
              +lamb*nu/eta**2*((b-d)*t-2*np.log((1-g2*np.exp(-d*t))/(1-g2))))
        
        return np.exp(1j*u*X[0] + f1*X[1] + f2)

    def mean(self,t):
        return (np.log(self.S0)
                +self.r*t
                +(1-np.exp(-self.lamb*t))*(self.nu-self.V0)/(2*self.lamb)
                -self.nu*t/2)

    def var(self,t):
        lamb = self.lamb
        eta  = self.eta
        rho  = self.rho
        nu   = self.nu
        V0   = self.V0
        et   = np.exp(-lamb*t)

        return (eta*t*lamb*et*(V0-nu)*(8*lamb*rho-4*eta)
                +lamb*eta*rho*(1-et)*(16*nu-8*V0)
                +2*nu*lamb*t*(-4*lamb*rho*eta+eta**2+4*lamb**2)
                +eta**2*((nu-2*V0)*et**2+nu*(6*et-7)+2*V0)
                +8*lamb**2*(V0-nu)*(1-et))/(8*lamb**3)

    def dens_logvar(self,t,vt,vs):
        lamb = self.lamb
        eta  = self.eta
        nu   = self.nu

        et = np.exp(-lamb*t)
        q  = 2*lamb*nu/eta**2-1
        c  = 2*lamb/((1-et)*eta**2)

        return (c*np.exp(-c*(vs*et+vt))*(vt/(vs*et))**(q/2)*vt
                *sp.iv(q,2*c*np.sqrt(vt*vs*et)))

    # First derivative
    def dens_logvar_dv(self,t,vt,vs):
        lamb = self.lamb
        eta  = self.eta
        nu   = self.nu
        et   = np.exp(-lamb*t)
        q    = 2*lamb*nu/eta**2-1
        c    = 2*lamb/((1-et)*eta**2)

        return (((-c*vt+q+1)*sp.iv(q,2*c*np.sqrt(vt*vs*et))
                 +c*np.sqrt(vt*vs*et)
                  *(sp.iv(q+1,2*c*np.sqrt(vt*vs*et))))
                *c*np.exp(-c*(vt+vs*et))*vt*(vt/(vs*et))**(q/2))

    # Equation (30) of Fang (2011)
    def cfcondv_dens(self,u,t,logS,vt,vs):
        lamb = self.lamb
        eta  = self.eta
        nu   = self.nu

        q   = 2*lamb*nu/eta**2-1
        w   = u*(lamb*self.rho/eta-0.5)+1j*u**2/2*(1-self.rho**2)
        g   = np.sqrt(lamb**2-1j*2*w*eta**2)
        et  = np.exp(-lamb*t)
        eg  = np.exp(-g*t)
        arg = np.sqrt(vt*vs*eg)*4*g/(eta**2*(1-eg))

        bfk1 = (np.exp((vs+vt)/(eta**2)*(-g*(1+eg)/(1-eg)))*sp.iv(q,arg))

        return (np.exp(1j*u*(logS+self.r*t+self.rho/eta*(vt-vs-lamb*nu*t)))
                *bfk1*g*(eg/et)**(0.5)*2/((eta**2)*(1-eg))
                *np.exp(lamb/eta**2*(vs-vt))*(vt/(vs*et))**(q/2)*vt)

