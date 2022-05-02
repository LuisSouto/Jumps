# Basis for the point process class. It contains the basic methods that the three classes: Poisson, Hawkes and ESEP should have.

class PointP():
    def __init__(self,a,b,hb,Q0,mJ,eJ,cfJ):
        self.set_param(a,b,hb,Q0,mJ,eJ,cfJ)

    def set_param(self,a=None,b=None,hb=None,Q0=None,mJ=None,eJ=None,cfJ=None):
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

    def cf_cj(self,u,v,t):
        pass