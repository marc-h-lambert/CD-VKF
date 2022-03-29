import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step, rk4stepSDP, checkCov
from abc import ABCMeta, abstractmethod
# Kalman API
from .Kalman import KalmanFilter, DynamicalSystem, KalmanPredictor, KalmanObservator 
import scipy

# Kalman Test
#from .Kalman import LinearKalmanFilter, UniformMotion1D, \
#    UniformMotion1D_Predictor, scalar_Observator, KalmanPlot

getcontext().prec = 6

class NonLinearPredictor(KalmanPredictor):

    @abstractmethod
    def jacobianDynamic(self,state):
        return 
    
class NonLinearObservator(KalmanObservator):

    @abstractmethod
    def jacobianObservation(self,state):
        return 
    
    @abstractmethod
    def hessianObservation(self,state):
        return 

class CD_EKF(KalmanFilter):
    
    def __init__(self,mean0,cov0,sqrt=False):
        super().__init__(mean0,cov0)
        # SQRT version not implemented
            
    def propagate(self,predictor,dt):
        #print("at time {} mean shape= {}".format(self.time,self.mean.shape))
        F=predictor.jacobianDynamic(self.mean)
        D=predictor.diffusionMatrix(self.mean,self.time)
        Q=predictor.covarianceMatrix()
        DQD=D.dot(Q).dot(D.T) 
        
        # Runge Kutta integration
        self.mean=rk4step(predictor.dynamic,dt,self.mean)
        #self.mean=self.mean+predictor.dynamic(self.mean)*dt
        self.cov=rk4step(predictor.dynamicCov,dt,self.cov,F,DQD)
        
        # If URM, UAM or UJM the matrix F is nilpotent and the following 
        # integration scheme is exact !! (better than Runge Kutta)
        
        #self.mean=self.mean+F.dot(self.mean)*dt
        #P=self.cov
        #self.cov=P+dt*F.dot(P) +dt*P.dot(F.T)+F.dot(P).dot(F.T)*dt**2
        return self.mean, self.cov
    
    # Bayesian inference using observation yt
    def update(self,observator,yt):
        H=observator.jacobianObservation(self.mean)
        R=observator.covarianceMatrix(self.mean)
        P=self.cov
        yhat=observator.predict(self.mean)
        # print("t=",self.time)
        # print("yhat=",yhat)
        # print("yt=",yt)
        # print("err=",yt-yhat)
        err=(yt-yhat).reshape(-1,1)
        S=H.dot(P).dot(H.T)+R
        K=P.dot(H.T).dot(LA.inv(S))
        self.mean=self.mean+K.dot(err)
        self.cov=P-K.dot(H).dot(P)
        return self.mean, self.cov

class CD_UKF(KalmanFilter):
    
    def __init__(self,mean0,cov0,sqrt=True):
        super().__init__(mean0,cov0)
        self.sqrt=sqrt
        if self.sqrt:
            self.R=LA.cholesky(cov0)
        
    def lower(A):
        B=np.tril(A)
        B=B-0.5*np.diag(np.diag(A))
        return B

    def fmeanUKF(f,mu,P,*args):
        n=mu.shape[0]
        P=checkCov(P,"fmean")
        #sqrtP=scipy.linalg.cholesky(P, lower=True)
        sqrtP=LA.cholesky(P)
        mu=mu.reshape(-1,1)
                
        k=3-n
        W0=2*k/(2*(n+k))
        fmean=W0*f(mu,*args)
        
        for i in range(0,n):
            vi=sqrtP[:,i].reshape(n,1)
            sigmaPointi_Plus=mu+vi*math.sqrt(n+k)
            sigmaPointi_Moins=mu-vi*math.sqrt(n+k)
            Wi=1/(2*(n+k))
            fmean=fmean+Wi*f(sigmaPointi_Plus,*args)+Wi*f(sigmaPointi_Moins,*args)
        return fmean
    
    def fmeanUKF_SQRT(f,mu,sqrtP,*args):
        n=mu.shape[0]
        mu=mu.reshape(-1,1)
                
        k=3-n
        W0=2*k/(2*(n+k))
        fmean=W0*f(mu,*args)
        
        for i in range(0,n):
            vi=sqrtP[:,i].reshape(n,1)
            sigmaPointi_Plus=mu+vi*math.sqrt(n+k)
            sigmaPointi_Moins=mu-vi*math.sqrt(n+k)
            Wi=1/(2*(n+k))
            fmean=fmean+Wi*f(sigmaPointi_Plus,*args)+Wi*f(sigmaPointi_Moins,*args)
        return fmean
        
    def Eyy(x,h,yhat):
        e=(h(x)-yhat).reshape(-1,1)
        return e.dot(e.T)

    def Exy(x,xmean,h,yhat):
        e=(h(x)-yhat).reshape(-1,1)
        return (x-xmean).dot(e.T)
    
    def Exf(x,xmean,f):
        return (x-xmean).dot(f(x).T)
    
    def GaussianDynamic(X,d,f,Jf,DQD,default_meanJF=None):
        mu=X[0:d].reshape(d,1)
        P=X[d:].reshape(d,d)
        #print(P)
        dmu=CD_UKF.fmeanUKF(f,mu,P)
        dmu=dmu.reshape(d,1)
        if default_meanJF is None:
            meanJf=CD_UKF.fmeanUKF(Jf,mu,P)
        else: #we do not update meanJF at each RK-step (may avoid negative P)
            meanJf=default_meanJF
        dP=meanJf.dot(P)+P.dot(meanJf.T)+DQD
        dP=dP.reshape(-1,1)
        dX=np.concatenate((dmu, dP), axis=0)
        return dX
    
    def GaussianDynamicSQRT(X,d,f,Jf,DQD,default_meanJF=None):
        mu=X[0:d].reshape(d,1)
        R=X[d:].reshape(d,d)
        # compute MEAN update
        dmu=CD_UKF.fmeanUKF_SQRT(f,mu,R)
        dmu=dmu.reshape(d,1)
        
        # compute SQRT update
        A=CD_UKF.fmeanUKF_SQRT(CD_UKF.Exf,mu,R,mu,f)
        XR=A+A.T+DQD
        invR=LA.inv(R)
        L=CD_UKF.lower(invR.dot(XR).dot(invR.T))
        dR=R.dot(L)
        dR=dR.reshape(-1,1)
        dX=np.concatenate((dmu, dR), axis=0)
        return dX
        
    def meanDynamic(mean,cov,f):
        return CD_UKF.fmeanUKF(f,mean,cov)
    
    def propagate(self,predictor,dt):
        #print("propag ",self.time)
        d=self.mean.shape[0]
        D=predictor.diffusionMatrix(self.mean,self.time)
        Q=predictor.covarianceMatrix()
        DQD=D.dot(Q).dot(D.T) 
        
        # Runge Kutta integration 
        if self.sqrt:
            X0=np.concatenate((self.mean.reshape(-1,1), self.R.reshape(-1,1)), axis=0)
            Xt=rk4step(CD_UKF.GaussianDynamicSQRT,dt,X0,d,predictor.dynamic,
                       predictor.jacobianDynamic,DQD)
            self.mean=Xt[0:d]
            self.R=Xt[d:].reshape(d,d)
            self.cov=self.R.dot(self.R.T)
        else:
            X0=np.concatenate((self.mean.reshape(-1,1), self.cov.reshape(-1,1)), axis=0)
            Xt=rk4step(CD_UKF.GaussianDynamic,dt,X0,d,predictor.dynamic,
                       predictor.jacobianDynamic,DQD)
            self.mean=Xt[0:d]
            self.cov=Xt[d:].reshape(d,d)
        return self.mean, self.cov
    
    # Bayesian inference using observation yt
    def update(self,observator,yt):
        #print("update !! ",self.time)
        Rs=observator.covarianceMatrix(self.mean)
        P=self.cov
        yhat=CD_UKF.fmeanUKF(observator.predict,self.mean,self.cov)
        err=(yt-yhat).reshape(-1,1)
        cov_yy=CD_UKF.fmeanUKF(CD_UKF.Eyy,self.mean,self.cov,observator.predict,yhat)
        cov_xy=CD_UKF.fmeanUKF(CD_UKF.Exy,self.mean,self.cov,self.mean,observator.predict,yhat)
        
        S=cov_yy+Rs
        K=cov_xy.dot(LA.inv(S))
        self.mean=self.mean+K.dot(err)
        self.cov=self.cov-K.dot(S).dot(K.T)
        self.cov=checkCov(self.cov,"Update")
        if self.sqrt:
            self.R=LA.cholesky(self.cov)
        return self.mean, self.cov
    
class CD_VKF(KalmanFilter):
    def __init__(self,mean0,cov0,sqrt=True):
        super().__init__(mean0,cov0)
        self.sqrt=sqrt
        if self.sqrt:
            self.R=LA.cholesky(cov0)
            
    def gradLogP(X,yt,polarObs):
        J=polarObs.jacobianObservation(X)
        invR=LA.inv(polarObs.covarianceMatrix(X))
        return J.T.dot(invR).dot(yt-polarObs.predict(X))
    
    def covXJmeanUKF(f,mu,sqrtP,*args):
        n=mu.shape[0]
        mu=mu.reshape(-1,1)
                
        k=3-n
        W0=2*k/(2*(n+k))
        fmean=0
        
        for i in range(0,n):
            vi=sqrtP[:,i].reshape(n,1)
            ci=math.sqrt(n+k)
            sigmaPointi_Plus=mu+vi*ci
            sigmaPointi_Moins=mu-vi*ci
            
            #compute sigma points on x
            eiPlus=np.identity(n)[:,i].reshape(n,1)
            eiMoins=-eiPlus
            Wi=1/(2*(n+k))
            fmean=fmean+Wi*ci*eiPlus.dot(f(sigmaPointi_Plus,*args).T)
            fmean=fmean+Wi*ci*eiMoins.dot(f(sigmaPointi_Moins,*args).T)
        return fmean
    
    def covXgradLogP(X,yt,polarObs,mean):
        v=CD_VKF.gradLogP(X,yt,polarObs)
        return (X-mean).dot(v.T)
    
    def propagate(self,predictor,dt):
        #print("propag ",self.time)
        d=self.mean.shape[0]
        D=predictor.diffusionMatrix(self.mean,self.time)
        Q=predictor.covarianceMatrix()
        DQD=D.dot(Q).dot(D.T) 
        
        # Runge Kutta integration 
        if self.sqrt:
            X0=np.concatenate((self.mean.reshape(-1,1), self.R.reshape(-1,1)), axis=0)
            Xt=rk4step(CD_UKF.GaussianDynamicSQRT,dt,X0,d,predictor.dynamic,
                       predictor.jacobianDynamic,DQD)
            self.mean=Xt[0:d]
            self.R=Xt[d:].reshape(d,d)
            self.cov=self.R.dot(self.R.T)
        else:
            X0=np.concatenate((self.mean.reshape(-1,1), self.cov.reshape(-1,1)), axis=0)
            Xt=rk4step(CD_UKF.GaussianDynamic,dt,X0,d,predictor.dynamic,
                       predictor.jacobianDynamic,DQD)
            self.mean=Xt[0:d]
            self.cov=Xt[d:].reshape(d,d)
        return self.mean, self.cov
        
    # Bayesian inference using observation yt
    def update(self,observator,yt):
        #print("update ",self.time)
        if self.sqrt:
            m0=self.mean
            R0=self.R
            I=np.identity(m0.shape[0])
            ExpectGrad=CD_UKF.fmeanUKF_SQRT(CD_VKF.gradLogP,m0,R0,yt,observator)
            self.mean=m0+R0.dot(R0.T).dot(ExpectGrad)
            ExpectCov=CD_VKF.covXJmeanUKF(CD_VKF.gradLogP,m0,R0,yt,observator)
            #print("ExpectCov=",ExpectCov)
            A=I+0.5*ExpectCov.dot(R0)+0.5*R0.T.dot(ExpectCov.T)
            A=checkCov(A,"Update")
            L=LA.cholesky(A)
            self.R=R0.dot(L)
            self.cov=self.R.dot(self.R.T)
        else:
            m0=self.mean
            P0=self.cov
            ExpectGrad=CD_UKF.fmeanUKF(CD_VKF.gradLogP,m0,P0,yt,observator)
            self.mean=m0+P0.dot(ExpectGrad)
            ExpectCov=CD_UKF.fmeanUKF(CD_VKF.covXgradLogP,m0,P0,yt,observator,m0)
            self.cov=P0+0.5*ExpectCov.dot(P0)+0.5*P0.dot(ExpectCov.T)
            self.cov=checkCov(self.cov,"Update")
        return self.mean, self.cov

class TrueEKF:
    def __init__(self,cov0):
        self.cov=cov0
        self.time=0
        self.Nobs=0
        self.traj_cov=[]
        self.traj_cov.append(self.cov)
    
    def propagate(self,predictor,dt,trueState):
        F=predictor.jacobianDynamic(trueState)
        D=predictor.diffusionMatrix(trueState,self.time)
        Q=predictor.covarianceMatrix()
        DQD=D.dot(Q).dot(D.T) 
        self.cov=rk4step(predictor.dynamicCov,dt,self.cov,F,DQD)
        return self.cov
    
    def update(self,observator,trueState):
        H=observator.jacobianObservation(trueState)
        R=observator.covarianceMatrix(trueState)
        S=H.dot(self.cov).dot(H.T)+R
        K=self.cov.dot(H.T).dot(LA.inv(S))
        self.cov=self.cov-K.dot(H).dot(self.cov)
        return self.cov

    def filtering(self,predictor,observator,system,dt,T,dobs,Ttrack=100000):
        self.system=system
        while self.time < T:    
            self.time=self.time+Decimal(dt)
            system.propagate(dt,dt,noise=True) # true state
            
            # Kalman propagation step
            self.propagate(predictor,dt,system.state)
            
            # Kalman observation step
            if self.time % dobs == 0 and self.time < Ttrack:
                if not isinstance(observator, list):  
                    yt=observator.observe(system.state)
                    if not yt is None:
                        self.Nobs=self.Nobs+1
                        self.update(observator,system.state) 
                else:# case of multi observator
                    for obs in observator:
                        yt=obs.observe(system.state)
                        if not yt is None:
                            self.Nobs=self.Nobs+1
                            self.update(obs,system.state) 
                                        
            self.traj_cov.append(self.cov)
        print('number of observations={}'.format(self.Nobs))
        return np.asarray(system.trajectory()), np.asarray(self.traj_cov)
    


# if __name__ == "__main__":
    
#     TEST=["LKF","EKF","UKF","VKF"]  
#     num=0
#     T=50
#     dobs=5
#     seed=10
    
#     # d=2/dt=1: uniform rectilinear motion (URM); 
#     # d=3/dt=0.1: uniform accelerated motion (UAM);
#     # d=4/dt=0.01: uniform jerked motion (UJM); 
#     d=4
#     dt=0.01 
#     print("state dimension={}; Rune Kutta step={}".format(d,dt))
    
#     if "LKF" in TEST:   
#         print("Running Linear Kalman filter...")
#         mean, cov, traj= TestUniformMotionNL(LinearKalmanFilter,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num, name="Linear Kalman")
#         num=num+2
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("Linear Kalman filtering \n" + r" Uniform motion $A^{}=0$".format(d))
#         plt.tight_layout()
#         plt.savefig("output/UniformMotion_LKF.pdf")
        
#     if "EKF" in TEST:
#         print("Running Extended Kalman filter...")
#         mean, cov, traj= TestUniformMotionNL(CD_EKF,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num,name="Extended Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("EKF filtering \n" + r" Uniform motion $A^{}=0$".format(d))
#         plt.tight_layout()
#         #plt.savefig("output/UniformMotion_EKF.pdf")
        
#     if "UKF" in TEST:
#         print("Running Unscented Kalman filter...")
#         mean, cov, traj= TestUniformMotionNL(CD_UKF,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num,name="Unscented Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("UKF filtering \n" + r" Uniform motion $A^{}=0$".format(d))
#         plt.tight_layout()
#         #plt.savefig("output/UniformMotion_UKF.pdf")
        
#     if "VKF" in TEST:
#         print("Running Variational Kalman filter...")
#         mean, cov, traj= TestUniformMotionNL(CD_VKF,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num,name="Variational Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("VKF filtering \n" + r" Uniform motion $A^{}=0$".format(d))
#         plt.tight_layout()
#         #plt.savefig("output/UniformMotion_UKF.pdf")
    
