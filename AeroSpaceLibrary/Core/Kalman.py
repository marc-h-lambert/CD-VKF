import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Integration import rk4step
from abc import ABCMeta, abstractmethod

getcontext().prec = 6

class DynamicalSystem:
    
    def __init__(self,state0):
        self.state0=state0
        self.state=state0
        self.traj=[]
        self.traj.append(np.concatenate(([Decimal("0")], self.state.reshape(-1,)), axis=0))
        self.time=0
    
    @abstractmethod
    def propagate(self,dt,T,noise=True):
        return 
    
    def trajectory(self):
        return np.array(self.traj)
    
class KalmanPredictor:

    @abstractmethod
    def dynamic(self,state):
        return 
        
    @abstractmethod
    def diffusionMatrix(self,state,t):
        return 

    @abstractmethod
    def covarianceMatrix(self):
        return 
    
    @abstractmethod
    def dynamicCov(self,P,F,Q):
        return F.dot(P)+P.dot(F.T)+Q
    
class KalmanLinearPredictor(KalmanPredictor):

    @abstractmethod
    def dynamicMatrix(self):
        return 
    
class KalmanObservator:

    @abstractmethod #h(x)
    def observe(self,state):
        return 
    
    @abstractmethod #h_hat(x)
    def predict(self,state):
        return

    @abstractmethod
    def covarianceMatrix(self,state):
        return 
    
class KalmanLinearObservator(KalmanObservator):

    @abstractmethod
    def observationMatrix(self):
        return 
    
class KalmanFilter:
    def __init__(self,mean0,cov0):
        self.mean=mean0
        self.cov=cov0
        self.time=0
        self.Nobs=0
        self.traj_mean=[]
        self.traj_mean.append(np.concatenate(([Decimal("0")], self.mean.reshape(-1,)), axis=0))
        self.traj_cov=[]
        self.traj_cov.append(self.cov)
    
    @abstractmethod
    # Propagation for a step
    def propagate(self,predictor,dt):
        return
    
    @abstractmethod
    # Bayesian inference using observation yt
    def update(self,observator,yt):
        return

    def filtering(self,predictor,observator,system,dt,T,dobs,Ttrack=100000):
        self.system=system
        while self.time < T:    
            #print(self.time)
            self.time=self.time+Decimal(dt)
            system.propagate(dt,dt,noise=True) # true state
            
            # Kalman propagation step
            self.propagate(predictor,dt)
            
            # Kalman observation step
            if self.time % dobs == 0 and self.time < Ttrack:
                
                if not isinstance(observator, list):  
                    yt=observator.observe(system.state)
                    if not yt is None:
                        self.Nobs=self.Nobs+1
                        self.update(observator,yt) 
                else:# case of multi observator
                    for obs in observator:
                        yt=obs.observe(system.state)
                        if not yt is None:
                            self.Nobs=self.Nobs+1
                            self.update(obs,yt) 
            
            self.traj_mean.append(np.concatenate(([self.time], self.mean.reshape(-1,)), axis=0))
            self.traj_cov.append(self.cov)
        print('number of observations={}'.format(self.Nobs))
        return np.asarray(self.traj_mean), np.asarray(self.traj_cov)
        
class LinearKalmanFilter(KalmanFilter):
    
    def propagate(self,predictor,dt):
        F=predictor.dynamicMatrix()
        D=predictor.diffusionMatrix(self.mean,self.time)
        Q=predictor.covarianceMatrix()
        DQD=D.dot(Q).dot(D.T) 
        
        # Runge Kutta integration
        #self.mean=rk4step(predictor.dynamic,dt,self.mean)
        #self.cov=rk4step(predictor.dynamicCov,dt,self.cov,F,DQD)
        
        # If URM, UAM or UJM the matrix F is nilpotent and the following 
        # integration scheme is exact !! (better than Runge Kutta)
        self.mean=self.mean+F.dot(self.mean)*dt
        P=self.cov
        self.cov=P+dt*F.dot(P) +dt*P.dot(F.T)+F.dot(P).dot(F.T)*dt**2
        return self.mean, self.cov
    
    # Bayesian inference using observation yt
    def update(self,observator,yt):
        H=observator.observationMatrix()
        R=observator.covarianceMatrix(self.mean)
        P=self.cov
        err=(yt-H.dot(self.mean))
        S=H.dot(P).dot(H.T)+R
        K=P.dot(H.T).dot(LA.inv(S))
        self.mean=self.mean+K.dot(err)
        self.cov=P-K.dot(H).dot(P)
        return self.mean, self.cov

def PlotOutput(ax,mean,cov,X0_True,P0,Q,T,dt,idx,nbRuns=3,seed=1,randX0=True):
    plotLegend=True
    for seed_run in np.linspace(seed,seed*nbRuns+1,nbRuns):
        np.random.seed(int(seed_run))
        if randX0:
            X0=np.random.multivariate_normal(X0_True.reshape(-1,),P0)
            X0=X0.reshape([5,1])
        else:
            X0=X0_True
        system=ReEntryVehicle(state0=X0,Q=Q)
        system.propagate(dt,T,noise=True)
        traj=system.trajectory()
        if plotLegend:
            ax.plot(traj[:,0],traj[:,idx],linewidth='0.7',color='k',label='random instances')
            plotLegend=False
        else:
            ax.plot(traj[:,0],traj[:,idx],linewidth='0.7',color='k')
    
    time=mean[:,0]
    #ax.plot(time,mean[:,idx],linewidth='2',linestyle='-.',color='r',label='mean')
    s=3
    #print('for idx={}, cov={}'.format(idx,cov[10:100,idx,idx]))
    std1=mean[:,idx]+s*np.sqrt(cov[:,idx-1,idx-1])
    std2=mean[:,idx]-s*np.sqrt(cov[:,idx-1,idx-1])
    ax.plot(time,std1,linewidth='1.5',linestyle='-.',color='r',label=r'std (at $3\sigma$)')
    ax.plot(time,std2,linewidth='1.5',linestyle='-.',color='r')
    ax.fill(np.append(time, time[::-1]), np.append(std1, std2[::-1]), 'lightgrey')
    ax.set_xlabel("time")
    
def KalmanPlot(mean, cov, traj,num,name,idx1=1,idx2=2,savefig=False):
    d=mean.shape[1]
    err=traj-mean
    MSE=(err)**2
    
    time=mean[:,0]
    traj=np.asarray(traj)
    s=3
        
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
    ax1.plot(time,mean[:,idx1],label='mean x')
    ax1.plot(time,traj[:,idx1],label='real x')
    std1=mean[:,idx1]+s*np.sqrt(cov[:,idx1-1,idx1-1])
    std2=mean[:,idx1]-s*np.sqrt(cov[:,idx1-1,idx1-1])
    ax1.plot(time,std1,linewidth='1.5',linestyle='-.',color='r',label=r'std (at $3\sigma$)')
    ax1.plot(time,std2,linewidth='1.5',linestyle='-.',color='r')
    ax1.fill(np.append(time, time[::-1]), np.append(std1, std2[::-1]), 'lightgrey')
    ax1.set_title("position")
    ax1.legend()
    
    ax2.plot(time,mean[:,idx2],label='mean v')
    ax2.plot(time,traj[:,idx2],label='real v')
    std1=mean[:,idx2]+s*np.sqrt(cov[:,idx2-1,idx2-1])
    std2=mean[:,idx2]-s*np.sqrt(cov[:,idx2-1,idx2-1])
    ax2.plot(time,std1,linewidth='1.5',linestyle='-.',color='r',label=r'std (at $3\sigma$)')
    ax2.plot(time,std2,linewidth='1.5',linestyle='-.',color='r')
    ax2.fill(np.append(time, time[::-1]), np.append(std1, std2[::-1]), 'lightgrey')
    ax2.set_title("velocity")
    ax2.legend()
    
    plt.suptitle("{} filter in dimension d={}".format(name,d))
    plt.tight_layout()
    if savefig:
        plt.savefig("outputs/KeplerMotionState_{}.pdf".format(name))
    
    num=num+1
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
    time=mean[:,0]
    
    ax1.semilogy(time,MSE[:,idx1],linewidth='2',linestyle='-',color='g',label='MSE(x)')
    ax1.semilogy(time,9*cov[:,idx1-1,idx1-1],linewidth='2',linestyle='-',color='r',label='9cov(x)')
    ax1.set_xlabel("time")
    ax1.set_ylabel("variance x")
    ax1.legend()

    ax2.semilogy(time,MSE[:,idx2],linewidth='2',linestyle='-',color='g',label='MSE(v)')
    ax2.semilogy(time,9*cov[:,idx2-1,idx2-1],linewidth='2',linestyle='-',color='r',label='9cov(v)')
    ax2.set_xlabel("time")
    ax2.set_ylabel("variance v")
    ax2.legend()
    if savefig:
        plt.savefig("outputs/KeplerMotionCOV_{}.pdf".format(name))
    #plt.xlim()
    #plt.ylim()
    
    plt.suptitle("{} filter in dimension d={}".format(name,d))
    plt.tight_layout()
    
    num=num+1
    fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
    time=mean[:,0]
    
    ax1.plot(time,err[:,1],linewidth='2',linestyle='-',color='g',label='error x')
    ax1.plot(time,3*np.sqrt(cov[:,idx1-1,idx1-1]),linewidth='2',linestyle='-',color='r',label='3 std(x)')
    ax1.plot(time,-3*np.sqrt(cov[:,idx1-1,idx1-1]),linewidth='2',linestyle='-',color='r')
    ax1.set_xlabel("time")
    ax1.set_ylabel("error")
    ax1.set_title("error in position")
    #ax1.set_ylim(-0.5,0.5)
    ax1.legend()

    ax2.plot(time,err[:,2],linewidth='2',linestyle='-',color='g',label='error v')
    ax2.plot(time,3*np.sqrt(cov[:,idx2-1,idx2-1]),linewidth='2',linestyle='-',color='r',label='3 std(v)')
    ax2.plot(time,-3*np.sqrt(cov[:,idx2-1,idx2-1]),linewidth='2',linestyle='-',color='r')
    ax2.set_xlabel("time")
    ax2.set_ylabel("error")
    ax2.set_title("error in velocity")
    #ax2.set_ylim(-0.5,0.5)
    ax2.legend()
    if savefig:
        plt.savefig("outputs/KeplerMotionSTD_{}.pdf".format(name))
        

# if __name__ == "__main__":
    
#     TEST=["URM","UAM","UJM"]  
#     num=0
#     T=50
#     dobs=5
#     seed=10

#     if "URM" in TEST:
#         d=2
#         # the step must be adapt to the problem if we use Runge Kutta integration
#         # If the motion is uniform the matrix F is nilpotent 
#         # and it exist an exact integration scheme
#         dt=1 
#         print("The step dt used for Runge Kutta is,",dt)
#         mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num, name="Linear Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("Linear Kalman filtering \n" + r" Uniform rectilinear motion $A^2=0$")
#         plt.tight_layout()
#         plt.savefig("output/URM.pdf")
    
#     if "UAM" in TEST:
#         d=3
#         # the step must be adapt to the problem if we use Runge Kutta integration
#         # If the motion is uniform the matrix F is nilpotent 
#         # and it exist an exact integration scheme
#         dt=0.1 
#         print("The step dt used for Runge Kutta is,",dt)
#         mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num,name="Linear Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("Linear Kalman filtering \n" + r" Uniform accelerated motion $A^3=0$")
#         plt.tight_layout()
#         plt.savefig("output/UAM.pdf")
        
#     if "UJM" in TEST:
#         d=4
#         # the step must be adapt to the problem if we use Runge Kutta integration
#         # If the motion is uniform the matrix F is nilpotent 
#         # and it exist an exact integration scheme
#         dt=0.01 
#         print("The step dt used for Runge Kutta is,",dt)
#         mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
#         KalmanPlot(mean, cov, traj, num, name="Linear Kalman")
#         num=num+3
        
#         plt.xlim()
#         plt.ylim()
#         plt.suptitle("Linear Kalman filtering \n" + r" Uniform jerk motion $A^4=0$")
#         plt.tight_layout()
#         plt.savefig("output/UJM.pdf")
    

             