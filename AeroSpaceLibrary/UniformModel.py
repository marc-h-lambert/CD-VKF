import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
#from Core.Integration import rk4step
from .Core.Kalman import DynamicalSystem, KalmanLinearPredictor, KalmanLinearObservator
from abc import ABCMeta, abstractmethod

getcontext().prec = 6

# simple exemple of class instance
class UniformMotion1D(DynamicalSystem):
    
    def __init__(self,state0):
        super().__init__(state0)
        d=self.state.shape[0]
        self.A=np.zeros([d,d])
        for i in range(0,d-1):
            self.A[i,i+1]=1
        
    def propagate(self,dt,T,noise=True):
        t=0
        d=self.state.shape[0]
        while t<T-1e-10:
            t=t+Decimal(dt)
            self.time=self.time+Decimal(dt)
            self.state=self.state+self.A.dot(self.state)*dt
            self.traj.append(np.concatenate(([self.time], self.state.reshape(d,)), axis=0))
        return self.state
    
# simple exemple of class instance
class UniformMotion1D_Predictor(KalmanLinearPredictor):
    
    def __init__(self,d):
        self.dim=d
        self.A=np.zeros([d,d])
        for i in range(0,d-1):
            self.A[i,i+1]=1
    
    @abstractmethod
    def dynamicMatrix(self):
        return self.A
    
    @abstractmethod
    def dynamic(self,x):
        return self.A.dot(x)
    
    @abstractmethod
    def diffusionMatrix(self,x,t):
        D=np.identity(self.dim)
        return D

    @abstractmethod
    def covarianceMatrix(self):
        Q=np.identity(self.dim)*0
        return Q
    
# In the case of 1D uniform motion, this predictor is linear 
# but satisfy the non linear API (for Test)
class U1D_NonLinearPredictor(UniformMotion1D_Predictor):

    def jacobianDynamic(self,state):
        return self.dynamicMatrix()
    
# simple exemple of class instance
class scalar_Observator(KalmanLinearObservator):
    
    def __init__(self,d):
        self.H=np.zeros([1,d])
        self.H[0,0]=1
            
    def observationMatrix(self):
        return self.H
    
    def observe(self,x):
        eps=np.random.normal(0,1)
        return self.H.dot(x)+eps

    def predict(self,x):
        return self.H.dot(x) 

    def covarianceMatrix(self,state):
        return np.identity(1)
    
# In the case of scalar observation, this observator is linear 
# but satisfy the non linear API (for Test)
class scalar_NonLinearObservator(scalar_Observator):

    def jacobianObservation(self,state):
        return self.observationMatrix()
    
    def hessianObservation(self,state):
        n,d=self.observationMatrix().shape
        return np.zeros([n,d,d])
    
# simple exemple 
def TestUniformMotion(TypeKalman,d,seed,T,dt,dobs):
    
    np.random.seed(seed)
    
    # initialize scenario
    X0_true=np.zeros([d,1])
    X0_true[-1]=1
    system=UniformMotion1D(X0_true)
    predictor=UniformMotion1D_Predictor(d)
    observator=scalar_Observator(d)
    
    # initial guess
    P0=np.identity(d)
    X0=np.random.multivariate_normal(X0_true.reshape(-1,),P0)
    X0=X0.reshape([d,1])
    print('X0=',X0)
    
    # filtering
    kalmanfilter=TypeKalman(X0, P0)
    mean,cov = kalmanfilter.filtering(predictor,observator,system,dt,T,dobs)
    traj=system.traj
    
    return mean, cov, traj

# simple exemple 
def TestUniformMotionNL(TypeKalman,d,seed,T,dt,dobs):
    
    np.random.seed(seed)
    
    # initialize scenario
    X0_true=np.zeros([d,1])
    X0_true[-1]=1
    system=UniformMotion1D(X0_true)
    predictor=U1D_NonLinearPredictor(d)
    observator=scalar_NonLinearObservator(d)
    
    # initial guess
    P0=np.identity(d)
    X0=np.random.multivariate_normal(X0_true.reshape(-1,),P0)
    X0=X0.reshape([d,1])
    
    # filtering
    kalmanfilter=TypeKalman(X0, P0)
    mean,cov = kalmanfilter.filtering(predictor,observator,system,dt,T,dobs)
    traj=system.traj
    
    return mean, cov, traj
             