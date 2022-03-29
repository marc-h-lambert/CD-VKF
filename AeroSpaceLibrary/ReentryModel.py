import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from .Core.Integration import rk4step, StepSDE, integrateRK
from .Core.NonLinearKalman import DynamicalSystem, NonLinearPredictor

getcontext().prec = 6

# constants
class PHY: 
    R0=6374
    H0=13.406
    Gm0=3.986*1e5

class INIT:
    x0=6500.4
    y0=349.14 
    vx0=-1.8093
    vy0=-6.7967
    a0_true=0.6932 
    beta0=-0.59783
    X0_true=np.array([x0,y0,vx0,vy0,a0_true]).reshape([5,1])
    
    qv=2.4064*1e-5
    qa=1e-6

class ReEntryVehicle(DynamicalSystem):
            
    def __init__(self,state0=INIT().X0_true,Q=np.zeros([5,5])):
        super().__init__(state0)
        self.Q=Q
    
     # X=(x,y,vx,vy,a)    
    def gravity(X):
        x=X[0]
        y=X[1]
        R=math.sqrt(x**2+y**2)
        F=np.zeros([5,1])
        F[2]=-x*PHY().Gm0/(R**3)
        F[3]=-y*PHY().Gm0/(R**3)
        return F
    
    # X=(x,y,vx,vy,a)    
    def drag(X):
        x=X[0]
        y=X[1]
        vx=X[2]
        vy=X[3]
        a=X[4]
        R=math.sqrt(x**2+y**2)
        V=math.sqrt(vx**2+vy**2)
        k=(PHY().R0-R)/PHY().H0
        #z=np.clip(k+a,-1e10,500)
        z=k+a
        u=INIT().beta0*math.exp(z)/2
        F=np.zeros([5,1])
        F[2]=u*V*vx
        F[3]=u*V*vy
        return F
    
    def dynamic(X):
        F=np.zeros([5,5])
        F[0,2]=1# vx
        F[1,3]=1#vy
        return F.dot(X)+ReEntryVehicle.gravity(X)+ReEntryVehicle.drag(X)
    
    def diffusionMatrix(self,X,t):
        return np.identity(5)

    def covarianceMatrix(self):
        return self.Q
    
    def propagate(self,dt,T,noise=False):
        t=0
        d=self.state.shape[0]
        while t<T-1e-10:
            t=t+Decimal(dt) 
            self.time=self.time+Decimal(dt) 
            if not noise:
                self.state=rk4step(ReEntryVehicle.dynamic,dt,self.state)
            else:
                D=self.diffusionMatrix(self.state,t)
                Q=self.covarianceMatrix()
                self.state=StepSDE(ReEntryVehicle.dynamic,D,dt,Q,self.state)
            self.traj.append(np.concatenate(([self.time], self.state.reshape(-1,)), axis=0))
        return self.state
            
    
class ReEntryPredictor(NonLinearPredictor):
        
    def __init__(self,beta0=INIT().beta0,Q=np.zeros([5,5])):
        super().__init__()
        self.beta0=beta0
        self.Q=Q
    
    def dynamic(self,X):
        return ReEntryVehicle.dynamic(X)
    
    def diffusionMatrix(self,X,t):
        return np.identity(5)

    def covarianceMatrix(self):
        return self.Q
    
    def jacobianDynamic(self,X):
        x=X[0]
        y=X[1]
        vx=X[2]
        vy=X[3]
        a=X[4]
        R=math.sqrt(x**2+y**2)
        V=math.sqrt(vx**2+vy**2)
        Gm0=PHY().Gm0
        H0=PHY().H0
        R0=PHY().R0
        
        k=(R0-R)/H0
        
        dgxdx=-Gm0/(R**3)+3*Gm0*x**2/(R**5)
        dgxdy=3*Gm0*x*y/(R**5)
        dgydx=3*Gm0*x*y/(R**5)
        dgydy=-Gm0/(R**3)+3*Gm0*y**2/(R**5)
        
        z=k+a
        u=self.beta0*math.exp(z)/2
        dfxdx=-x/(H0*R)*u*V*vx
        dfxdy=-y/(H0*R)*u*V*vx
        dfxdVx=(V+vx**2/V)*u
        dfxdVy=(vx*vy/V)*u
        dfxda=u*V*vx
        
        dfydx=-x/(H0*R)*u*V*vy
        dfydy=-y/(H0*R)*u*V*vy
        dfydVx=(vx*vy/V)*u
        dfydVy=(V+vy**2/V)*u
        dfyda=u*V*vy
        
        F=np.zeros([5,5])
        F[0,2]=1 # dx=Vx
        F[1,3]=1 # dy=Vy
        F[2,0]=dgxdx+dfxdx #dVx=df/dx
        F[2,1]=dgxdy+dfxdy #dVx=df/dy
        F[2,2]=dfxdVx #dVx=df/dVx
        F[2,3]=dfxdVy #dVx=df/dVy
        F[2,4]=dfxda #dVx=df/da
        F[3,0]=dgydx+dfydx #dVy=df/dx
        F[3,1]=dgydy+dfydy #dVy=df/dy
        F[3,2]=dfydVx #dVy=df/dVx
        F[3,3]=dfydVy #dVy=df/dVy
        F[3,4]=dfyda #dVy=df/da
        return F 

if __name__ == "__main__":

    TEST=[ "Grad","Traj"]
    num=0
    
    if "Grad" in TEST:
        print(" ----------- TEST Grad ----------- ")
        x=6500
        y=300
        vx=-1.8
        vy=-6.7
        a=0.7
        Xtest=np.array([x,y,vx,vy,a]).reshape(5,1)
            
        RVpred=ReEntryPredictor()
        dF=np.zeros([5,5])
        dFg=RVpred.jacobianDynamic(Xtest)
        epsilon=1e-6
        for k in range(0,5):
            Xtest2=Xtest.copy()
            Xtest2[k]=Xtest[k]+epsilon
            dF[:,k]=((ReEntryVehicle.dynamic(Xtest2)-ReEntryVehicle.dynamic(Xtest))/epsilon).reshape(5,)
        print('Diff-dF={}'.format(dF))
        print('Grad-dF={}'.format(dFg))
    
    if "Traj" in TEST:
        print(" ----------- TEST Traj ----------- ")
        seed=2
        np.random.seed(seed)
        T=200
        dt=0.1
        traj=integrateRK(ReEntryVehicle.dynamic,T,dt,INIT().X0)
        dobs=1
    
        num=num+1
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(10,5),num=num)
        ax1.plot(traj[:,1],traj[:,2],linewidth='2',label='traj-real')
        ax1.legend()
        ax1.set_title("position: y vs x")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_xlim([6350, 6500])
        ax1.set_ylim([-200, 500])
        ax1.grid()
        ax2.plot(traj[:,0],traj[:,3],linewidth='2',label='Vx')
        ax2.plot(traj[:,0],traj[:,4],linewidth='2',label='Vy')
        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('vel (m/s)')
        ax2.set_title("velocities: Vx and Vy")
        ax2.grid()
        ax2.legend()
        plt.tight_layout()