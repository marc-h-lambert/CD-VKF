import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sm
from sympy.abc import a
import numpy.linalg as LA
from .Core.NonLinearKalman import NonLinearObservator

class Polar:
    def Rot(Theta):
        Rot=np.zeros([2,2])
        Rot[0,0]=math.cos(Theta)
        Rot[0,1]=-math.sin(Theta)
        Rot[1,0]=math.sin(Theta)
        Rot[1,1]=math.cos(Theta)
        return Rot
    
class PolarObservatorActive(NonLinearObservator):

    def __init__(self,x,y,sigmaTheta,sigmaR,d=4):
        self.x=x
        self.y=y
        self.sigmaTheta=sigmaTheta
        self.sigmaR=sigmaR
        self.dimState=d
        
    def observe(self,X):
        x=X[0]
        y=X[1]
        [R, Theta]=self.predict(X)
        Rn=np.random.normal(R,self.sigmaR)
        Thetan=np.random.normal(Theta,self.sigmaTheta)
        return np.array([Rn,Thetan]).reshape(-1,1)
        
    # predict in global coord
    def predict(self,X):
        x=X[0]
        y=X[1]
        Theta=math.atan2(y-self.y,x-self.x)
        if Theta<0:
            Theta=Theta+2*math.pi
        R=math.sqrt((x-self.x)**2+(y-self.y)**2)
        return np.array([R,Theta]).reshape(-1,1)

    def covarianceMatrix(self,X):
        CovObs=np.identity(2)
        CovObs[0,0]=self.sigmaR**2
        CovObs[1,1]=self.sigmaTheta**2
        return CovObs 
        
    def jacobianObservation(self,X):
        x=X[0]
        y=X[1]        
        R=math.sqrt((x-self.x)**2+(y-self.y)**2)
        dRdx=(x-self.x)/R
        dRdy=(y-self.y)/R
        K=(y-self.y)/(x-self.x)
        dThetadx=-K/(x-self.x)/(1+K**2)
        dThetady=1/(x-self.x)/(1+K**2)
        H=np.zeros([2,self.dimState])
        H[0,0]=dRdx
        H[0,1]=dRdy
        H[1,0]=dThetadx
        H[1,1]=dThetady
        return H 
    
    def hessianObservation(self,X):
        x=X[0]
        y=X[1]
        R=math.sqrt((x-self.x)**2+(y-self.y)**2)
        d2Rd2x=1/R-(x-self.x)**2/R**3
        d2Rd2y=1/R-(y-self.y)**2/R**3
        d2Rdxy=-(y-self.y)*(x-self.x)/R**3
        K=(y-self.y)/(x-self.x)
        d2Thetad2x=2*K/(K**2+1)**2/(x-self.x)**2 
        d2Thetad2y=-2*K/(K**2+1)**2/(x-self.x)**2 
        d2Thetadxy=(K**2-1)/(K**2+1)**2/(x-self.x)**2 
        dH=np.zeros([2,self.dimState,self.dimState])
        dH[0,0,0]=d2Rd2x
        dH[0,0,1]=d2Rdxy
        dH[0,1,0]=d2Rdxy
        dH[0,1,1]=d2Rd2y
        dH[1,0,0]=d2Thetad2x
        dH[1,0,1]=d2Thetadxy
        dH[1,1,0]=d2Thetadxy
        dH[1,1,1]=d2Thetad2y
        return dH 
    
class PolarObservatorPassive(NonLinearObservator):

    def __init__(self,x,y,sigmaTheta,d=4):
        self.x=x
        self.y=y
        self.sigmaTheta=sigmaTheta
        self.dimState=d
        
    def observe(self,X):
        x=X[0]
        y=X[1]
        #Theta=math.atan2(y-self.y,x-self.x)
        Theta=self.predict(X)
        Thetan=np.random.normal(Theta,self.sigmaTheta)
        return np.array([Thetan]).reshape(-1,1)
        
    def predict(self,X):
        x=X[0]
        y=X[1]
        Theta=math.atan2(y-self.y,x-self.x)
        if Theta<0:
            Theta=Theta+2*math.pi
        return np.array([Theta]).reshape(-1,1)

    def covarianceMatrix(self,X):
        CovObs=np.identity(1)
        CovObs[0,0]=self.sigmaTheta**2
        return CovObs 
    
    def jacobianObservation(self,X):
        x=X[0]
        y=X[1]
        K=(y-self.y)/(x-self.x)
        dThetadx=-K/(x-self.x)/(1+K**2)
        dThetady=1/(x-self.x)/(1+K**2)
        H=np.zeros([1,self.dimState])
        H[0,0]=dThetadx
        H[0,1]=dThetady
        return H 
    
    def hessianObservation(self,X):
        x=X[0]
        y=X[1]
        K=(y-self.y)/(x-self.x)
        d2Thetad2x=2*K/(K**2+1)**2/(x-self.x)**2 
        d2Thetad2y=-2*K/(K**2+1)**2/(x-self.x)**2 
        d2Thetadxy=(K**2-1)/(K**2+1)**2/(x-self.x)**2 
        dH=np.zeros([1,self.dimState,self.dimState])
        dH[0,0,0]=d2Thetad2x
        dH[0,0,1]=d2Thetadxy
        dH[0,1,0]=d2Thetadxy
        dH[0,1,1]=d2Thetad2y
        return dH 
    

def trajObs(traj,polarObs,dobs):
    T=traj[-1,0]
    trajObs=np.zeros([int(T/dobs)+1,3])
    nObs=0
    for t in traj[:,0]:
        #print('t=',t)
        #print('dobs=',t)
        #print('t % dobs=',t % dobs)
        if t % dobs == 0:
            idx=np.where(traj[:,0] == t)
            X=traj[idx,1:3]
            yt=polarObs.observe(X.reshape(2,1))
            R=yt[0]
            theta=yt[1]
            xObs=polarObs.x+R*math.cos(theta)
            yObs=polarObs.y+R*math.sin(theta)
            trajObs[nObs]=np.array([t,xObs,yObs]).reshape(-1,)
            nObs=nObs+1
    return trajObs

if __name__ == "__main__":
    
    TEST=["Polar"]#,"Grad","Tensor"]
    
    sensor=PolarObservatorActive(0,0,10,0.1)
    if "Polar" in TEST:
        print(" ----------- TEST Polar ----------- ")
        x=sensor.x+100
        y=sensor.y+100
        X=np.array([x,y,0,0,0])
        print('[x,y]=',x,y)
        [R,theta]=sensor.observe(X)
        print('[R,theta]=',R,theta)
        xN=sensor.x+R*math.cos(theta)
        yN=sensor.y+R*math.sin(theta)
        print('[x2,y2]=',xN,yN)
        
    if "Grad" in TEST:
        print(" ----------- TEST Grad ----------- ")
        x=6500
        y=300
        vx=-1.8
        vy=-6.7
        a=0.7
        Xtest=np.array([x,y,vx,vy,a]).reshape(5,1)
        
        dH=np.zeros([2,2])
        dHg=sensor.jacobianObservation(Xtest)
        epsilon=1e-6
        for k in range(0,2):
            Xtest2=Xtest.copy()
            Xtest2[k]=Xtest[k]+epsilon
            dH[:,k]=((sensor.predict(Xtest2)-sensor.predict(Xtest))/epsilon).reshape(2,)
        print('Diff-dH={}'.format(dH))
        print('Grad-dH={}'.format(dHg[0:2,0:2]))
        
        d2H=np.zeros([2,2,2])
        d2Hg=sensor.hessianObservation(Xtest)
        epsilon=1e-6
        for k in range(0,2):
            Xtest2=Xtest.copy()
            Xtest2[k]=Xtest[k]+epsilon
            d2H[:,:,k]=((sensor.jacobianObservation(Xtest2)[0:2,0:2]-sensor.jacobianObservation(Xtest)[0:2,0:2])/epsilon).reshape(2,2)
        print('Diff-d2H={}'.format(d2H))
        print('Grad-d2H={}'.format(d2Hg[0:2,0:2,0:2]))
        
        # dHlogp=np.zeros([2,2])
        # Y=sensor.observe(Xtest)
        # yt=np.array([Y[0]-10,Y[1]-0.1]).reshape(2,)
        # invR=np.identity(2)
        # invR[0,0]=1/(1e-3)
        # invR[0,1]=1/(1e-4)
        # invR[1,0]=1/(1e-4)
        # invR[1,1]=1/(17*1e-3)
        # d2Hlogp=hessianLogP(Xtest,yt.reshape(2,1),sensor)
        # epsilon=1e-6
        # for k in range(0,2):
        #     Xtest2=Xtest.copy()
        #     Xtest2[k]=Xtest[k]+epsilon
        #     dHlogp[:,k]=((gradLogP(Xtest2,yt.reshape(2,1),sensor)[0:2]-gradLogP(Xtest,yt.reshape(2,1),sensor)[0:2])/epsilon).reshape(2,)
        # print('Diff-Hlogp={}'.format(dHlogp))
        # print('Grad-Hlogp={}'.format(d2Hlogp[0:2,0:2]))
    
    if "Tensor" in TEST:
        print(" ----------- TEST Tensor ----------- ")
        x0=6500
        y0=300
        X=np.array([x0,y0]).reshape(2,1)
        Y=sensor.observe(X)
        yt=np.array([Y[0]-10,Y[1]-0.1]).reshape(2,)
        invR=np.identity(2)
        invR[0,0]=1/(1e-3)
        invR[0,1]=1/(1e-4)
        invR[1,0]=1/(1e-4)
        invR[1,1]=1/(17*1e-3)
        
        J_obs=sensor.jacobianObservation(X)
        H_obs=sensor.hessianObservation(X)
        J_logp=gradLogP(X,yt.reshape(2,1),sensor)
        H_logp=hessianLogP(X,yt.reshape(2,1),sensor)
        
        print("grad_polarObs method1=",J_obs[0:2,0:2])
        print("hessian_polarObs method1=",H_obs[0:2,0:2,0:2])
        print("gradLogP method1=",J_logp[0:2,0:2])
        print("hessianLogP method1=",H_logp[0:2,0:2])
        
        x,y= sm.Symbol('x'),sm.Symbol('y')
        R=sm.sqrt((x-sensor.x)**2+(y-sensor.y)**2)
        Theta=sm.atan2(y-sensor.y,x-sensor.x)
        f=-0.5*((yt-(x,y)).T.dot(invR).dot(yt-(x,y)))
        #f=-0.5*sm.exp(invR[0,0]*(yt[0]-x)**2+invR[1,1]*(yt[1]-y)**2+2*invR[0,1]*(yt[0]-x)*(yt[1]-y))
        #print('f=',f.subs([(x,Y[0]),(y,Y[1])]))
        #print('R=',R.subs([(x,x0),(y,y0)]).evalf())
        #print('Theta=',Theta.subs([(x,x0),(y,y0)]).evalf()) 
    
        Jg=sm.derive_by_array([R,Theta],[x,y])
        print('Jg=',Jg.subs([(x,x0),(y,y0)])) 
            
        Hg=sm.derive_by_array(Jg,[x,y]) 
        print('Hg=',Hg.subs([(x,x0),(y,y0)])) 
        
        Jf=sm.derive_by_array(f,[x,y]) 
        print('Jf=',Jf.subs([(x,x0),(y,y0)])) 
        
        Hf=sm.derive_by_array(Jf,(x,y))  
        print('Hf=',Hf.subs([(x,x0),(y,y0)]))    
    
        # first part of the chain rule: Dˆ2f(g)*Dg*Dg
        res1=Hf.subs((x,R),(y,Theta))
        res1=sm.tensorcontraction(sm.tensorproduct(res1,Jg),[1,3])
        res1=sm.tensorcontraction(sm.tensorproduct(res1,Jg),[0,3])
    
        # second part of the chain rule : Df(g)*Dˆ2g
        print("Jf.shape",Jf.shape)
        print("Hg.shape",Hg.shape)
        res2=sm.tensorcontraction(sm.tensorproduct(Jf,Hg),[0,3])
    
        Hfg=sm.simplify(res1+res2)
        Hfg=Hfg.subs([(x,x0),(y,y0)])
        print("Hfg=",sm.simplify(Hfg))
        
        # g = sm.Matrix([R,Theta])
        # Jg=g.jacobian([x,y])
        # print('Jg=',Jg.evalf(subs={x:x0,y:y0}))
        # Hg=sm.hessian([R,Theta],[x,y])
        # print('Hg=',Hg.evalf(subs={x:x0,y:y0}))