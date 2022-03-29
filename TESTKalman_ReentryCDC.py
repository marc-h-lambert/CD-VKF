###################################################################################
# THE KALMAN FILTER LIBRARY                                                       #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "The continuous-discrete variational Kalman filter (CD-VKF)"                    #
# Authors: Marc Lambert, Silvere Bonnabel and Francis Bach                        #
###################################################################################

import numpy as np
import matplotlib.pyplot as plt
#from plot4latex import set_size
import math
from decimal import *
import numpy.linalg as LA
from AeroSpaceLibrary.ReentryModel import INIT, ReEntryVehicle, ReEntryPredictor
from AeroSpaceLibrary.Core.Integration import integrateRK, IntegrateEuler, IntegrateSDE, StepSDE, rk4step
from AeroSpaceLibrary.Sensor2D import PolarObservatorActive,  trajObs
from AeroSpaceLibrary.Core.NonLinearKalman import CD_EKF, CD_UKF, CD_VKF

getcontext().prec = 6
    
def plotTraj(ax1,ax2,traj,sensor,label='traj-Real',trajO=None):
    ax1.plot(traj[:,1],traj[:,2],linewidth='2',label=label)
    if not trajO is None:
        ax1.scatter(trajO[:,1],trajO[:,2],s=10,color='r',label='traj-obs')
        ax1.scatter(sensor.x,sensor.y,marker='*',color='g',label='sensor')
    ax1.set_title("position: y vs x")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim([6350, 6500])
    ax1.set_ylim([-200, 500])
    ax1.grid()
    ax1.legend()
    
    ax2.plot(traj[:,0],traj[:,3],linewidth='2',label='Vx'+'('+label+')')
    ax2.plot(traj[:,0],traj[:,4],linewidth='2',label='Vy'+'('+label+')')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('vel (km/s)')
    ax2.set_title("velocities: Vx and Vy")
    ax2.grid()
    ax2.legend()

def plotErrors(kalman,X0_true,P0,name,num,nbMC,seed=1,sqrt=False):        
    qv=2.4064*1e-5
    Q_vehic=np.zeros([5,5])
    Q_vehic[2,2]=qv
    Q_vehic[3,3]=qv
    Q_vehic[4,4]=0
    P0_vehic=P0.copy()
    P0_vehic[4,4]=0 # the aerodynamic coef does not change in MC
    
    X0_filter=X0_true.copy().reshape(-1,)
    X0_filter[4]=0 # the aerodynamic coef is systematicaly wrong in filter
    
    qa=1e-6
    Q_filter=np.zeros([5,5])
    Q_filter[2,2]=qv
    Q_filter[3,3]=qv
    Q_filter[4,4]=qa
    predictor=ReEntryPredictor(Q=Q_filter)
    
    # The limit value for VKF are sigmaR=0.01 and sigmaTheta=0.01
    # above the matrix R^-1 make the cov update no more SDP
    xs=6374
    ys=0
    sigmaR=0.1#1e-3
    sigmaTheta=0.1#17*1e-3
    observator=PolarObservatorActive(xs,ys,sigmaTheta,sigmaR,d=5)
    
    run=0
    for seed_run in np.linspace(seed,seed*nbMC+1,nbMC):
        run=run+1
        print("processing run N°",run)
        np.random.seed(int(seed_run))
        
        X0_vehic=np.random.multivariate_normal(X0_true.reshape(-1,),P0_vehic)
        X0_vehic=X0_vehic.reshape([-1,1])
        X0=np.random.multivariate_normal(X0_filter,P0)
        X0=X0.reshape([-1,1])
        
        system=ReEntryVehicle(state0=X0_true,Q=Q_vehic)
        kf=kalman(X0,P0,sqrt)
        mean, cov=kf.filtering(predictor,observator,system,dt,T,dobs)
        traj=system.trajectory()
        
        if run==1:
            errors=(traj-mean)**2
            _mean=mean
            _cov=cov
        else:
            errors=errors+(traj-mean)**2
            _mean=_mean+mean
            _cov=_cov+cov
    
    errors = np.array(errors, dtype=np.float64) 
    return _mean/nbMC, _cov/nbMC, errors/nbMC

def PlotOutput(ax,mean,cov,X0_True,P0,Q,T,dt,idx,nbRuns=3,seed=1,randX0=True,s=3,s2=9):
    P0_vehic=P0.copy()
    P0_vehic[4,4]=0 # the aerodynamic coef does not change 
    plotLegend=True
    for seed_run in np.linspace(seed,seed*nbRuns+1,nbRuns):
        np.random.seed(int(seed_run))
        if randX0:
            X0=np.random.multivariate_normal(X0_True.reshape(-1,),P0_vehic)
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

    std1=mean[:,idx]+s*np.sqrt(cov[:,idx-1,idx-1])
    std2=mean[:,idx]-s*np.sqrt(cov[:,idx-1,idx-1])
    ax.plot(time,std1,linewidth='1.5',linestyle='-',color='r',label=r'std (at $3\sigma$)')
    ax.plot(time,std2,linewidth='1.5',linestyle='-',color='r')
    std1=mean[:,idx]+s2*np.sqrt(cov[:,idx-1,idx-1])
    std2=mean[:,idx]-s2*np.sqrt(cov[:,idx-1,idx-1])
    ax.fill(np.append(time, time[::-1]), np.append(std1, std2[::-1]), 'darkgray')
    ax.set_xlabel("time")
    
def initialize():
    x0=6500.4
    y0=349.14 
    vx0=-1.8093
    vy0=-6.7967
    a0_true=0.6932 
    beta0=-0.59783
    X0_true=np.array([x0,y0,vx0,vy0,a0_true]).reshape([5,1])
    return X0_true
    
if __name__ == "__main__":
    
    
    TEST=["CDVKF_ReentryTracking_output","ReentryTracking_bench"]
    
    num=0
    T=200
    seed=1
    nbMC=5
    dobs=Decimal("0.5")
    dt=0.25
    X0=initialize()
    ex=1e-6
    ev=1e-6
    ea=1
    P0=np.diag([ex,ex,ev,ev,ea])
    sqrt=True
    
    if "ReentryTracking_bench" in TEST:
        print("-------- compute EKF -------")
        mean_ekf, cov_ekf, MSE_ekf=plotErrors(CD_EKF,X0,P0,"ekf",num,nbMC=nbMC,seed=seed,sqrt=sqrt)
        print("-------- compute UKF -------")
        mean_ukf, cov_ukf, MSE_ukf=plotErrors(CD_UKF,X0,P0,"ukf",num,nbMC=nbMC,seed=seed,sqrt=sqrt)
        print("-------- compute VKF -------")
        mean_vkf, cov_vkf, MSE_vkf=plotErrors(CD_VKF,X0,P0,"vkf",num,nbMC=nbMC,seed=seed,sqrt=sqrt)
        time=mean_ekf[:,0]
                
        # Plot covariances
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(10,5),num=num)

        ax1.semilogy(time,np.sqrt(cov_ekf[:,1,1]),linewidth='3',linestyle='-',color='b')
        ax1.semilogy(time,np.sqrt(cov_ukf[:,1,1]),linewidth='3',linestyle='-',color='g')
        ax1.semilogy(time,np.sqrt(cov_vkf[:,1,1]),linewidth='3',linestyle='-',color='r')
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"std x $(km)^2$")
        
        ax2.semilogy(time,np.sqrt(cov_ekf[:,2,2]),linewidth='3',linestyle='-',color='b')
        ax2.semilogy(time,np.sqrt(cov_ukf[:,2,2]),linewidth='3',linestyle='-',color='g')
        ax2.semilogy(time,np.sqrt(cov_vkf[:,2,2]),linewidth='3',linestyle='-',color='r')
        ax2.set_xlabel("time")
        ax2.set_ylabel(r"std Vx $(km/s)^2$")
        
        ax3.semilogy(time,np.sqrt(cov_ekf[:,4,4]),linewidth='3',linestyle='-',color='b',label='EKF')
        ax3.semilogy(time,np.sqrt(cov_ukf[:,4,4]),linewidth='3',linestyle='-',color='g',label='UKF')
        ax3.semilogy(time,np.sqrt(cov_vkf[:,4,4]),linewidth='3',linestyle='-',color='r',label='VKF')
        ax3.set_xlabel("time")
        ax3.set_ylabel(r"std coef a")
        ax3.legend()
        plt.suptitle("Estimated covariance matrix : standard deviation on position, velocity and aerodynamic coefficient")
        plt.tight_layout()
        plt.savefig("outputs/CDKF_COV.pdf") 
        
        num=num+1
        # Plot MSE
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(10,5),num=num)
        ax1.semilogy(time,np.sqrt(MSE_ekf[:,2]),linewidth='3',linestyle='-',color='b')
        ax1.semilogy(time,np.sqrt(MSE_ukf[:,2]),linewidth='3',linestyle='-',color='g')
        ax1.semilogy(time,np.sqrt(MSE_vkf[:,2]),linewidth='3',linestyle='-',color='r')
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"std x $(km)^2$")
        
        ax2.semilogy(time,np.sqrt(MSE_ekf[:,3]),linewidth='3',linestyle='-',color='b')
        ax2.semilogy(time,np.sqrt(MSE_ukf[:,3]),linewidth='3',linestyle='-',color='g')
        ax2.semilogy(time,np.sqrt(MSE_vkf[:,3]),linewidth='3',linestyle='-',color='r')
        ax2.set_xlabel("time")
        ax2.set_ylabel(r"std Vx $(km/s)^2$")
        
        ax3.semilogy(time,np.sqrt(MSE_ekf[:,5]),linewidth='3',linestyle='-',color='b',label='EKF')
        ax3.semilogy(time,np.sqrt(MSE_ukf[:,5]),linewidth='3',linestyle='-',color='g',label='UKF')
        ax3.semilogy(time,np.sqrt(MSE_vkf[:,5]),linewidth='3',linestyle='-',color='r',label='VKF')
        ax3.set_xlabel("time")
        ax3.set_ylabel(r"std coef a")
        ax3.legend()
        plt.suptitle("Estimation error : RMSE error on position, velocity and aerodynamic coefficient")
        plt.tight_layout()
        plt.savefig("outputs/CDKF_COV.pdf") 
        
       
        
    if "CDVKF_ReentryTracking_output" in TEST:
        print("-------- compute outputs for VKF -------")
        name="CD-VKF"
        mean_kf, cov_kf, MSE_kf=plotErrors(CD_VKF,X0,P0,"vkf",num,nbMC=1,seed=seed,sqrt=sqrt)
        qv=2.4064*1e-5
        Q_vehic=np.zeros([5,5])
        Q_vehic[2,2]=qv
        Q_vehic[3,3]=qv
        Q_vehic[4,4]=0
        nbRuns=5
        
        num=num+1
        ksigma=3
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(10,5),num=num)
        PlotOutput(ax1,mean_kf,cov_kf,X0,P0,Q_vehic,T,dt,idx=1,nbRuns=nbRuns,seed=seed,randX0=True, s=ksigma)
        ax1.set_ylabel("position x")
        #ax1.set_ylim(-3,1)
        PlotOutput(ax2,mean_kf,cov_kf,X0,P0,Q_vehic,T,dt,idx=3,nbRuns=nbRuns,seed=seed,randX0=True, s=ksigma)
        ax2.set_ylabel("velocity Vx")
        PlotOutput(ax3,mean_kf,cov_kf,X0,P0,Q_vehic,T,dt,idx=5,nbRuns=nbRuns,seed=seed,randX0=True, s=ksigma)
        ax3.set_ylabel("aerodynamic coef a")
        ax3.legend()
        #ax2.set_ylim(-200,300)
        plt.suptitle("Uncertainties estimated by the CD-VKF filter ")
        plt.tight_layout()
        plt.savefig("outputs/"+name+"_Output.pdf") 
        
    if "Test_CDVKF" in TEST:
        print("-------- compute VKF -------")
        mean_vkf, cov_vkf, MSE_vkf=plotErrors(CD_VKF,X0,P0,"vkf",num,nbMC=nbMC,seed=seed,sqrt=sqrt)
        time=mean_vkf[:,0]
        
        num=num+1
        fig, (ax1) = plt.subplots(1, 1,figsize=(10,5),num=num)
        ax1.semilogy(time,np.sqrt(MSE_vkf[:,5]),linewidth='3',linestyle='--',color='b',label='RMSE')
        ax1.semilogy(time,3*np.sqrt(cov_vkf[:,4,4]),linewidth='3',linestyle='-',color='r',label='deviation at 3 sigma')
        ax1.legend()
        plt.suptitle("True RMSE error vs estimated deviation on aero. coef")
        plt.tight_layout()
        plt.savefig("outputs/CDKF_VKF.pdf")
        
    if "TestCovariance" in TEST:
        print("-------- compute EKF -------")
        mean_ekf, cov_ekf, MSE_ekf=plotErrors(CD_EKF,X0,P0,"ekf",num,nbMC=1,seed=seed,sqrt=sqrt)
        print("-------- compute UKF -------")
        mean_ukf, cov_ukf, MSE_ukf=plotErrors(CD_UKF,X0,P0,"ukf",num,nbMC=1,seed=seed,sqrt=sqrt)
        print("-------- compute VKF -------")
        mean_vkf, cov_vkf, MSE_vkf=plotErrors(CD_VKF,X0,P0,"vkf",num,nbMC=1,seed=seed,sqrt=sqrt)
        time=mean_ekf[:,0]
                
        # Plot covariances
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(10,5),num=num)

        #ax1.semilogy(time,MSE_kf[:,2],linewidth='2',linestyle='--',color='b',label='MSE(y)-'+name)
        ax1.semilogy(time,np.sqrt(cov_ekf[:,1,1]),linewidth='3',linestyle='-',color='b')
        ax1.semilogy(time,np.sqrt(cov_ukf[:,1,1]),linewidth='3',linestyle='-',color='g')
        ax1.semilogy(time,np.sqrt(cov_vkf[:,1,1]),linewidth='3',linestyle='-',color='r')
        ax1.set_xlabel("time")
        ax1.set_ylabel(r"std x $(km)^2$")
        
        #ax2.semilogy(time,MSE_kf[:,3],linewidth='2',linestyle='--',color='b',label='MSE(Vx)-'+name)
        ax2.semilogy(time,np.sqrt(cov_ekf[:,2,2]),linewidth='3',linestyle='-',color='b')
        ax2.semilogy(time,np.sqrt(cov_ukf[:,2,2]),linewidth='3',linestyle='-',color='g')
        ax2.semilogy(time,np.sqrt(cov_vkf[:,2,2]),linewidth='3',linestyle='-',color='r')
        ax2.set_xlabel("time")
        ax2.set_ylabel(r"std Vx $(km/s)^2$")
        
        #ax3.semilogy(time,MSE_kf[:,5],linewidth='2',linestyle='--',color='b',label='MSE(a)-'+name)
        ax3.semilogy(time,np.sqrt(cov_ekf[:,4,4]),linewidth='3',linestyle='-',color='b',label='EKF')
        ax3.semilogy(time,np.sqrt(cov_ukf[:,4,4]),linewidth='3',linestyle='-',color='g',label='UKF')
        ax3.semilogy(time,np.sqrt(cov_vkf[:,4,4]),linewidth='3',linestyle='-',color='r',label='VKF')
        ax3.set_xlabel("time")
        ax3.set_ylabel(r"std coef a")
        ax3.legend()
        plt.suptitle("Estimated covariance matrix : standard deviation on position, velocity and aerodynamic coefficient")
        plt.tight_layout()
        plt.savefig("outputs/CDKF_COV.pdf") 
   
   
    

             