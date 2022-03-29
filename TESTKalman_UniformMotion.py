import numpy as np
import matplotlib.pyplot as plt
import math
from decimal import *
import numpy.linalg as LA
from AeroSpaceLibrary.Core.Integration import rk4step
from AeroSpaceLibrary.UniformModel import UniformMotion1D, UniformMotion1D_Predictor,\
    scalar_Observator, TestUniformMotion, TestUniformMotionNL
from AeroSpaceLibrary.Core.NonLinearKalman import CD_EKF, CD_UKF,CD_VKF
from AeroSpaceLibrary.Core.Kalman import LinearKalmanFilter, KalmanPlot

if __name__ == "__main__":
    
    TEST=["EKF","URM"]#,"UAM","UJM"]  
    num=0
    T=100
    dobs=5
    seed=10
    dt=0.1 
     
    if "EKF" in TEST:
        d=2
        # the step must be adapt to the problem if we use Runge Kutta integration
        # If the motion is uniform the matrix F is nilpotent 
        # and it exist an exact integration scheme
       
        print("The step dt used for Runge Kutta is,",dt)
        mean, cov, traj= TestUniformMotionNL(CD_UKF,d,seed,T,dt,dobs)
        KalmanPlot(mean, cov, traj, num,"EKFURM")
        num=num+2
        
        plt.xlim()
        plt.ylim()
        plt.suptitle("Linear Kalman filtering \n" + r" Uniform rectilinear motion $A^2=0$")
        plt.tight_layout()
        plt.savefig("outputs/CDEKF_URM.pdf")
        
    if "URM" in TEST:
        d=2
        # the step must be adapt to the problem if we use Runge Kutta integration
        # If the motion is uniform the matrix F is nilpotent 
        # and it exist an exact integration scheme
        print("The step dt used for Runge Kutta is,",dt)
        mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
        KalmanPlot(mean, cov, traj, num, "URM")
        num=num+2
        
        plt.xlim()
        plt.ylim()
        plt.suptitle("Linear Kalman filtering \n" + r" Uniform rectilinear motion $A^2=0$")
        plt.tight_layout()
        plt.savefig("outputs/URM.pdf")
    
    if "UAM" in TEST:
        d=3
        # the step must be adapt to the problem if we use Runge Kutta integration
        # If the motion is uniform the matrix F is nilpotent 
        # and it exist an exact integration scheme
        print("The step dt used for Runge Kutta is,",dt)
        mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
        KalmanPlot(mean, cov, traj, num, "UAM")
        num=num+2
        
        plt.xlim()
        plt.ylim()
        plt.suptitle("Linear Kalman filtering \n" + r" Uniform accelerated motion $A^3=0$")
        plt.tight_layout()
        plt.savefig("outputs/UAM.pdf")
        
    if "UJM" in TEST:
        d=4
        # the step must be adapt to the problem if we use Runge Kutta integration
        # If the motion is uniform the matrix F is nilpotent 
        # and it exist an exact integration scheme
        print("The step dt used for Runge Kutta is,",dt)
        mean, cov, traj= TestUniformMotion(LinearKalmanFilter,d,seed,T,dt,dobs)
        KalmanPlot(mean, cov, traj, num, "UJM")
        num=num+2
        
        plt.xlim()
        plt.ylim()
        plt.suptitle("Linear Kalman filtering \n" + r" Uniform jerk motion $A^4=0$")
        plt.tight_layout()
        plt.savefig("outputs/UJM.pdf")

