import rospy
import numpy as np
from numpy import tan, arctan, cos, sin, pi
from numpy.random import randn,rand
import random

class simulator(object):

    def __init__(self):

        self.L_f    = rospy.get_param("lf")
        self.L_r    = rospy.get_param("lr")
        self.m      = rospy.get_param("m")
        self.I_z    = rospy.get_param("Iz")
        self.Cf     = rospy.get_param("Cf")
        self.Cr     = rospy.get_param("Cr")
        self.mu     = rospy.get_param("mu")

        self.g = 9.81

        # with noise
        self.x      = 0.0
        self.y      = 0.0
        self.vx     = rospy.get_param("init_vx")
        self.vy     = 0.0
        self.ax     = 0.0
        self.ay     = 0.0

        self.yaw        = rospy.get_param("init_yaw")
        self.dist_mode  = rospy.get_param("dist_mode")
        self.mu_sf      = rospy.get_param("mu_sf")
        self.Cd         = rospy.get_param("Cd")
        self.A_car      = rospy.get_param("A_car")


        self.psiDot = 0.0

        self.x_his      = []
        self.y_his      = []
        self.vx_his     = []
        self.vy_his     = []
        self.ax_his     = []
        self.ay_his     = []
        self.psiDot_his = []
        self.noise_hist = []

        self.dt         = rospy.get_param("dt")
        self.rate       = rospy.Rate(1.0/self.dt)
        # self.rate         = rospy.Rate(1.0)
        self.time_his   = []

        # Get process noise limits
        self.x_std           = rospy.get_param("x_std_pr")/self.dt
        self.y_std           = rospy.get_param("y_std_pr")/self.dt
        self.vx_std          = rospy.get_param("vx_std_pr")/self.dt
        self.vy_std          = rospy.get_param("vy_std_pr")/self.dt
        self.psiDot_std      = rospy.get_param("psiDot_std_pr")/self.dt
        self.psi_std         = rospy.get_param("psi_std_pr")/self.dt
        self.n_bound         = rospy.get_param("n_bound_pr")/self.dt

        #Get sensor noise limits

        self.x_std_s           = rospy.get_param("x_std")
        self.y_std_s           = rospy.get_param("y_std")
        self.vx_std_s          = rospy.get_param("vx_std")
        self.vy_std_s          = rospy.get_param("vy_std")
        self.psiDot_std_s      = rospy.get_param("psiDot_std")
        self.psi_std_s         = rospy.get_param("psi_std")
        self.n_bound_s         = rospy.get_param("n_bound")
        self.slip_angle        = False

    def f(self,u):
        a_F = 0.0
        a_R = 0.0

        if abs(self.vx) > 0.2:
            a_F = u[1] - arctan((self.vy + self.L_f*self.psiDot)/abs(self.vx))
            a_R = arctan((- self.vy + self.L_r*self.psiDot)/abs(self.vx))

        # FyF = self.pacejka(a_F)
        # FyR = self.pacejka(a_R)
        FyF = 60*a_F
        FyR = 60*a_R


        if abs(a_F) > 30.0/180.0*pi or abs(a_R) > 30.0/180.0*pi:
            print ("WARNING: Large slip angles in simulation")
            self.slip_angle        = True
        else:
            self.slip_angle        = False
        x   = self.x
        y   = self.y
        ax  = self.ax
        ay  = self.ay
        vx  = self.vx
        vy  = self.vy
        yaw = self.yaw
        psiDot = self.psiDot
                
        self.x      += self.dt*(cos(yaw)*vx - sin(yaw)*vy)
        self.y      += self.dt*(sin(yaw)*vx + cos(yaw)*vy)
        self.vx     += self.dt*(ax + psiDot*vy)
        self.vy     += self.dt*(ay - psiDot*vx)
        self.ax      = u[0] - self.mu*vx - FyF/self.m*sin(u[1])
        self.ay      = 1.0/self.m*(FyF*cos(u[1])+FyR)
        self.yaw    += self.dt*(psiDot)                                        
        self.psiDot += self.dt*(1.0/self.I_z*(self.L_f*FyF*cos(u[1]) - self.L_r*FyR))
        print ("self.yaw",self.yaw*180.0/pi)
        self.vx = abs(self.vx)
    

    def pacejka(self,ang):
        D = self.c_f*self.m*self.g/2    # Friction force/2
        Stiffness_Force = D*sin(self.C*arctan(self.B*ang))
        return Stiffness_Force

    def saveHistory(self):
        self.x_his.append(self.x)
        self.y_his.append(self.y)
        self.vx_his.append(self.vx)
        self.vy_his.append(self.vy)
        self.ax_his.append(self.ax)
        self.ay_his.append(self.ay)
        self.psiDot_his.append(self.psiDot)
        self.time_his.append(rospy.get_rostime().to_sec()) 
