#!/usr/bin/env python



import gym
import rospy
import roslaunch
import time
import numpy as np
from numpy import tan, arctan, cos, sin, pi
from numpy.random import randn,rand
from tf import transformations
import cv2
import sys
import os
import random
from gazebo_connection import GazeboConnection

from gym.envs.registration import register
from gym import utils, spaces
#from gym_gazebo.envs import gazebo_env
#from geometry_msgs.msg import Twist
from std_srvs.srv import Empty

from sensor_msgs.msg import Image
#from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError

import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
# from gazebo_connection import GazeboConnection
################################################################################
# message type to publish acceleration and steering angle
from simulation_control.msg import ECU

################################################################################
# message type to receive the velocity and position
from simulation_control.msg import simulatorStates

from simulation_control.msg import errorStates

from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Vector3, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from vehicle_error import *
from vehicle_error import _initializeFigure_xy

###############################################################################
#REGISTERING THE ENVIRONMENT, first arg in entry point name file, second class name
###############################################################################
reg = register(
    id='VehicleDQN-v0',
    entry_point='env_vehicle_dqn:vehicle_dqn',
    max_episode_steps=1000
    )

class Simulatornew(object):
    """ Object collecting GPS measurement data
    Attributes:
        Model params:
            1.L_f 2.L_r 3.m(car mass) 3.I_z(car inertial) 4.c_f(equivalent drag coefficient)
        States:
            1.x 2.y 3.vx 4.vy 5.ax 6.ay 7.psiDot
        States history:
            1.x_his 2.y_his 3.vx_his 4.vy_his 5.ax_his 6.ay_his 7.psiDot_his
        Simulator sampling time:
            1. dt
        Time stamp:
            1. time_his
    Methods:
        f(u):
            System model used to update the states
        pacejka(ang):
            Pacejka lateral tire modeling
    """
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
        u = np.array([0.56,0.1])
        print ">>steering",u[1]
        if abs(self.vx) > 0.7:
            a_F = u[1] - arctan((self.vy + self.L_f*self.psiDot)/abs(self.vx))
            a_R = arctan((- self.vy + self.L_r*self.psiDot)/abs(self.vx))

        # FyF = self.pacejka(a_F)
        # FyR = self.pacejka(a_R)

        FyF = self.Cf * a_F
        FyR = self.Cr * a_R


        if abs(a_F) > 30.0/180.0*pi or abs(a_R) > 30.0/180.0*pi:
            print "WARNING: Large slip angles in simulation"

        x       = self.x
        y       = self.y
        ax      = self.ax
        ay      = self.ay
        vx      = self.vx
        vy      = self.vy
        yaw     = self.yaw
        psiDot  = self.psiDot

        if self.dist_mode:

            dist = ( 10*self.Cd*1.225*self.A_car*(vx**2) + self.mu_sf*9.81*self.m)/self.m

        else:
            dist = self.mu*vx
        dist = 0
        #print("disturbance: " , dist)


        #despreciem forces longitudinals i no fem l'aproximacio rara dels angles (respecte al pdf)

        n4 = max(-self.x_std*self.n_bound, min(self.x_std*0.66*(randn()), self.x_std*self.n_bound))
        n4 = 0

        self.x      += self.dt*(cos(yaw)*vx - sin(yaw)*vy + n4)

        n5 = max(-self.y_std*self.n_bound, min(self.y_std*0.66*(randn()), self.y_std*self.n_bound))
        n5 = 0

        self.y      += self.dt*(sin(yaw)*vx + cos(yaw)*vy + n5)

        n1 = max(-self.vx_std*self.n_bound, min(self.vx_std*0.66*(randn()), self.vx_std*self.n_bound))
        n1 = 0

        self.vx     += self.dt*(ax + psiDot*vy + n1 - dist) 

        n2 = max(-self.vy_std*self.n_bound, min(self.vy_std*0.66*(randn()), self.vy_std*self.n_bound))
        n2 = 0

        self.vy     += self.dt*(ay - psiDot*vx + n2)

        # self.ax    = u[0]*cos(u[1]) - self.mu*vx - FyF/self.m*sin(u[1])  # front driven vehicle
        # self.ay    = u[0]*sin(u[1]) + 1.0/self.m*(FyF*cos(u[1])+FyR)
        self.ax      = u[0] - FyF/self.m*sin(u[1])  # front driven vehicle
        self.ay      = 1.0/self.m*(FyF*cos(u[1])+FyR)

        #n3 = 0.66*(randn());
        n3 = max(-self.psi_std*self.n_bound, min(self.psi_std*0.66*(randn()), self.psi_std*self.n_bound))
        n3 = 0

        self.yaw    += self.dt*(psiDot + n3)
        print ("yaw",yaw)
        n6 = max(-self.psiDot_std*self.n_bound, min(self.psiDot_std*0.66*(randn()), self.psiDot_std*self.n_bound))
        n6 = 0

        self.psiDot     += self.dt*((self.L_f*FyF*cos(u[1]) - self.L_r*FyR)/self.I_z + n6)




        self.vx = abs(self.vx)



    def pacejka(self,ang):
        D = self.C_f*self.m*self.g/2    # Friction force/2
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

class vehicle_dqn(gym.Env):

    def __init__(self):

        ########################################################################
        # Parameters to modify
        ########################################################################
        K_vel=100 # More value, more punishment on small velocities
        K_angle=10 # More value, more punishment on angle deviation
        w1 = 0 #weight for rewarding  if car is inside track
        # self.max_y_entrance_trk=
        # self.min_y_entrance_trk=
        # self.max_x_entrance_trk=
        # self.min_x_entrance_trk=

        ########################################################################
        # TOPIC THAT THE SIMULATION CAR NEEDS TO SEND THE ACCELERATION AND ANGLE
        ########################################################################
        self.pub_cmd = rospy.Publisher('ecu', ECU, queue_size=1)
        self.cmd_vehicle=ECU()




        self.pub_linkStates = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.link_msg = ModelState()

        self.pub_flag        = rospy.Publisher('flag', Bool, queue_size=1)
        self.rise_flag       = Bool()

        # self.link_msg.model_name = "seat_car"
        # self.link_msg.reference_frame = "world"

        # self.link_msg.pose.position.z = 0.031
        # self.link_msg.twist.linear.x = 0.0
        # self.link_msg.twist.linear.y = 0
        # self.link_msg.twist.linear.z = 0

        # self.link_msg.twist.angular.x = 0.0
        # self.link_msg.twist.angular.y = 0
        # self.link_msg.twist.angular.z = 0
        rospy.Subscriber("simulatorStates", simulatorStates, self.states_callback)
        self.sim = SimulatorClass()
        rospy.Subscriber("errorStates", errorStates, self.errstates_callback)
        self.errstate = VehicleErrorClass()

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.running_step = rospy.get_param("running_step")

        # self.sim = Simulator()
        self.track_map   = Map()
        ( self.fig, axtr, line_planning, self.line_tr, line_pred, line_SS, line_cl, line_gps_cl, self.rec,\
         rec_sim, rec_planning ) = _initializeFigure_xy(track_map)

        ########################################################################
        # Actions definition, making a vector of actions
        ########################################################################
        max_acceleration=3.0 # max m/s2
        max_angle_steering=0.249 #Absolut value
        actions=[]
        for i in np.linspace(-max_acceleration,max_acceleration,5): # 5 values of acceleration
            for j in np.linspace(-max_angle_steering,max_angle_steering,7): # / values of angle steering
                actions.append([round(i,1),round(j,3)])

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.action_space = spaces.Discrete(7)
        self.reward_range = (-np.inf, np.inf)
        self.sim.f([0,0])
        self._seed()

        self.last50actions = [0] * 50

        ########################################################################
        # Modify image size
        ########################################################################
        self.img_rows = 256
        self.img_cols = 256
        self.img_channels = 1

    def calculate_observation(self,data,last_y): # Episode done when the robot will finish the track x=0 min<y<max
        done = False
        success_inside = False
        if (self.max_x_entrance_trk> data.x > self.min_y_entrance_trk):
            if ((data.y > last_y) and (last_y <= 0)):
                rospy.logdebug("EPISODE COMPLETED")
                done = True
        now_y = rospy.wait_for_message("simulatorStates", simulatorStates, timeout=1).y
        is_insidemap = rospy.wait_for_message("errorStates", errorStates, timeout=1).insideMap
        ey = rospy.wait_for_message("errorStates", errorStates, timeout=1).ey
        epsi = rospy.wait_for_message("errorStates", errorStates, timeout=1).epsi

        if is_insidemap==1:
            success_inside=True


        return [done,now_y, success_inside,ey,epsi]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        

        steering = 0
        acceleration = 0

        ########################################################################
        # Actions publishing
        ########################################################################
        if action == 0: #FORWARD
            steering      = 0
            acceleration = 1.0
        elif action == 1: #LEFT
            steering      = -0.1
            acceleration = 0.8
        
        elif action == 2: #RIGHT
            steering      = 0.1
            acceleration = 0.8

        elif action == 3: #LEFT
            steering      = -0.5
            acceleration = 0.7
        
        elif action == 4: #RIGHT
            steering      = 0.5
            acceleration = 0.7

        elif action == 5: #LEFT
            steering      = -1.0
            acceleration = 0.6
        
        elif action == 7: #RIGHT
            steering      = 1.0
            acceleration = 0.6

        elif action == 8: #LEFT
            steering      = -1.5
            acceleration = 0.5
        
        elif action == 9: #RIGHT
            steering      = 1.5
            acceleration = 0.5
        elif action == 10: #LEFT
            steering      = -2.3
            acceleration = 0.2
        
        elif action == 11: #RIGHT
            steering      = 2.3
            acceleration = 0.2


        elif action == 12: #BREAK
            steering      = 0.0
            acceleration = -0.5

        u = [acceleration,steering]
        # print "uu",u
        # u = [0.5,0.1]
        # self.sim.f(u)
        self.cmd_vehicle.servo = steering
        self.cmd_vehicle.motor = acceleration
        self.rise_flag.data  = True
        pub_flag.publish(rise_flag)

        # print "velocity",self.sim.vx
        print "action",action,"servo",steering,"motor",acceleration,"-self.sim.yaw",-self.sim.yaw
        
        sim_offset_x    = rospy.get_param("init_x")
        sim_offset_y    = rospy.get_param("init_y")
        sim_offset_yaw  = rospy.get_param("init_yaw")

        # if self.pub_linkStates.get_num_connections() > 0:
        #     self.link_msg.pose.position.x = self.sim.x + sim_offset_x
        #     self.link_msg.pose.position.y = -(self.sim.y + sim_offset_y)
        #     self.link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -self.sim.yaw))
        #     # print "sim.x",self.sim.x,"sim_offset_x",sim_offset_x
        #     # print "link_msg.pose.position.x",self.link_msg.pose.position.x
            
        #     self.gazebo.unpauseSim()
        #     self.pub_linkStates.publish(self.link_msg)

        #     rospy.wait_for_service('/gazebo/set_model_state')
        # else:
        #     print "not able to retrive simulatorStates"




        self.pub_cmd.publish(self.cmd_vehicle) # Publishing acceleration and steering angle


        
        # [done,now_y, success_inside,ey,epsi] = self.calculate_observation(pose_data_sim, now_y) # Function to check if the robot has finished the episode


        image_data = None
        cv_image = None
        while image_data is None:
            try:
                ################################################################
                # CHANGE ACCORGIND TO THE TYPE OF TOPIC THAT THE SIMULATION OUTPUTS
                ################################################################
                # OBTAINING THE IMAGE FROM THE CAR CAMERA AND CHECKING THAT WE ARE STILL ON THE TRACK
                # change topic name of shivams camera in the launch file to match with this one
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=2)
                # np_arr = np.fromstring(image_data, np.uint8)
                # # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
                # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # cv2.imshow("image",image_np)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                pass

        self.gazebo.pauseSim()

        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)


        print "self.sim.vx",self.sim.vx,"self.sim.x",self.sim.x,self.errstate.ey,"self.errstate.ey"

        ########################################################################
        # Reward function
        ########################################################################
        # deviation_angle_vehicle = deviation angle from the vector of the center line
        # CHECKING THAT THE SENSORS ARE WORKING, IF THEY DO NOT WORK, PASS
        # pose_data_sim = None # WTF change data for vel_data or angle_data
        # while pose_data_sim is None:
        #     try:
        #         ################################################################
        #         # SIMULATION OUTPUTS-> GYROSCOPE VALUES AND VELOCITY VALUES
        #         ################################################################
        #         pose_data_sim = rospy.wait_for_message("simulatorStates", simulatorStates, timeout=5.0)
        #         error_data_sim = rospy.wait_for_message("errorStates", errorStates, timeout=5.0)
        #         vel_vehicle = pose_data_sim.vx
        #         print "vel_vehicle",vel_vehicle
        #         deviation_angle_vehicle = error_data_sim.ey
        #         print "deviation_angle_vehicle",deviation_angle_vehicle
        #         inside = error_data_sim.insideMap
        #         print "inside",inside

        #     except:
        #         print "Failed"
        #         pass
        # elf.EstimatedData = [data.vx, data.vy, data.psiDot, data.x, data.y, data.psi]
        vel_vehicle = self.sim.vx
        s, ey, epsi, insideMap = self.track_map.getLocalPosition(self.sim.x, self.sim.y, self.sim.yaw)

        self.line_tr.set_data(self.sim.x, -self.sim.y)
        l = 0.4; w = 0.2 #legth and width of the car
        car_x, car_y = getCarPosition(self.sim.x, -self.sim.y, self.sim.yaw, w, l)
        # print ("car_x,car_y",car_x,car_y)
        self.rec.set_xy(np.array([car_x, car_y]).T) 
        if insideMap == 1:
            self.fig.canvas.draw()     
        # plt.show()
        # plt.pause(2)

        self.rise_flag.data  = True
        pub_flag.publish(rise_flag)

        deviation_angle_vehicle = ey
        inside = insideMap

        done =False
        K_vel=100 # More value, more punishment on small velocities
        K_angle=10 # More value, more punishment on angle deviation
        w1 = 0 
        # if inside:
        #     if  5 > vel_vehicle >= 0: # 1m/s
        #         reward= -K_vel/(np.absolute(vel_vehicle)) - np.power(K_angle*deviation_angle_vehicle,2)
        # else:
        #     reward = -10 

        reward = reward_fun(self.sim.vx,self.sim.yaw,steering,acceleration)
        print "ey radian",ey,"degree",(ey*180.0)/3.14,"yaw",self.sim.yaw
        # if abs(ey)>(3.14/12):
        #     print "rotated too much"
        #     time.sleep(2)
        #     reward = reward - 1

        if (self.sim.slip_angle == True) or inside!=1:# or abs(ey)>(3.14/12):

            # if abs(ey)>(30/180*pi):
            #     print "rotated too much"
            #     reward = reward - 10

            if (self.sim.slip_angle == True):
                print "large slip angle"
                reward = reward - 5

            if inside!=1:
                print "outside the map"
                reward = reward - 10
            # time.sleep(8)
            done = True

        # if inside!=1:
        #     if inside!=1:
        #         reward = reward - 10
        #     done = True

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        print "\nREWARD::",reward
        # print "done",done
        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state, reward, done, {}


    def reset(self):

        self.last50actions = [0] * 50 #used for looping avoidance

        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()

                # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        # reset vehicle simulator
        self.sim = Simulator()


        # 4th: takes an observation of the initial condition of the robot
        
        image_data = None
        cv_image = None
        while image_data is None:
            try:
                ################################################################
                # CHANGE ACCORGIND TO THE TYPE OF TOPIC THAT THE SIMULATION OUTPUTS
                ################################################################
                # OBTAINING THE IMAGE FROM THE CAR CAMERA AND CHECKING THAT WE ARE STILL ON THE TRACK
                image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
            except:
                pass

        # 5th: pauses simulation
        self.gazebo.pauseSim()

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

    ############################################################################
    # Callback functions to obtain the velocity and angle for punishment and reward
    ############################################################################
    def states_callback(self, data):
        self.position_x=data.x
        self.position_y=data.y
        self.velocity_x=data.vx
        self.velocity_y=data.vy
        self.angle_psi=data.psi
        self.angle_psidot=data.psiDot
    def errstates_callback(self, data):
        self.err_ey=data.ey
        self.err_epsi=data.epsi
        self.err_insideMap=data.insideMap


def reward_fun(v,yaw,steer,acc ):
    vmin = 5
    vbrake = 20
    vmax = 18
    yaw_th = 0.05
    reward =0
    # min_acc =
    # max_acc = 
    # min_steer

    if v > vbrake and acc < 0:
        reward = reward + 1 
    if v > vbrake and acc > 0:
        reward = reward - 1
    if v < vmax and acc > 0:
        reward = reward + 1  
    if v < vmax and acc < 0:
        reward = reward - 1
    if yaw < -1.0*yaw_th and steer > 0: 
        reward = reward + 1
    if yaw < -1.0*yaw_th and steer < 0:
        reward = reward - 1
    if yaw > yaw_th and steer < 0:
        reward = reward + 1
    if yaw < yaw_th and steer >0:
        reward = reward - 1
    if v<vmin and acc < 0:
        reward = reward - 1
    return reward    