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
from sensor_msgs.msg import CompressedImage
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
from simulator import simulator
from vehicle_error import *
from vehicle_error import _initializeFigure_xy


###############################################################################
#REGISTERING THE ENVIRONMENT, first arg in entry point name file, second class name
###############################################################################
reg = register(
    id='VehicleDDPG-v0',
    entry_point='env_vehicle_ddpg:vehicle_ddpg',
    max_episode_steps=1000
    )


class vehicle_ddpg(gym.Env):

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
        # self.pub_cmd = rospy.Publisher('ecu', ECU, queue_size=1)
        # self.cmd_vehicle=ECU()



        # main(sys.argv)
        self.pub_linkStates = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
        self.link_msg = ModelState()
        self.displacement = 0
        # self.pub_flag        = rospy.Publisher('flag', Bool, queue_size=1)
        # self.rise_flag       = Bool()

        self.link_msg.model_name = "seat_car"
        self.link_msg.reference_frame = "world"

        self.link_msg.pose.position.z = 0.031
        self.link_msg.twist.linear.x = 0.0
        self.link_msg.twist.linear.y = 0
        self.link_msg.twist.linear.z = 0

        self.link_msg.twist.angular.x = 0.0
        self.link_msg.twist.angular.y = 0
        self.link_msg.twist.angular.z = 0
        # rospy.Subscriber("simulatorStates", simulatorStates, self.states_callback)
        # self.sim = SimulatorClass()
        # rospy.Subscriber("errorStates", errorStates, self.errstates_callback)
        # self.errstate = VehicleErrorClass()

        self.x_next = 0
        self.y_next = 0
        self.psi_next =0 
        self.v_next = 0 

        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        self.running_step = rospy.get_param("running_step")


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # self.sim = simulator()
        self.track_map   = Map()
        ( self.fig, axtr, line_planning, self.line_tr, line_pred, line_SS, line_cl, line_gps_cl, self.rec,\
         rec_sim, rec_planning ) = _initializeFigure_xy(self.track_map)

        ########################################################################
        # Actions definition, making a vector of actions
        ########################################################################
        self.max_acceleration=2.0 # max m/s2
        self.max_angle_steering=0.249 #Absolut value
        self.actions=[]
        for i in np.linspace(-0.5,self.max_acceleration,5): # 5 values of acceleration
            for j in np.linspace(-self.max_angle_steering,max_angle_steering,5): # / values of angle steering
                self.actions.append([round(i,1),round(j,3)])

        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        # self.action_space = spaces.Discrete(70)
        self.reward_range = (-np.inf, np.inf)
        # self.sim.f([0,0])
        self._seed()

        self.last50actions = [0] * 50

        ########################################################################
        # Modify image size
        ########################################################################
        self.img_rows = 256
        self.img_cols = 256
        self.img_channels = 1

        self.observation_space = spaces.Box(low=0, high=255, shape=(self.img_rows, self.img_cols, 1), dtype=np.uint8)

        self.stepNumber = 0
        self.dpathcovered = 0
        # self.gameScore = 0
        self.episodeStepNumber = 0

        # self.gazebo_image = rospy.Subscriber("/camera/rgb/image_raw/compressed",CompressedImage, self.callback,  queue_size = 1)
        self.gazebo_image = rospy.Subscriber("/camera/rgb/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:


    def calculate_observation(self): # Episode done when the robot will finish the track x=0 min<y<max
        image_data = None
        while image_data is None:
            # try:
            image_data = rospy.wait_for_message('/camera/rgb/image_raw/compressed',CompressedImage, timeout=5)
            np_arr = np.fromstring(image_data.data, np.uint8)
                # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

            # image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
            # h = image_data.height
            # w = image_data.width
            # image_np = CvBridge().imgmsg_to_cv2(image_data, "bgr8")

            # except:
            #     rospy.loginfo("Current image pose not ready yet, retrying for getting image ")

        return image_np

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_info(self):
        return {"x": self.x_next,
                "y": self.y_next,
                "yaw": self.psi_next}

    def step(self, action):
        
        reward = 0
        # steering = 0
        # acceleration = 0

        # ########################################################################
        # # Actions publishing
        # ########################################################################
        # if action == 0: #FORWARD
        #     steering      = 0
        #     acceleration = 2.0
        # elif action == 1: #LEFT
        #     steering      = -0.02
        #     acceleration = 1.5
        
        # elif action == 2: #RIGHT
        #     steering      = 0.02
        #     acceleration = 1.5

        # elif action == 3: #LEFT
        #     steering      = -0.05
        #     acceleration = 1.2
        
        # elif action == 4: #RIGHT
        #     steering      = 0.05
        #     acceleration = 1.2

        # elif action == 5: #LEFT
        #     steering      = -0.10
        #     acceleration = 1.0
        
        # elif action == 7: #RIGHT
        #     steering      = 0.10
        #     acceleration = 1.0

        # elif action == 8: #LEFT
        #     steering      = -0.15
        #     acceleration = 0.8
        
        # elif action == 9: #RIGHT
        #     steering      = 0.15
        #     acceleration = 0.8

        # elif action == 10: #LEFT
        #     steering      = -0.2
        #     acceleration = 0.5
        
        # elif action == 11: #RIGHT
        #     steering      = 0.2
        #     acceleration = 0.5


        # elif action == 12: #BREAK
        #     steering      = 0.0
        #     acceleration  =  0.0



        u = self.actions[action]
        print "Actions >>" , u
        # u = np.array([acceleration,steering])
        z = [self.x_next,self.y_next,self.psi_next,self.v_next]
        [self.x_next,self.y_next,self.psi_next,self.v_next] = kin_sim(z,u,0.1)
        print "self.x_next,self.y_next,self.psi_next,self.v_next",self.x_next,self.y_next,self.psi_next,self.v_next
        # print "uu",u
        # u = [0.5,0.1]
        # self.sim.f(u)

        # self.rise_flag.data  = True
        # pub_flag.publish(rise_flag)

        # print "velocity",self.sim.vx
        # print "action",action,"servo",steering,"motor",acceleration,"-self.sim.yaw",-self.sim.yaw
        
        sim_offset_x    = rospy.get_param("init_x")
        sim_offset_y    = rospy.get_param("init_y")
        sim_offset_yaw  = rospy.get_param("init_yaw")

        if self.pub_linkStates.get_num_connections() > 0:
            # self.link_msg.pose.position.x = self.sim.x + sim_offset_x
            # self.link_msg.pose.position.y = -(self.sim.y + sim_offset_y)
            # self.link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -self.sim.yaw))
            
            ## kinematic
            self.link_msg.pose.position.x = self.x_next + sim_offset_x
            self.link_msg.pose.position.y = -(self.y_next + sim_offset_y)
            self.link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -self.psi_next))
            
            # print "sim.x",self.sim.x,"sim_offset_x",sim_offset_x
            # print "link_msg.pose.position.x",self.link_msg.pose.position.x
            
            self.gazebo.unpauseSim()
            self.pub_linkStates.publish(self.link_msg)
            time.sleep(0.5)
            rospy.wait_for_service('/gazebo/set_model_state')
        else:
            print "not able to retrive simulatorStates"


        cam_image = self.calculate_observation()

        self.gazebo.pauseSim()

        self.last50actions.pop(0) #remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)


        vel_vehicle = self.v_next
        # s, ey, epsi, insideMap = self.track_map.getLocalPosition(self.x_next, self.y_next, self.psi_next)
        
        # # vel_vehicle = self.sim.vx
        # # s, ey, epsi, insideMap = self.track_map.getLocalPosition(self.sim.x, self.sim.y, self.sim.yaw)
        
        # self.line_tr.set_data(self.x_next, -self.y_next)
        # l = 0.4; w = 0.2 #legth and width of the car
        # car_x, car_y = getCarPosition(self.x_next, -self.y_next, self.psi_next, w, l)
        

        # # self.line_tr.set_data(self.sim.x, -self.sim.y)
        # # l = 0.4; w = 0.2 #legth and width of the car
        # # car_x, car_y = getCarPosition(self.sim.x, -self.sim.y, self.sim.yaw, w, l)
        # # print ("car_x,car_y",car_x,car_y)
        # self.rec.set_xy(np.array([car_x, car_y]).T) 
        

        l = 0.4; w = 0.2 #legth and width of the car
        self.line_tr.set_data(self.x_next, self.y_next)
        car_x, car_y = getCarPosition(self.x_next, self.y_next, self.psi_next, w, l)
        print ("car_x,car_y",car_x,car_y)
    
        s, ey, epsi, insideMap = self.track_map.getLocalPosition(self.x_next, self.y_next, self.psi_next)

        self.rec.set_xy(np.array([car_x, car_y]).T)     

        self.displacement += s
        print ("s",s, "ey",ey, "epsi",(epsi*180.0)/pi,"insideMap",insideMap,"self.displacement",self.displacement)
        self.dpathcovered = s-self.dpathcovered 
        
        # print ("s",s,"ey",ey,"epsi",epsi,"self.displacement",self.displacement)
        
        if insideMap == 1:
            self.fig.canvas.draw()     
        plt.show()
        # plt.pause(2)

        # self.rise_flag.data  = True
        # pub_flag.publish(rise_flag)

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

        # reward = reward_fun(self.sim.vx,self.sim.yaw,steering,acceleration)
        print "ey distance",ey,"epsi",(epsi*180.0)/3.14,"yaw",self.psi_next*180/pi
        # if abs(ey)>(3.14/12):
        #     print "rotated too much"
        #     time.sleep(2)
        #     reward = reward - 1

        self.stepNumber += 1
        

        reward = 0
        # reward = reward + 1.0*s/10
        angle_dev = 8 #vehicle yaw deviation with center track
        vmin = 5
        if inside!=0:
            print "inside map"
            #reward on each increment 
            if self.v_next > 0:
                if self.dpathcovered>0:
                    reward = reward + 1
                    print "\nself.dpathcovered>0",self.dpathcovered>0,"reward+1",reward
                    

            #saftey region inside track, track half width is 4.0
            if abs(ey) > 0.35:
                reward = reward - 1
                print "\nabs(ey) > 0.35:","reward-1",reward
                    
            #deviation anagle fro center track
            if abs((epsi*180.0)/pi) > angle_dev:
                reward = reward - 1
                print "\nabs((epsi*180.0)/pi)>15 >> Deviation","reward-1",reward
                
            if abs((epsi*180.0)/pi) < angle_dev:
                reward = reward + 1
                print "\nabs((epsi*180.0)/pi)<15 >> Deviation","reward+1",reward
                
            # reward if velocity is increasing
            if self.v_next > 1:
                reward = reward + int(self.v_next)
                print "\nself.v_next > 1 >> Velcoity","reward+int(self.v_next)",reward
                
            # if self.v_next < vmin:
            #     reward = reward - 0.25


            #reward if velocity is negtive or break is applied
            if self.v_next <= 0:
                reward = reward - 1
                print "\nself.v_next <= 0 >> Velcoity","<reward - 1>",reward
            
            #don't get too far from center line
            if (abs(ey) <0.25):
                reward = reward + 1
                print "\n(abs(ey) <0.25)","<reward + 1>",reward

            # penality for action, steering turn in opposite direction
            # if (epsi <0) and (u[1]<0):
            #     reward = reward - 1
            #     print "\n(epsi <0) and (u[1]<0)","<reward - 1>",reward
                
            # if (epsi <0) and (u[1]>0):
            #     reward = reward + 1
            #     print "\n(epsi <0) and (u[1]>0)","<reward + 1>",reward
                
            # if (epsi >0) and (u[1]>0):
            #     reward = reward - 1
            #     print "\n(epsi >0) and (u[1]>0)","<reward - 1>",reward
                
            # if (epsi >0) and (u[1]<0):
            #     reward = reward + 1
            #     print "\n(epsi >0) and (u[1]<0)","<reward + 1>",reward


        # if (self.sim.slip_angle == True):
        #     print "large slip angle"
        #     reward = reward - 100
        #     done = True

        #penality if it goes outside
        if inside!=1:
            print "outside the map"
            reward = reward - 100
            print "\noutside the map","<reward - 100>",reward

            # time.sleep(8)
            done = True

        # if (inside!=1 and (self.sim.slip_angle == True)):
        #     print "large slip angle and outside the map"
        #     reward = - 200
        #     # time.sleep(8)
        #     done = True


        # if inside!=1:
        #     if inside!=1:
        #         reward = reward - 10
        #     done = True

        cv_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        cv2.imshow("cv_image",cv_image)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        print "\nREWARD::",reward
        # print "done",done
        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        info = self._get_info()
        return state, reward, done, info


    def reset(self):

        self.last50actions = [0] * 50 #used for looping avoidance

        # 1st: resets the simulation to initial values
        self.gazebo.resetSim()
        self.stepNumber = 0
                # 2nd: Unpauses simulation
        # self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        # reset vehicle simulator
        # self.sim = simulator()

        sim_offset_x    = rospy.get_param("init_x")
        sim_offset_y    = rospy.get_param("init_y")
        sim_offset_yaw  = rospy.get_param("init_yaw")
        self.x_next = 0
        self.y_next = 0
        self.psi_next =0 
        self.v_next = 0 
        self.displacement = 0

        if self.pub_linkStates.get_num_connections() > 0:
            # self.link_msg.pose.position.x = self.sim.x + sim_offset_x
            # self.link_msg.pose.position.y = -(self.sim.y + sim_offset_y)
            # self.link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -self.sim.yaw))
            
            ## kinematic
            self.link_msg.pose.position.x = self.x_next + sim_offset_x
            self.link_msg.pose.position.y = -(self.y_next + sim_offset_y)
            self.link_msg.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -self.psi_next))
            
            # print "sim.x",self.sim.x,"sim_offset_x",sim_offset_x
            # print "link_msg.pose.position.x",self.link_msg.pose.position.x
            
            self.gazebo.unpauseSim()
            self.pub_linkStates.publish(self.link_msg)
            time.sleep(0.5)
            rospy.wait_for_service('/gazebo/set_model_state')
        else:
            print "not able to retrive simulatorStates"
        
        # 4th: takes an observation of the initial condition of the robot
        cam_image = self.calculate_observation()
        cv_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        
        # image_data = None
        # cv_image = None
        # while image_data is None:
        #     try:
        #         ################################################################
        #         # CHANGE ACCORGIND TO THE TYPE OF TOPIC THAT THE SIMULATION OUTPUTS
        #         ################################################################
        #         # OBTAINING THE IMAGE FROM THE CAR CAMERA AND CHECKING THAT WE ARE STILL ON THE TRACK
        #         image_data = rospy.wait_for_message('/camera/rgb/image_raw', Image, timeout=5)
        #         h = image_data.height
        #         w = image_data.width
        #         cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
        #     except:
        #         pass

        # 5th: pauses simulation
        self.gazebo.pauseSim()

        # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))

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

    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        self.gazebo_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

def kin_sim(z,u,dt):
        # get states / inputs
        x       = z[0]
        y       = z[1]
        psi     = z[2]
        v       = z[3]
        d_f     = u[1]
        a       = u[0]

        # extract parameters
        L_a = 0.125
        L_b = 0.125

        # compute slip angle
        bta         = arctan( L_a / (L_a + L_b) * tan(d_f) )

        # compute next state
        x_next      = x + dt*( v*cos(psi + bta) )
        y_next      = y + dt*( v*sin(psi + bta) )
        psi_next    = psi + dt*v/L_b*sin(bta)
        v_next      = v + dt*a

        return [x_next, y_next, psi_next, v_next]


def reward_fun(v,yaw,steer,acc ):
    vmin = 5
    vbrake = 10
    vmax = 8
    yaw_th = 0.15 #8.66 degree deviation
    reward =0
    max_acc = 1
    # min_acc =
    # max_acc = 
    # min_steer

    if v > vbrake and acc < 0:
        reward = reward + 1 
    if v > vbrake and acc > max_acc:
        reward = reward - 1
    if v < vmax and acc > 0:
        reward = reward + 1  
    if v < vmax and acc < 0:
        reward = reward - 1
    # if yaw < -1.0*yaw_th and steer > 0: 
    #     reward = reward + 1
    # if yaw < -1.0*yaw_th and steer < 0:
    #     reward = reward - 1
    # if yaw > yaw_th and steer < 0:
    #     reward = reward + 1
    # if yaw < yaw_th and steer >0:
    #     reward = reward - 1
    # if v<vmin and acc < 0:
    #     reward = reward - 1
    return reward    