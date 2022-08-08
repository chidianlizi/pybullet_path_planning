import sys
import os
import numpy as np
import pybullet as p
import gym
from gym import spaces
import time
import math
import random
import string
from random import choice
import logging
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH)) 
ROOT = os.path.dirname(BASE) 
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from pybullet_util import go_to_target

# LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
# DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
# ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
# logging.basicConfig(filename='general_env_'+ran_str+'.log', 
#                     level=logging.DEBUG, 
#                     format=LOG_FORMAT, 
#                     datefmt=DATE_FORMAT)
# logger = logging.getLogger(__name__)

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.

    axes : One of 24 axis sequences as string or encoded tuple

    Note that many Euler angle triplets can describe one matrix.

    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.

    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.

    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    angle=np.arccos(np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2)))
    angle_deg = np.rad2deg(angle)
    return angle, angle_deg

def show_target(position):
    visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1,0,0,1])
    point_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=position,
                )

class MySimpleReachEnv(gym.Env):
    def __init__(self, is_render=True, is_good_view=False, is_train=False):
        '''
        is_render: start GUI
        is_good_view: slow down the motion to have a better look
        is_tarin: training or testing
        '''
        self.is_render=is_render
        self.is_good_view=is_good_view
        self.is_train = is_train
        self.DISPLAY_BOUNDARY = False
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        # init ros node
        # rospy.init_node('pybullet_env', anonymous=True)
        # set pc publisher to ros
        # self.pc_pub = rospy.Publisher("converted_pc", PointCloud2, queue_size=10)
        
        # set the area of the workspace
        self.x_low_obs=-0.4
        self.x_high_obs=0.4
        self.y_low_obs= 0.1
        self.y_high_obs=0.5
        self.z_low_obs=0
        self.z_high_obs=0.3

        # action sapce
        self.action = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32) # angular velocities
        
        # parameters for spatial infomation
        self.home = [0, np.pi/2, -np.pi/6, -2*np.pi/3, -4*np.pi/9, np.pi/2, 0.0]
        self.target_position = None
        self.current_pos = None
        self.current_orn = None
        self.current_joint_position = None
        # parameters for image observation
        # self.WIDTH = 128
        # self.HEIGHT = 128
        
        # observation space
        self.state = np.zeros((11,), dtype=np.float32)
        self.observation_space=spaces.Box(low=-2, high=2, shape=(11,), dtype=np.float32)
        

        # step counter
        self.step_counter=0
        # max steps in one episode
        self.max_steps_one_episode = 1024
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5.urdf')
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        
        # parameters of augmented targets for training
        if self.is_train:
            
            self.distance_threshold = 0.1
            self.distance_threshold_last = 0.1
            self.distance_threshold_increment_p = 0.01
            self.distance_threshold_increment_m = 0.01
            self.distance_threshold_max = 0.1
            self.distance_threshold_min = 0.02
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.1
            self.distance_threshold_last = 0.1
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.1
            self.distance_threshold_min = 0.1
        
        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0
    
    def _set_home(self):
        home = [0.0, np.random.uniform(5*np.pi/12, 7*np.pi/12),
                        np.random.uniform(-np.pi/4, -np.pi/12),
                        np.random.uniform(-3*np.pi/4, -np.pi/2),
                        np.random.uniform(-np.pi/4, -np.pi/12),
                        np.random.uniform(5*np.pi/12, 7*np.pi/12),0.0]
        return home
    def _set_target(self):
        rand = np.float32(np.random.rand(3,))
        target_x = self.x_low_obs+rand[0]*(self.x_high_obs-self.x_low_obs)
        target_y = self.y_low_obs+rand[1]*(self.y_high_obs-self.y_low_obs)
        target_z = self.z_low_obs+rand[2]*(self.z_high_obs-self.z_low_obs)
        target_position = [target_x, target_y, target_z]
        # print (target_position)
        show_target(target_position)
        
        return target_position
                     
    def reset(self):
        p.resetSimulation()
        # print(time.time())

        self.target_position = self._set_target()
        
        # reset
        self.step_counter = 0
        self.collided = False

        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated=False
        p.setGravity(0, 0, 0)

        # display boundary
        if self.DISPLAY_BOUNDARY:
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_low_obs],
                                lineToXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_low_obs],
                                lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_low_obs],
                                lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_high_obs,self.z_low_obs],
                                lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])

            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                                lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs],
                                lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                                lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs],
                                lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
            
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_low_obs],
                                lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_low_obs],
                                lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_low_obs],
                                lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_low_obs],
                                lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_low_obs])
        
        # load the robot arm
        baseorn = p.getQuaternionFromEuler([0,0,0])
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[-0.1,-0.12,0.0], baseOrientation=baseorn, useFixedBase=True)
        


        # robot goes to the initial position
        # self.home = self._set_home()
        for i in range(self.base_link, self.effector_link):
            p.resetJointState(bodyUniqueId=self.RobotUid,
                                    jointIndex=i,
                                    targetValue=self.home[i],
                                    )

        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]

        self.current_joint_position = [0]

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])


        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter/self.episode_interval
            self.success_counter = 0
            if success_rate < 0.8 and self.distance_threshold<self.distance_threshold_max:                            
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.8 and self.distance_threshold>self.distance_threshold_min:
                self.distance_threshold -= self.distance_threshold_increment_m
            elif success_rate ==1 and self.distance_threshold==self.distance_threshold_min:
                self.distance_threshold == 0.1
            else:
                self.distance_threshold = self.distance_threshold_last
            print ('current distance threshold: ', self.distance_threshold)

        # do this step in pybullet
        p.stepSimulation()
        
        # input("Press ENTER")

        return self._get_obs()
    
    def step(self,action):
        # print (action)
        # set a coefficient to prevent the action from being too large
        self.action = action
        dv = 0.005
        dx = action[0]*dv
        dy = action[1]*dv
        dz = action[2]*dv
        droll= action[3]*dv
        dpitch = action[4]*dv
        dyaw = action[5]*dv

        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        current_rpy = euler_from_quaternion(self.current_orn)
        new_robot_pos=[self.current_pos[0]+dx,
                            self.current_pos[1]+dy,
                            self.current_pos[2]+dz]
        new_robot_rpy=[current_rpy[0]+droll,
                            current_rpy[1]+dpitch,
                            current_rpy[2]+dyaw]
        go_to_target(self.RobotUid, self.base_link, self.effector_link, new_robot_pos, new_robot_rpy)

        
        # update current pose
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])
        
        # logging.debug("self.current_pos={}\n".format(self.current_pos))
 
        
        
        p.stepSimulation()
        if self.is_good_view:
            time.sleep(0.05)
               
        self.step_counter+=1
        # input("Press ENTER")
        return self._reward()
    
    
    def _reward(self):
        # distance between torch head and target postion
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos))-np.asarray(self.target_position), ord=None)
        # print(self.distance)
        dd = 0.1
        if self.distance < dd:
            r1 = -0.5*self.distance*self.distance
        else:
            r1 = -dd*(abs(self.distance)-0.5*dd)
        
        x=self.current_pos[0]
        y=self.current_pos[1]
        z=self.current_pos[2]
        out=bool(
            x<self.x_low_obs
            or x>self.x_high_obs
            or y<self.y_low_obs
            or y>self.y_high_obs
            or z<self.z_low_obs
            or z>self.z_high_obs
        )
        
        # success
        is_success = False
        if self.distance<self.distance_threshold:
            self.terminated=True
            is_success = True
            reward = 1
        elif self.step_counter>self.max_steps_one_episode:
            self.terminated=True
            if out:
                reward = -1
            else:
                reward = 0
        # this episode goes on
        else:
            self.terminated=False
            reward = 0

        info={'step':self.step_counter,
              'distance':self.distance,
              'terminated':self.terminated,
              'reward':reward,
              'is_success': is_success}
        
        if self.terminated: 
            print(info)
            # logger.debug(info)
        
        return self._get_obs(),reward,self.terminated,info
    
    def _get_obs(self):
        self.state[0:3] = self.target_position
        self.state[3:6] = self.current_pos
        self.state[6:10] = self.current_orn
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos))-np.asarray(self.target_position), ord=None)
        self.state[10] = self.distance
        return self.state
    
if __name__ == '__main__':
    
    env = MySimpleReachEnv(is_render=True, is_good_view=False)
    episodes = 100
    for episode in range(episodes):
        state = env.reset()
        done = False
        i = 0
        while not done:   
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(info)
    
    # p.connect(p.GUI)
    # p.setGravity(0, 0, 0)
    # urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5.urdf')
    # RobotUid = p.loadURDF(urdf_root_path, basePosition=[-0.1,-0.12,0.0], useFixedBase=True)
    # home = [0.0, np.pi/2, -np.pi/2, -np.pi/2, 0.0, np.pi/2, 0.0]
    # for i in range(1,7):
    #     p.resetJointState(bodyUniqueId=RobotUid,
    #                             jointIndex=i,
    #                             targetValue=home[i],
    #                             )
    # while True:
    #     go_to_target(RobotUid, 1,7,[-0.4,0.5,0.2],[])
    #     print(p.getLinkState(RobotUid,7))
    #     p.stepSimulation()


    
    


