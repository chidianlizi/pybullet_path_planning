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

class ReachEnv(gym.Env):
    def __init__(self, is_render=False, is_good_view=False, is_train=True):
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
        self.y_low_obs= 0
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
        self.obs_rays = np.zeros(shape=(577,),dtype=np.float32)
        obs_spaces = {
            'position': spaces.Box(low=-5.0, high=5.0, shape=(11,), dtype=np.float32), # target position, torch position and quaternion
            'rays': spaces.Box(low=0, high=2, shape=(577,),dtype=np.float32),
        } 
        self.observation_space=spaces.Dict(obs_spaces)
        

        # step counter
        self.step_counter=0
        # max steps in one episode
        self.max_steps_one_episode = 2048
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5.urdf')
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        self.distance_threshold = 0.04
        
        # # parameters of augmented targets for training
        # if self.is_train:
            
        #     self.distance_threshold = 0.2
        #     self.distance_threshold_last = 0.2
        #     self.distance_threshold_increment_p = 0.01
        #     self.distance_threshold_increment_m = 0.01
        #     self.distance_threshold_max = 0.15
        #     self.distance_threshold_min = 0.02
        # # parameters of augmented targets for testing
        # else:
        #     self.distance_threshold = 0.1
        #     self.distance_threshold_last = 0.1
        #     self.distance_threshold_increment_p = 0.0
        #     self.distance_threshold_increment_m = 0.0
        #     self.distance_threshold_max = 0.1
        #     self.distance_threshold_min = 0.1
        
        # self.episode_counter = 0
        # self.episode_interval = 50
        # self.success_counter = 0
    
    def _set_home(self):
        home = [0.0, np.random.uniform(5*np.pi/12, 7*np.pi/12),
                        np.random.uniform(-np.pi/4, -np.pi/12),
                        np.random.uniform(-3*np.pi/4, -np.pi/2),
                        np.random.uniform(-np.pi/4, -np.pi/12),
                        np.random.uniform(5*np.pi/12, 7*np.pi/12),0.0]
        return home
    def create_visual_box(self, halfExtents):
        visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5,0.5,0.5,1])
        return visual_id
    def create_collision_box(self, halfExtents):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        return collision_id
    def create_visual_sphere(self, radius):
        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[0.5,0.5,0.5,1])
        return visual_id
    def create_collision_sphere(self, radius):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        return collision_id    
    
    def add_obstacles(self):
        rand = np.float32(np.random.rand(3,))
        target_x = self.x_low_obs+rand[0]*(self.x_high_obs-self.x_low_obs)
        target_y = self.y_low_obs+rand[1]*(self.y_high_obs-self.y_low_obs)
        target_z = self.z_low_obs+rand[2]*(self.z_high_obs-self.z_low_obs)
        target_position = [target_x, target_y, target_z]
        # print (target_position)
        show_target(target_position)
        obsts = []
        rand = np.float32(np.random.rand(3,))
        obst_x = self.x_high_obs/2-rand[0]*(self.x_high_obs/2-self.x_low_obs/2)
        obst_y = 0.05+self.y_high_obs-rand[1]*(self.y_high_obs-self.y_low_obs)
        obst_z = self.z_high_obs
        obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self.create_visual_box([0.12,0.1,0.003]),
                        baseCollisionShapeIndex=self.create_collision_box([0.12,0.1,0.003]),
                        basePosition=[obst_x, obst_y, obst_z]
                    )
        obsts.append(obst_id)
        for i in range(4):
            obst_position = target_position
            val = False
            type = np.random.random()
            rate = np.random.random()
            if (rate > 0.33) and (type > 0.15):
                while not val:
                    rand = np.float32(np.random.rand(3,))
                    obst_x = self.x_high_obs-rand[0]*(self.x_high_obs-self.x_low_obs)
                    obst_y = 0.05+self.y_high_obs-rand[1]*0.8*(self.y_high_obs-self.y_low_obs)
                    obst_z = self.z_low_obs+(rand[2])*0.6*(self.z_high_obs-self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position)-np.asarray(obst_position))
                    val = (diff>0.05).all() and (np.linalg.norm(diff)<0.4)
                halfExtents = list(np.float32(np.random.uniform(0.8,1.2)*np.array([0.12,0.12,0.01])))
                obst_orientation = [[0.707, 0, 0, 0.707], [0, 0.707, 0, 0.707], [0, 0, 0.707, 0.707]]
                obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self.create_visual_box(halfExtents),
                        baseCollisionShapeIndex=self.create_collision_box(halfExtents),
                        basePosition=obst_position,
                        baseOrientation=choice(obst_orientation)
                    )
                obsts.append(obst_id)
            if (rate > 0.33) and (type <= 0.15):
                while not val:
                    rand = np.float32(np.random.rand(3,))
                    obst_x = self.x_high_obs-rand[0]*(self.x_high_obs-self.x_low_obs)
                    obst_y = self.y_high_obs-(rand[1]/2)*(self.y_high_obs-self.y_low_obs)
                    obst_z = self.z_high_obs-rand[2]*(self.z_high_obs-self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position)-np.asarray(obst_position))
                    val = (diff>0.05).all()
                radius = np.float32(np.random.uniform(0.08,0.1))                
                obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self.create_visual_sphere(radius),
                        baseCollisionShapeIndex=self.create_collision_sphere(radius),
                        basePosition=obst_position,
                    )
                obsts.append(obst_id)
        return target_position, obsts

                     
    def reset(self):
        p.resetSimulation()
        # print(time.time())

        self.target_position, self.obsts = self.add_obstacles()
        
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
        self.home = self._set_home()
        for i in range(self.base_link, self.effector_link):
            p.resetJointState(bodyUniqueId=self.RobotUid,
                                    jointIndex=i,
                                    targetValue=self.home[i],
                                    )

        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.current_joint_position = [0]
        # get lidar observation
        lidar_results = self._set_lidar()
        for i, ray in enumerate(lidar_results):
            self.obs_rays[i] = ray[2]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        # do this step in pybullet
        p.stepSimulation()
        
        # input("Press ENTER")

        return self._get_obs()
    
    def step(self,action):
        # print (action)
        # set a coefficient to prevent the action from being too large
        self.action = action
        dv = 0.01
        vel = np.zeros((7,))
        vel[1:] = action * dv
        
        # get current pos
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])
        
        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        
        # calculate the new pose
        new_robots_pos = self.current_joint_position + vel

        for i in range(self.base_link, self.effector_link):
            p.resetJointState(bodyUniqueId=self.RobotUid,
                                    jointIndex=i,
                                    targetValue=new_robots_pos[i],
                                    )

        # update current pose
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])
        # get lidar observation
        lidar_results = self._set_lidar()
        for i, ray in enumerate(lidar_results):
            self.obs_rays[i] = ray[2]
        # print (self.obs_rays)
        
        # check collision
        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])        
            if len(contacts)>0:
                self.collided = True
        
        
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

        # be punished when collided
        if self.collided:
            r2 = -200
        else:
            r2 = 0
        
        reward = 2000*r1+r2

        
        # success
        is_success = False
        if self.distance<self.distance_threshold:
            self.terminated=True
            is_success = True
        elif self.step_counter>self.max_steps_one_episode:
            self.terminated=True
        elif self.collided:
            self.terminated=True
        # this episode goes on
        else:
            self.terminated=False

        info={'step':self.step_counter,
              'distance':self.distance,
              'terminated':self.terminated,
              'reward':reward,
              'collided': self.collided,
              'is_success': is_success}
        
        # if self.terminated: 
            # print(info)
            # logger.debug(info)
        
        return self._get_obs(),reward,self.terminated,info
    
    def _get_obs(self):
        self.state[0:6] = self.current_joint_position[1:]
        self.state[6:9] = np.asarray(self.current_pos)-np.asarray(self.target_position)
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos))-np.asarray(self.target_position), ord=None)
        self.state[9] = self.distance
        self.state[10] = float(self.collided)
        # print (self.distance)
        return{
            'position': self.state,
            'rays': self.obs_rays
        }
    
    def _set_lidar(self, ray_length=2, ray_num_hor=72, render=False):
        ray_froms = []
        ray_tops = []
        frame = quaternion_matrix(self.current_orn)
        frame[0:3,3] = self.current_pos
        ray_froms.append(list(self.current_pos))
        ray_tops.append(np.matmul(np.asarray(frame),np.array([0.0,0.0,ray_length,1]).T)[0:3].tolist())

        # set the angle of rays
        for angle in range(254, 270, 2):
            for i in range(ray_num_hor):
                z = -ray_length * math.sin(angle*np.pi/180)
                l = ray_length * math.cos(angle*np.pi/180)
                x_end = l*math.cos(2*math.pi*float(i)/ray_num_hor)
                y_end = l*math.sin(2*math.pi*float(i)/ray_num_hor)
                start = list(self.current_pos)
                end = np.matmul(np.asarray(frame),np.array([x_end,y_end,z,1]).T)[0:3].tolist()
                ray_froms.append(start)
                ray_tops.append(end)


        results = p.rayTestBatch(ray_froms, ray_tops)

       
        if render:
            hitRayColor = [0, 1, 0]
            missRayColor = [1, 0, 0]

            p.removeAllUserDebugItems()

            for index, result in enumerate(results):
                if result[0] == -1:
                    p.addUserDebugLine(ray_froms[index], ray_tops[index], missRayColor)
                else:
                    p.addUserDebugLine(ray_froms[index], ray_tops[index], hitRayColor)
        return results

    
if __name__ == '__main__':
    
    env = ReachEnv(is_render=True, is_good_view=False)
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


    
    


