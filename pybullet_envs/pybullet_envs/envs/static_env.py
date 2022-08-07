import sys
import os
import numpy as np
import pybullet as p
import gym
from gym import spaces
import time
import math
from random import choice
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH)) 
ROOT = os.path.dirname(BASE) 
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from pybullet_util import go_to_target, getinversePoisition
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

class StaticReachEnv(gym.Env):
    def __init__(self, is_render=True, is_good_view=True, is_train=True):
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
        self.x_low_obs=-0.3
        self.x_high_obs=0.3
        self.y_low_obs=0.1
        self.y_high_obs=0.3
        self.z_low_obs=0.1
        self.z_high_obs=0.7

        # action sapce
        self.action = None
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(6,), dtype=np.float32) # angular velocities
        
        # parameters for spatial infomation

        self.home = [0, np.pi/2, -np.pi/6, -2*np.pi/3, -np.pi/6, np.pi/2, 0.0]
        self.target = None
        self.target_position = None
        self.current_pos = None
        self.current_orn = None
        self.current_joint_position = None
        self.current_vel = np.zeros((6,))

        
        # observation space
        self.state = np.zeros((11,), dtype=np.float32)
        self.observation_space=spaces.Box(low=-5.0, high=5.0, shape=(11,), dtype=np.float32)
        
        # time step
        self.dt = 0.01
        # step counter
        self.step_counter=0
        # max steps in one episode
        self.max_steps_one_episode = 1000
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5.urdf')
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        self.distance_threshold = 0.04        
        
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
    
    def build_shelf(self):
        obsts = []
        for i in np.linspace(self.z_low_obs, self.z_high_obs, 4):
            obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self.create_visual_box([(self.x_high_obs-self.x_low_obs)/2,
                                                                (self.y_high_obs-self.y_low_obs)/2,
                                                                0.03]),
                    baseCollisionShapeIndex=self.create_collision_box([(self.x_high_obs-self.x_low_obs)/2,
                                                                (self.y_high_obs-self.y_low_obs)/2,
                                                                0.03]),
                    basePosition=[0, 0.5*(self.y_high_obs+self.y_low_obs), i]
                )
            obsts.append(obst_id)
        for i in np.linspace(self.x_low_obs, self.x_high_obs, 4):
            obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self.create_visual_box([0.03,
                                                                (self.y_high_obs-self.y_low_obs)/2,
                                                                (self.z_high_obs-self.z_low_obs)/2]),
                    baseCollisionShapeIndex=self.create_collision_box([0.03,
                                                                (self.y_high_obs-self.y_low_obs)/2,
                                                                (self.z_high_obs-self.z_low_obs)/2]),
                    basePosition=[i, 0.5*(self.y_high_obs+self.y_low_obs), 0.5*(self.z_high_obs+self.z_low_obs)]
                )
            obsts.append(obst_id)        
   
        obst_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=self.create_visual_box([(self.x_high_obs-self.x_low_obs)/2,
                                                             0.03,
                                                             (self.z_high_obs-self.z_low_obs)/2]),
                baseCollisionShapeIndex=self.create_collision_box([(self.x_high_obs-self.x_low_obs)/2,
                                                             0.03,
                                                             (self.z_high_obs-self.z_low_obs)/2]),
                basePosition=[0, self.y_high_obs, 0.5*(self.z_high_obs+self.z_low_obs)]
            )
        obsts.append(obst_id)
        return obsts
       
            
    def set_target(self):
        targets = []
        for i, x in enumerate([-0.27,-0.13,-0.07,0.07,0.13,0.27]):
            for j, z in enumerate([0.13,0.27,0.33,0.47,0.53,0.67]):
                pose = [x, 0.16, z]
                if i%2 == 0 and j%2 == 0:
                    orn = [3*np.pi/4, 0, -3*np.pi/4]
                if i%2 == 0 and j%2 == 1:
                    orn = [-np.pi/4, -np.pi/4, 0]
                if i%2 == 1 and j%2 == 0:
                    orn = [3*np.pi/4, 0, 3*np.pi/4]
                if i%2 == 1 and j%2 == 1:
                    orn = [-np.pi/4, np.pi/4, 0]
                targets.append({'position': pose, 'orientation': orn})
                    
        target = choice(targets)
        # print (target)
        show_target(target['position'])
        
        return target   
        
        # region = np.random.randint(0,3)
        # # xoz plane
        # if region == 0:            
        #     x_val = False
        #     while not x_val:
        #         x = np.random.uniform(self.x_low_obs+0.03, self.x_high_obs-0.03)
        #         if not (-0.13<x<-0.07 or 0.07<x<0.13):
        #             x_val = True
        #     z_val = False
        #     while not z_val:
        #         z = np.random.uniform(self.z_low_obs+0.03, self.z_high_obs-0.03)
        #         if not (0.27<z<0.33 or 0.47<z<0.53):
        #             z_val = True
        #     target = [x,0.27,z]
        # # xoy plane
        # if region == 1:
        #     x_val = False
        #     while not x_val:
        #         x = np.random.uniform(self.x_low_obs+0.03, self.x_high_obs-0.03)
        #         if not (-0.13<x<-0.07 or 0.07<x<0.13):
        #             x_val = True
        #     y = np.random.uniform(0.1, 0.27)
        #     z = choice([0.13,0.27,0.33,0.47,0.53,0.67])
        #     target = [x,y,z]
        # # yoz plane
        # if region == 2:
        #     x = choice([-0.27,-0.13,-0.07,0.07,0.13,0.27])
        #     y = np.random.uniform(0.1, 0.27)
        #     z_val = False
        #     while not z_val:
        #         z = np.random.uniform(self.z_low_obs+0.03, self.z_high_obs-0.03)
        #         if not (0.27<z<0.33 or 0.47<z<0.53):
        #             z_val = True
        #     target = [x,y,z]
        # show_target(target)
        
        return target
                   
    def reset(self):
        p.resetSimulation()
        # print(time.time())
        self.obsts = self.build_shelf()
   
        # reset
        self.target = self.set_target()
        self.target_position = self.target['position']
        self.target_orientation = self.target['orientation']
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
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[-0.1,-0.22,-0.1], baseOrientation=baseorn, useFixedBase=True)
        


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

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])
        
        self.target_joint_position = getinversePoisition(self.RobotUid, self.base_link, 
                                                         self.effector_link, self.target_position, 
                                                         self.target_orientation)
        # do this step in pybullet
        p.stepSimulation()
        
        # input("Press ENTER")

        return self._get_obs()
    
    def step(self,action):
        # print (action)
        # set a coefficient to prevent the action from being too large
        self.action = action
        vel = np.zeros((7,))
        vel[1:] = action
        
        # get current pos
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])
        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        diff_position = np.zeros((7,))
        diff_position[1:] = np.asarray(self.target_joint_position) - np.asarray(self.current_joint_position[1:])

        # calculate the new pose
        self.current_vel = (np.log(1+self.step_counter)*diff_position+vel)[1:]
        new_robots_pos = self.current_joint_position+(np.log(1+self.step_counter)*diff_position+vel)*self.dt

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
        
        # check collision
        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])        
            if len(contacts)>0:
                self.collided = True
                break
               
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
        
        # be punished when high speed
        r3 = - np.linalg.norm(self.current_vel)

        reward = 2000*r1+r2+r3
        
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
        #     print(info)
        
        return self._get_obs(),reward,self.terminated,info
    
    def _get_obs(self):
        self.state[0:6] = self.current_joint_position[1:]
        self.state[6:9] = np.asarray(self.current_pos)-np.asarray(self.target_position)
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos))-np.asarray(self.target_position), ord=None)
        self.state[9] = self.distance
        self.state[10] = float(self.collided)
        # print (self.distance)
        return self.state
    
    
if __name__ == '__main__':
    
    env = StaticReachEnv(is_render=True, is_good_view=False)
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
    # RobotUid = p.loadURDF(urdf_root_path, basePosition=[-0.1,-0.22,-0.1], useFixedBase=True)
    # home = [0.0, np.pi/2, -np.pi/2, -np.pi/2, 0.0, np.pi/2, 0.0]
    # for i in range(1,7):
    #     p.resetJointState(bodyUniqueId=RobotUid,
    #                             jointIndex=i,
    #                             targetValue=home[i],
    #                             )
    # go_to_target(RobotUid, 1,7,[-0.07,0.16,0.13],[3*np.pi/4, 0, -3*np.pi/4])
    # print(p.getLinkState(RobotUid,7))
    # while True:

    #     p.stepSimulation()

# zuo shang [-np.pi/4, -np.pi/4, 0]
# you shang [-np.pi/4, np.pi/4, 0]
# you xia [3*np.pi/4, 0, 3*np.pi/4]
# zuo xia [3*np.pi/4, 0, -3*np.pi/4]


    
    


