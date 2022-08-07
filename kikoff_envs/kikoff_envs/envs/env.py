#!/usr/bin/env python3

from rospkg import get_ros_paths
import rospy
import sys
import os
import random
import numpy as np
import pybullet as p
import pybullet_data as pd
import gym
from gym import spaces
import time
import math

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH)) 
ROOT = os.path.dirname(BASE) 
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from tf_util import quaternion_matrix, euler_from_quaternion
from xml_parser import parse_frame_dump, list2array
from pybullet_util import go_to_target

PATH_COMP = os.path.join(BASE, 'components')
COMP_LIST = os.listdir(PATH_COMP)

def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    angle=np.arccos(np.dot(vector_1,vector_2)/(np.linalg.norm(vector_1)*np.linalg.norm(vector_2)))
    angle_deg = np.rad2deg(angle)
    return angle, angle_deg

def random_choice_welding_spot(torch=0):
    '''
    from components list select a random component and from this component
    select one welding spot randomly
    args:
    torch: 0 is MRW510_10GH, 1 is TAND_GERAD_DD
    return:
    component: str
    welding spot with torch, position, normals, rotation matrix: array
    '''
    comp = random.choice(COMP_LIST)
    path_xml = os.path.join(PATH_COMP, comp, comp+'.xml')
    all_spots = list2array(parse_frame_dump(path_xml))[:,3:].astype(float)
    spots_with_torch0 = all_spots[(all_spots[:,0]==0.),:]
    current_spot = spots_with_torch0[np.random.choice(spots_with_torch0.shape[0], size=1, replace=False), :][0,1:19]
    return comp, current_spot

def choice_welding_spot(i, torch=0):
    comp = COMP_LIST[0]
    path_xml = os.path.join(PATH_COMP, comp, comp+'.xml')
    all_spots = list2array(parse_frame_dump(path_xml))[:,3:].astype(float)
    spots_with_torch0 = all_spots[(all_spots[:,0]==0.),:]
    current_spot = spots_with_torch0[i][1:19]
    return comp, current_spot


def show_welding_target(position):
    visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.04, rgbaColor=[1,0,0,1])
    point_id = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_id,
                basePosition=position,
                )

class LearnPoseEnv(gym.Env):
    def __init__(self, is_render=False, is_good_view=False, is_train=True):
        self.is_render=is_render
        self.is_good_view=is_good_view
        self.is_train = is_train
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs=-0.31
        self.x_high_obs=0.31
        self.y_low_obs=-0.31
        self.y_high_obs=0.31
        self.z_low_obs=-0.1
        self.z_high_obs=0.6


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64) # torch position and euler
        # observation space
        self.WIDTH = 64
        self.HEIGHT = 64
        obs_spaces = {
            'position': spaces.Box(low=-2, high=2, shape=(7,), dtype=np.float64), # target position, torch position and quaternion
            'rays': spaces.Box(low=0, high=2, shape=(73,),dtype=np.float64),
            # 'images':spaces.Box(low=0, high=1, shape=(self.WIDTH,self.HEIGHT),dtype=np.float64)
        }   
        self.obs_pos = np.zeros(shape=(7,),dtype=np.float64)
        self.obs_rays = np.zeros(shape=(73,),dtype=np.float64)
        # self.obs_img = np.zeros(shape=(self.WIDTH, self.HEIGHT),dtype=np.float64)
        self.observation_space=spaces.Dict(obs_spaces)
        
        
        self.target_position = None
        self.current_pos = None
        self.current_orn = None
        self.norm1 = None
        self.norm2 = None
        self.step_counter=0
        self.max_steps_one_episode = 512
        
        self.collided = None
        self.comp = None
        self.urdf_root_path = os.path.join(BASE, 'ur5_description/urdf/ur5_with_MRW510_10GH.urdf')
        self.base_link = 1
        self.effector_link = 7

        # training
        if self.is_train:
            
            self.distance_threshold = 0.15
            self.distance_threshold_last = 0.15
            self.distance_threshold_increment_p = 0.01
            self.distance_threshold_increment_m = 0.01
            self.distance_threshold_max = 0.15
            self.distance_threshold_min = 0.02
        # testing
        else:
            self.distance_threshold = 0.12
            self.distance_threshold_last = 0.12
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.12
            self.distance_threshold_min = 0.12
        
        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0
        
    def reset(self, component=None, spot=None):
        if self.is_train:
            #p.connect(p.GUI)
            self.comp, curr_spot = random_choice_welding_spot()
        else:
            self.comp = component
            curr_spot = spot
        comp_position = (-curr_spot[0:3]/1000).tolist()
        self.norm1 = curr_spot[3:6]
        self.norm2 = curr_spot[6:9]
        norm = self.norm1 + self.norm2
        comp_position[0] -= norm[0]*0.3
        comp_position[1] -= norm[1]*0.3
        self.target_position = [-norm[0]*0.3, -norm[1]*0.3, 0.0]
        
        
        self.step_counter = 0
        self.collided = False
        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated=False
        p.setGravity(0, 0, -9.8)

        # display boundary
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,0],
                            lineToXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,0],
                            lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,0],
                            lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_high_obs,0],
                            lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])

        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                            lineToXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs],
                            lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_low_obs,self.y_low_obs,self.z_high_obs],
                            lineToXYZ=[self.x_low_obs,self.y_high_obs,self.z_high_obs])
        p.addUserDebugLine(lineFromXYZ=[self.x_high_obs,self.y_low_obs,self.z_high_obs],
                            lineToXYZ=[self.x_high_obs,self.y_high_obs,self.z_high_obs])
        
        baseorn = p.getQuaternionFromEuler([0,np.pi,0])
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[-0.1, -0.6, 1.0], baseOrientation=baseorn, useFixedBase=True)
        # print ('current component: ',self.comp)
        self.object_id=p.loadURDF(PATH_COMP+'/'+self.comp+'/model.urdf',basePosition=comp_position,useFixedBase=True)

        go_to_target(self.RobotUid, self.base_link, self.effector_link, [0,0,0.4], [0,0,0])
        show_welding_target(self.target_position)
        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        self.obs_pos[0:3] = np.asarray(self.current_pos)-np.asarray(self.target_position)
        self.obs_pos[3:7] = self.current_orn
        # get lidar observation
        lidar_results = self._set_lidar()
        for i, ray in enumerate(lidar_results):
            self.obs_rays[i] = ray[2]
        # get depth image observation
        # rgbImg, depthImg, segImg = self._setCameraPicAndGetPic()
        # self.obs_img = depthImg
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
        
        
        p.stepSimulation()
        
        # input("Press ENTER")

        return self._get_obs()

            
    
    def step(self,action):
        dv = 0.01
        dx = action[0]*dv
        dy = action[1]*dv
        dz = action[2]*dv
        droll= action[3]*dv
        dpitch = action[4]*dv
        dyaw = action[5]*dv

        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        current_rpy = euler_from_quaternion(self.current_orn)
       # logging.debug("self.current_pos={}\n".format(self.current_pos))
        new_robot_pos=[self.current_pos[0]+dx,
                            self.current_pos[1]+dy,
                            self.current_pos[2]+dz]
        new_robot_rpy=[current_rpy[0]+droll,
                            current_rpy[1]+dpitch,
                            current_rpy[2]+dyaw]
        #logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))
        go_to_target(self.RobotUid, self.base_link, self.effector_link, new_robot_pos, new_robot_rpy)
        self.current_pos = p.getLinkState(self.RobotUid,self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid,self.effector_link)[5]
        
        # get position observation
        self.obs_pos[0:3] = np.asarray(self.current_pos)-np.asarray(self.target_position)
        self.obs_pos[3:7] = self.current_orn
        # get lidar observation
        lidar_results = self._set_lidar()
        for i, ray in enumerate(lidar_results):
            self.obs_rays[i] = ray[2]
        # print(self.obs_rays)
        # get depth image observation
        # rgbImg, depthImg, segImg = self._setCameraPicAndGetPic()
        # self.obs_img = depthImg
        
        contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.object_id)
        if (len(contacts)>0):
            self.collided = True
        p.stepSimulation()

        if self.is_good_view:
            time.sleep(0.05)
               
        self.step_counter+=1
        return self._reward()
    
    
    def _reward(self):
        # distance between torch head and target postion
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos))-np.asarray(self.target_position), ord=None)
        # print(self.distance)

        x=self.current_pos[0]
        y=self.current_pos[1]
        z=self.current_pos[2]

        
        terminated=bool(
            x<self.x_low_obs
            or x>self.x_high_obs
            or y<self.y_low_obs
            or y>self.y_high_obs
            or z<self.z_low_obs
            or z>self.z_high_obs
        )
        success = False
        # be punished when collided
        if self.collided:
            reward = -10
            self.terminated=True
        # be punished when out of range 
        elif terminated:
            reward = -2
            self.terminated=True
        # be punished when do nothing
        elif self.step_counter>self.max_steps_one_episode:
            frame = quaternion_matrix(self.current_orn)
            zvek = frame[0:3, 2]
            _, angle1 = get_angle(zvek, self.norm1)
            _, angle2 = get_angle(zvek, self.norm2)
            print(angle1, angle2)
            # be punished when the torch doesnt point to the welding spot
            if angle1<90 and angle2<90:
                reward = -self.distance
                self.terminated=True
            else:
                reward = -2
                self.terminated=True

        elif self.distance<self.distance_threshold:
            reward = 10
            self.terminated=True
            success = True
            self.success_counter += 1  
        
        else:
            reward=0
            self.terminated=False

        info={'step': self.step_counter,
              'distance': self.distance,
              'terminated': self.terminated,
              'reward': reward,
              'is_success': success}
        if self.terminated: 
            print(info)
        # print(info)
        return self._get_obs(),reward,self.terminated,info
    
    def _get_obs(self):
        return{
            'position': self.obs_pos,
            'rays': self.obs_rays,
            # 'images': self.obs_img
        }
        
    def _set_lidar(self, ray_length=5, ray_num_hor=36, render=False):
        ray_froms = []
        ray_tops = []
        frame = quaternion_matrix(self.current_orn)
        frame[0:3,3] = self.current_pos
        ray_froms.append(list(self.current_pos))
        ray_tops.append(np.matmul(np.asarray(frame),np.array([0.0,0.0,-ray_length,1]).T)[0:3].tolist())

        for i in range(ray_num_hor):
            z = -ray_length * math.sin(np.pi/3)
            l = ray_length * math.cos(np.pi/3)
            x_end = l*math.cos(2*math.pi*float(i)/ray_num_hor)
            y_end = l*math.sin(2*math.pi*float(i)/ray_num_hor)
            start = list(self.current_pos)
            end = np.matmul(np.asarray(frame),np.array([x_end,y_end,z,1]).T)[0:3].tolist()
            ray_froms.append(start)
            ray_tops.append(end)
        
        for i in range(ray_num_hor):
            z = -ray_length * math.sin(np.pi/4)
            l = ray_length * math.cos(np.pi/4)
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
    
    def _setCameraPicAndGetPic(self):

        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=[0,0,1],
            cameraTargetPosition=[0,0,0],
            cameraUpVector=[0,1,0],
            # physicsClientId=self.physicsClient
        )
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=50.0, 
            aspect=1.0,
            nearVal=0.01,
            farVal=20,
            # physicsClientId=self.physicsClient
        )

        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=self.WIDTH, height=self.HEIGHT,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            # physicsClientId=self.physicsClient
        )

        return rgbImg, depthImg, segImg
    
 
    
if __name__ == '__main__':
    env = LearnPoseEnv(is_render=True)
    episodes = 10
    for episode in range(episodes):
        state = env.reset()
        done = False
        i = 0
        while not done:   
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(info)


