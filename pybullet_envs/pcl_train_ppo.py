import os
import sys
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from typing import Callable
import pybullet_envs

import torch
import torch.nn as nn
import torch.nn.functional as F
CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH))
ROOT = os.path.dirname(BASE) 
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from pointnet2_utils import PointNetSetAbstraction

# change here
IS_TRAIN = True

def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == 'pc':
                self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
                self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
                self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
                self.fc1 = nn.Linear(1024, 512)
                self.bn1 = nn.BatchNorm1d(512)
                total_concat_size += 512
            elif key == 'position':
                self.mlp = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key in observations:
            if key == 'pc':
                observations[key] = observations[key].permute(0, 2, 1)
                B, _, _ = observations[key].shape
                l1_xyz, l1_points = self.sa1(observations[key], None)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
                x = l3_points.view(B, 1024)
                x = F.relu(self.bn1(self.fc1(x)))                
                encoded_tensor_list.append(x)
            else:
                x = self.mlp(observations[key])
                encoded_tensor_list.append(x)       
        # print (torch.cat(encoded_tensor_list, dim=1).shape)
        return torch.cat(encoded_tensor_list, dim=1)


if __name__=='__main__':
     if IS_TRAIN:
        # load env
        env_id = 'pcl-v0'
        env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])
        # Stops training when the model reaches the maximum number of episodes
        callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=50000, verbose=1)
        
        # Separate evaluation env
        eval_env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])

        # Use deterministic actions for evaluation
        eval_callback = EvalCallback(eval_env, best_model_save_path='./models/pcl_best_general/',
                            log_path='./models/pcl_best_general/', eval_freq=10000,
                            deterministic=True, render=False)
        
        # Save a checkpoint every ? steps
        checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/pcl_ppo_ckp_logs/',
                                            name_prefix='pcl_general')
        # Create the callback list
        callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])
        
        policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor)
        model = PPO("MultiInputPolicy", env, batch_size=128, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log='./models/pcl_ppo_tf_logs/')

        # model = PPO.load('./ppo_ckp_logs/reach_?????_steps', env=env)
        model.learn(
            total_timesteps=1e10,
            n_eval_episodes=64,
            callback=callback)
        model.save('./models/pcl_ppo_general')
     else:
          # load env
          env = gym.make('general-v0')
          # load drl model
          model = PPO.load('./ppo_reach', env=env)

          while True:
               done = False
               obs = env.reset()
               while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
