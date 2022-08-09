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
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('is_train', type=bool, default=True, help='set true for training, false for testing')
args = parser.parse_args()

CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from reach_env import MySimpleReachEnv
# change here
IS_TRAIN = args.is_train

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
               if key == "rays":
                    extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], 128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 64)
                                        )
                    total_concat_size += 64
               elif key == "position":
                    # Run through a simple MLP
                    extractors[key] = nn.Linear(subspace.shape[0], 16)
                    total_concat_size += 16
          self.extractors = nn.ModuleDict(extractors)

          # Update the features dim manually
          self._features_dim = total_concat_size

     def forward(self, observations) -> torch.Tensor:
          encoded_tensor_list = []

          # self.extractors contain nn.Modules that do all the processing.
          for key, extractor in self.extractors.items():
               encoded_tensor_list.append(extractor(observations[key]))
          # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
          return torch.cat(encoded_tensor_list, dim=1)

if __name__=='__main__':
     if IS_TRAIN:                   
          # Separate evaluation env
          eval_env = MySimpleReachEnv(is_render=False, is_good_view=False, is_train=False)
          
          # load env
          env = MySimpleReachEnv(is_render=False, is_good_view=False, is_train=True)
          # Stops training when the model reaches the maximum number of episodes
          callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)

          # # Use deterministic actions for evaluation
          eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_reach/',
                             log_path='./models/best_reach/', eval_freq=10000,
                             deterministic=True, render=False)
          
          # Save a checkpoint every ? steps
          checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/reach_ppo_ckp_logs/',
                                             name_prefix='reach')
          # Create the callback list
          callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])
          model = PPO("MultiInputPolicy", env, batch_size=256, verbose=1, tensorboard_log='./models/reach_ppo_tf_logs/')
          # model = PPO.load('./ppo_ckp_logs/reach_?????_steps', env=env)
          model.learn(
               total_timesteps=1e10,
               n_eval_episodes=64,
               callback=callback)
          model.save('./models/reach_ppo')
     else:
          # load env
          env = MySimpleReachEnv(is_render=True, is_good_view=True, is_train=False)
          # load drl model
          model = PPO.load('./models/best_reach/best_model', env=env)

          while True:
               done = False
               obs = env.reset()
               while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
