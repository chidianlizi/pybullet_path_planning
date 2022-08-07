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
from typing import Callable
import kikoff_envs

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

if __name__=='__main__':
     params = {
         'gamma': 0.99,
         'learning_rate': 0.003,
         'n_steps': 256,
         'batch_size': 4,
         'n_epochs': 4,
         'total_timesteps': 1e12,
         'n_eval_episodes': 64        
     }
     env_id = 'kikoff-v0'
     env = gym.make(env_id, is_render=False, is_good_view=False, is_train=True)
     env.env
     # env.seed(0)
     # Stops training when the model reaches the maximum number of episodes
     callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)
     # Separate evaluation env
     eval_env = gym.make(env_id, is_render=False, is_good_view=False, is_train=True)

     # Use deterministic actions for evaluation
     eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_pose/',
                         log_path='./models/best_pose/', eval_freq=10000,
                         deterministic=True, render=False)
     # Save a checkpoint every ? steps
     checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/ckp_logs/',
                                         name_prefix='pose')
     # Create the callback list
     callback = CallbackList([checkpoint_callback, callback_max_episodes])
     model = PPO(
          "MultiInputPolicy",
          env,
          # gamma=params['gamma'],
          # learning_rate=params['learning_rate'],
          # n_steps=params['n_steps'],
          # batch_size=params['batch_size'],
          # n_epochs=params['n_epochs'],
          verbose=1,
          tensorboard_log="./models/tf_logs/")
     # model = PPO.load('pose_5', env=env)
     model.learn(
          total_timesteps=params['total_timesteps'],
          n_eval_episodes=params['n_eval_episodes'],
          callback=callback)
     
     # while True:
     #      done = False
     #      obs = env.reset()
     #      while not done:
     #           action, _states = model.predict(obs)
     #           obs, rewards, done, info = env.step(action)