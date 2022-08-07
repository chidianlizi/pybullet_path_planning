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

# change here
IS_TRAIN = True


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
     if IS_TRAIN:
          # load env
          env_id = 'kuka_kikoff-v0'
          env = gym.make(env_id, is_render=False, is_good_view=False, is_train=True)
          # Stops training when the model reaches the maximum number of episodes
          callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)
          # Separate evaluation env
          eval_env = gym.make(env_id, is_render=False, is_good_view=False, is_train=True)

          # Use deterministic actions for evaluation
          eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_kuka_pose/',
                              log_path='./models/best_kuka_pose/', eval_freq=10000,
                              deterministic=True, render=False)
          # Save a checkpoint every ? steps
          checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/kuka_ckp_logs/',
                                             name_prefix='kuka_pose')
          # Create the callback list
          callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])
          model = PPO(
               "MultiInputPolicy",
               env,
               # gamma=params['gamma'],
               # learning_rate=params['learning_rate'],
               # n_steps=params['n_steps'],
               # batch_size=params['batch_size'],
               # n_epochs=params['n_epochs'],
               verbose=1,
               tensorboard_log="./models/kuka_tf_logs/")
          # model = PPO.load('pose_5', env=env)
          model.learn(
               total_timesteps=params['total_timesteps'],
               n_eval_episodes=params['n_eval_episodes'],
               callback=callback)
     else:
          # load env
          env_id = 'kikoff-v0'
          env = gym.make(env_id)
          # load drl model
          model = PPO.load('./kuka/kuka_pose_1107', env=env)

          while True:
               done = False
               obs = env.reset()
               while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)