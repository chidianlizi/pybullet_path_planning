import os
import sys
import gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from typing import Callable
import pybullet_envs



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

if __name__=='__main__':
     if IS_TRAIN:
          # load env
          env_id = 'general-v0'
          env = SubprocVecEnv([make_env(env_id, i) for i in range(1)])

          # env = gym.make('general-v0')
          # env = ReachEnv(is_render=False, is_good_view=False, is_train = True)

          # Stops training when the model reaches the maximum number of episodes
          callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e10, verbose=1)
          # Separate evaluation env
          # eval_env = ReachEnv(is_render=False, is_good_view=False, is_train=False)

          # # Use deterministic actions for evaluation
          # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs_2/',
          #                    log_path='./logs_2/', eval_freq=500,
          #                    deterministic=True, render=False)
          # Save a checkpoint every ? steps
          checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/ddpg_ckp_logs/',
                                             name_prefix='general')
          # Create the callback list
          callback = CallbackList([checkpoint_callback, callback_max_episodes])
          n_actions = env.action_space.shape[-1]
          action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
          model = DDPG("MultiInputPolicy", env, batch_size=128, action_noise=action_noise, verbose=1, tensorboard_log="./models/ddpg_tf_logs/")
          # model = DDPG.load('./ddpg_ckp_logs/reach_?????_steps', env=env)

          model.learn(
               total_timesteps=1e10,
               n_eval_episodes=64,
               callback=callback)
          model.save('./models/ddpg_general')
     else:
          # load env
          env = gym.make('general-v0')
          # load drl model
          model = DDPG.load('./ddpg_ckp_logs/reach_2713600_steps', env=env)

          while True:
               done = False
               obs = env.reset()
               while not done:
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
