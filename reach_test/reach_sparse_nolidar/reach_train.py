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
import pybullet_envs
CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from my_reach_env_simp import MySimpleReachEnv
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
          env = MySimpleReachEnv(is_render=False, is_good_view=False, is_train=True)
          # Stops training when the model reaches the maximum number of episodes
          callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)
          
          # Separate evaluation env
          eval_env = MySimpleReachEnv(is_render=False, is_good_view=False, is_train=False)

          # # Use deterministic actions for evaluation
          eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_reach/',
                             log_path='./models/best_reach/', eval_freq=10000,
                             deterministic=True, render=False)
          
          # Save a checkpoint every ? steps
          checkpoint_callback = CheckpointCallback(save_freq=51200, save_path='./models/reach_ppo_ckp_logs/',
                                             name_prefix='reach')
          # Create the callback list
          callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])
          model = PPO("MlpPolicy", env, batch_size=128, verbose=1, tensorboard_log="./models/reach_ppo_tf_logs/")
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
