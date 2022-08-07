- Register gym environments

```
pip install -e .
```

- Directly run 
```
python train_ddpg.py
```
or 
```
python train_ppo.py
```
***
- Dependency:
  - numpy
  - pybullet
  - gym
  - stable_baselines3
  - tensorflow 2
***
- Introduction:
  - env:
    - general_env: action space is velocity of joints, observation is position+rays, with obstacles
    - static_env: reproduction of that paper, action space is velocity of joints, observation is position only
    - reach_env: action space is velocity of joints, observation is position+rays, without obstacles
    - my_general_env: action space is posture of end effector, observation is position+rays, with obstacles
    - pcl_general_env: action space is posture of end effector, observation is position+pcl, with obstacles, using pn++ for training
***
- Notes:

  - The tensorboard logs are stored in the folder './ddpg_tf_logs/' or './ppo_tf_logs/', use "tensorboard --logdir ./ddpg_tf_logs"
  - Every certain number of steps the model will be stored in the folder './ddpg_ckp_logs' or './ppo_ckp_logs', so you can safely interrupt the training process.
  - To resume training, in train.ddpg.py uncomment line 61 and change the name to the latest checkpoint file name, comment out line 60, and run train.ddpg.py to continue training.
