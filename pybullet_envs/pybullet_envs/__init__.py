from gym.envs.registration import register

register(
    id='general-v0', 
    entry_point='pybullet_envs.envs:ReachEnv', 
)
register(
    id='static-v0', 
    entry_point='pybullet_envs.envs:StaticReachEnv', 
)
register(
    id='reach-v0', 
    entry_point='pybullet_envs.envs:ReachWithoutObstaclesEnv', 
)
register(
    id='my_general-v0', 
    entry_point='pybullet_envs.envs:MyReachEnv', 
)
register(
    id='pcl-v0', 
    entry_point='pybullet_envs.envs:ReachWithPCLEnv', 
)
