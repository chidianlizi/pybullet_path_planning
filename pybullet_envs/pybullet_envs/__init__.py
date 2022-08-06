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
    id='my_general-v0', 
    entry_point='pybullet_envs.envs:MyReachEnv', 
)
