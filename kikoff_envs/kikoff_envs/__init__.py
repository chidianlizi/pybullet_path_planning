from gym.envs.registration import register

register(
    id='kikoff-v0', 
    entry_point='kikoff_envs.envs:LearnPoseEnv', 
    kwargs={'is_render': False, 'is_good_view': False, 'is_train': True}
)
register(
    id='kuka_kikoff-v0', 
    entry_point='kikoff_envs.envs:LearnPoseKukaEnv', 
    kwargs={'is_render': False, 'is_good_view': False, 'is_train': True}
)
