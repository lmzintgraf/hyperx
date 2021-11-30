from gym.envs.registration import register


# Mujoco
# ----------------------------------------

register(
    'AntGoal-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'AntGoalSparse-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalSparseEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
            'max_episode_steps': 200},
    max_episode_steps=200
)

register(
    'HalfCheetahDirSparse-v0',
    entry_point='environments.wrappers:mujoco_wrapper',
    kwargs={
        'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirSparseEnv',
        'sparse_dist': 5.0,
        'max_episode_steps': 200,
    },
    max_episode_steps=200,
)

# Navigation
# ----------------------------------------

register(
    'SparsePointEnv-v0',
    entry_point='environments.navigation.point_robot:SparsePointEnv',
    kwargs={'goal_radius': 0.2,
            'max_episode_steps': 100},
    max_episode_steps=100,
)

# Multi-Stage GridWorld Rooms
register(
    'RoomNavi-v0',
    entry_point='environments.navigation.rooms:RoomNavi',
    kwargs={'num_cells': 3, 'corridor_len': 3, 'num_steps': 50},
)

# Mountain Treasure
register(
    'TreasureHunt-v0',
    entry_point='environments.navigation.treasure_hunt:TreasureHunt',
    kwargs={'max_episode_steps': 100,
            'mountain_height': 1,
            'treasure_reward': 10,
            'timestep_penalty': -5,
            },
)

