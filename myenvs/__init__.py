from gym.envs.registration import register

register(
    id='2d_maze-v0', 
    entry_point='myenvs.env2DMaze:Env'
)
register(
    id='2d_maze-v1', 
    entry_point='myenvs.env2DMaze:Env_v1'
)
register(
    id='2d_maze-v10', 
    entry_point='myenvs.env2DMaze:Env_v10'
)
register(
    id='2d_maze-v11', 
    entry_point='myenvs.env2DMaze:Env_v11'
)
