from gym.envs.registration import register

register(
    id='agx-v0',
    entry_point='gym_agx.envs:AgxEnv',
)
register(
    id='bend-wire-v0',
    entry_point='gym_agx.envs:BendWireEnv',
)
