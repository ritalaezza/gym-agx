import gym
from gym.envs.registration import register

# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2

def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='BendWire{}-v0'.format(suffix),
        entry_point='gym_agx.envs:BendWireEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
