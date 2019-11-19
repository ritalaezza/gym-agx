from gym.envs.registration import register

def _merge(a, b):
    a.update(b)
    return a

for reward_type in ['sparse', 'dense']:
    suffix = '-dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    register(
        id='bend-wire{}-v0'.format(suffix),
        entry_point='gym_agx.envs:BendWireEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )
