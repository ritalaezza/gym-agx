import os
import logging.config
from gym.envs.registration import register

# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2

PATH = 'logging.conf'
if os.path.exists(PATH):
    with open(PATH, 'rt') as f:
        try:
            logging.config.fileConfig(PATH)
        except Exception as e:
            print(e)
            print('Error in Logging Configuration. Using default configs')
else:
    logging.basicConfig(level=logging.DEBUG)
    print('Failed to load configuration file. Using default configs')


def _merge(a, b):
    a.update(b)
    return a


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
        'n_substeps': 2,
    }

    register(
        id='BendWire{}-v0'.format(suffix),
        entry_point='gym_agx.envs:BendWireEnv',
        kwargs=kwargs,
        max_episode_steps=int(1000),
    )
