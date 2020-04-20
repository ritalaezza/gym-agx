import os
import logging.config
from gym.envs.registration import register

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root

LOG_DIR = os.path.join(ROOT_DIR, 'logging.conf')
if os.path.exists(LOG_DIR):
    with open(LOG_DIR, 'rt') as f:
        try:
            # Quick fix: change working directory to Project root before setting log configuration
            current_dir = os.getcwd()
            os.chdir(ROOT_DIR)
            logging.config.fileConfig(LOG_DIR)
            os.chdir(current_dir)
        except Exception as e:
            print(e)
else:
    logging.basicConfig(level=logging.DEBUG)
    print('Failed to load configuration file. Using default configs.')


def _merge(a, b):
    a.update(b)
    return a


# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version)    env-name is group 1, version is group 2
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
