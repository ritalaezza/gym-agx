import os
import logging
import logging.config
import logging.handlers
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
        except Exception as e:
            print("Problem setting log directory: {}".format(e))
        finally:
            os.chdir(current_dir)
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
    print("Registered {}".format('BendWire{}-v0'.format(suffix)))


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
        'n_substeps': 2,
    }

    register(
        id='BendWireObstacle{}-v0'.format(suffix),
        entry_point='gym_agx.envs:BendWireObstacleEnv',
        kwargs=kwargs,
        max_episode_steps=int(1500),
    )
    print("Registered {}".format('BendWireObstacle{}-v0'.format(suffix)))


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
        'n_substeps': 20,
    }

    register(
        id='PushRope{}-v0'.format(suffix),
        entry_point='gym_agx.envs:PushRopeEnv',
        kwargs=kwargs,
        max_episode_steps=int(3000),
    )
    print("Registered {}".format('PushRope{}-v0'.format(suffix)))


for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
        'n_substeps': 20,
    }

    register(
        id='InsertORing{}-v0'.format(suffix),
        entry_point='gym_agx.envs:InsertORingEnv',
        kwargs=kwargs,
        max_episode_steps=int(1500),
    )
    print("Registered {}".format('InsertORing{}-v0'.format(suffix)))
