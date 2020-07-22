import os
import logging.config
import logging.handlers
from gym.envs.registration import register

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root
LOG_DIR = os.path.join(ROOT_DIR, 'logging.conf')
if os.path.exists(LOG_DIR):
    try:
        # Quick directory fix: change working directory to Project root before setting log configuration
        current_dir = os.getcwd()
        os.chdir(ROOT_DIR)
        logging.config.fileConfig(LOG_DIR)
    except Exception as e:
        print("Problem setting log directory: {}".format(e))
    finally:
        os.chdir(current_dir)
else:
    logging.basicConfig(level=logging.DEBUG)
    print('Failed to load logging configuration file. Using default configs.')

HEBBE_LOG_DIR = os.path.join(ROOT_DIR, 'logging_hebbe.conf')
TMP_DIR = os.getenv('TMPDIR', default=None)
if TMP_DIR is not None:
    try:
        logging.config.fileConfig(HEBBE_LOG_DIR)
    except Exception as e:
        print("Problem setting hebbe log directory: {}".format(e))


def _merge(a, b):
    a.update(b)
    return a


# This format is true today, but it's *not* an official spec.
# [username/](env-name)-v(version) env-name is group 1, version is group 2
register(
    id='BendWire-v0',
    entry_point='gym_agx.envs:BendWireEnv',
    kwargs={'n_substeps': 2},
    max_episode_steps=int(1000),
)

register(
    id='BendWireObstacle-v0',
    entry_point='gym_agx.envs:BendWireObstacleEnv',
    kwargs={'n_substeps': 2},
    max_episode_steps=int(2000),
)

register(
    id='PushRope-v0',
    entry_point='gym_agx.envs:PushRopeEnv',
    kwargs={'n_substeps': 20},
    max_episode_steps=int(3000),
)

register(
    id='InsertORing-v0',
    entry_point='gym_agx.envs:InsertORingEnv',
    kwargs={'n_substeps': 20},
    max_episode_steps=int(2000),
)

register(
    id='PegInHole-v0',
    entry_point='gym_agx.envs:PegInHoleEnv',
    kwargs={'n_substeps': 2},
    max_episode_steps=int(200),
)

register(
    id='RubberBand-v0',
    entry_point='gym_agx.envs:RubberBandEnv',
    kwargs={'n_substeps': 5},
    max_episode_steps=int(1000),
)

register(
    id='ClipClosing-v0',
    entry_point='gym_agx.envs:ClipClosingEnv',
    kwargs={'n_substeps': 5},
    max_episode_steps=int(250),
)
