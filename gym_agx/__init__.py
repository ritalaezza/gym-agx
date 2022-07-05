import os
import tempfile
import logging.config
import logging.handlers
from gym.envs.registration import register

TMP_DIR = tempfile.gettempdir()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # Project Root
LOG_DIR = os.path.join(ROOT_DIR, 'logging.conf')
if 'TMPDIR' not in os.environ:
    os.environ['TMPDIR'] = TMP_DIR  # To avoid cluttering your tmp/ directory set the TMPDIR environment variable

if os.path.exists(LOG_DIR):
    try:
        logging.config.fileConfig(LOG_DIR)
    except Exception as e:
        print("Problem setting log directory: {}".format(e))
else:
    logging.basicConfig(level=logging.ERROR)
    print('Failed to load logging configuration file. Using default configs.')


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
    kwargs={'n_substeps': 5},
    max_episode_steps=int(200),
)

register(
    id='RubberBand-v0',
    entry_point='gym_agx.envs:RubberBandEnv',
    kwargs={'n_substeps': 10},
    max_episode_steps=int(250),
)

register(
    id='CableClosing-v0',
    entry_point='gym_agx.envs:CableClosingEnv',
    kwargs={'n_substeps': 1},
    max_episode_steps=int(150),
)
