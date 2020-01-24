import os
import sys
import agx
import math
import logging

from gym_agx.envs import wire_env

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire_goal.agx')

logger = logging.getLogger(__name__)


class BendWireEnv(wire_env.WireEnv):
    """Subclass which inherits from Wire environment.
    """
    def __init__(self, reward_type='sparse'):
        """Initializes BendWire environment
        The radius and length should be consistent with the model defined in 'SCENE_PATH'.
        :param reward_type: either 'sparse' or 'dense'
        """
        radius = 0.01
        length = 0.1 + 2*radius
        sim_timestep = 0.01      # seconds

        observation_type = 'vector'
        env_timestep = (1 / 60)  # seconds
        n_substeps = int(env_timestep / sim_timestep)
        camera = {
            'eye': agx.Vec3(length / 2, -5 * length, 0),
            'center':  agx.Vec3(length / 2, 0, 0),
            'up': agx.Vec3(0., 0., 1.)
        }
        grippers = {'gripper_right'}

        args = sys.argv
        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(BendWireEnv, self).__init__(scene_path=SCENE_PATH,
                                          n_substeps=n_substeps,
                                          grippers=grippers,
                                          length=length,
                                          n_actions=1,
                                          camera=camera,
                                          args=args,
                                          distance_threshold=math.sqrt(5e-7),  # in m
                                          reward_type=reward_type,
                                          terminate_when_unhealthy=False,
                                          damage_threshold=1e3,
                                          observation_type=observation_type,
                                          randomized_goal=False,
                                          goal_scene_path=GOAL_SCENE_PATH)
