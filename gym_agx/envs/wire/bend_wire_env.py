import os
import sys
import agx
from gym_agx.envs import wire_env

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire.agx')


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
        camera = {
            'eye': agx.Vec3(length / 2, -5 * length, 0),
            'center':  agx.Vec3(length / 2, 0, 0),
            'up': agx.Vec3(0., 0., 1.)
        }
        grippers = {'gripper_right'}

        args = sys.argv
        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % self.scene_path)
        print("Fetching environment from {}".format(SCENE_PATH))

        super(BendWireEnv, self).__init__(scene_path=SCENE_PATH, n_substeps=10, grippers=grippers, length=length,
                                          n_actions=6, camera=camera, args=args, distance_threshold=0.01**3,
                                          reward_type=reward_type,  terminate_when_unhealthy=True, damage_threshold=1e7)
