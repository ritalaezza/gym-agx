import os
import sys
import agx
import logging

from gym_agx.envs import dlo_env
from gym_agx.utils.agx_classes import CameraSpecs, EndEffector, EndEffectorConstraint

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_goal.agx')
# TODO: Make scene_path and goal_scene_path be passed in kwargs. Maybe just keep one as default.

logger = logging.getLogger('gym_agx.envs')


class PushRopeEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment.
    """

    def __init__(self, reward_type='sparse', n_substeps=20, **kwargs):
        """Initializes PushRope environment
        The radius and length should be consistent with the model defined in 'SCENE_PATH'.
        :param reward_type: either 'sparse' or 'dense'
        """
        camera_distance = 0.5  # meters
        camera = CameraSpecs(
            eye=agx.Vec3(0, 0, 0.5),
            center=agx.Vec3(0, 0, 0),
            up=agx.Vec3(0., 0., 0.),
            light_position=agx.Vec4(0.1, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        pusher = EndEffector(
            name='pusher',
            controllable=True,
            observable=False,
            max_velocity=14 / 1000,  # m/s
            max_acceleration=10 / 1000,  # m/s^2
        )
        pusher.add_constraint(name='pusher_joint_base_x',
                              end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATIONAL,
                              compute_forces_enabled=False,
                              velocity_control=True,
                              compliance_control=False)
        pusher.add_constraint(name='pusher_joint_base_y',
                              end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATIONAL,
                              compute_forces_enabled=False,
                              velocity_control=True,
                              compliance_control=False)
        pusher.add_constraint(name='pusher_joint_base_z',
                              end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATIONAL,
                              compute_forces_enabled=False,
                              velocity_control=False,
                              compliance_control=False)

        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv

        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(PushRopeEnv, self).__init__(args=args,
                                          scene_path=SCENE_PATH,
                                          n_substeps=n_substeps,
                                          end_effectors=[pusher],
                                          camera=camera,
                                          distance_threshold=0.06,
                                          reward_type=reward_type,
                                          reward_limit=1.5,
                                          randomized_goal=False,
                                          goal_scene_path=GOAL_SCENE_PATH)
