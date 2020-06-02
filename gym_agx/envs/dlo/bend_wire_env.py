import os
import sys
import agx
import logging

from gym_agx.envs import dlo_env
from gym_agx.utils.agx_utils import CameraSpecs, EndEffector, EndEffectorConstraint

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire_hinge.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire_hinge_goal.agx')

logger = logging.getLogger('gym_agx.envs')


class BendWireEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment.
    """

    def __init__(self, reward_type='sparse', n_substeps=2):
        """Initializes BendWire environment
        The length should be consistent with the model defined in 'SCENE_PATH'.
        :param reward_type: either 'sparse' or 'dense'
        """
        length = 0.1  # meters

        camera_distance = 0.5  # meters
        camera = CameraSpecs(
            eye=agx.Vec3(length / 2, -5 * length, 0),
            center=agx.Vec3(length / 2, 0, 0),
            up=agx.Vec3(0., 0., 1.),
            light_position=agx.Vec4(length / 2, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        gripper_right = EndEffector(
            name='gripper_right',
            controllable=True,
            observable=True,
            max_velocity=14 / 1000,  # m/s
            max_acceleration=10 / 1000,  # m/s^2
            min_compliance=1,  # 1/Nm
            max_compliance=1e12  # 1/Nm
        )
        gripper_right.add_constraint(name='prismatic_joint_right',
                                     end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATIONAL,
                                     compute_forces_enabled=True,
                                     velocity_control=True,
                                     compliance_control=False)
        gripper_right.add_constraint(name='hinge_joint_right',
                                     end_effector_dof=EndEffectorConstraint.Dof.Y_ROTATIONAL,
                                     compute_forces_enabled=False,
                                     velocity_control=False,
                                     compliance_control=False)

        gripper_left = EndEffector(
            name='gripper_left',
            controllable=False,
            observable=False,
        )
        gripper_left.add_constraint(name='prismatic_joint_left',
                                    end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATIONAL,
                                    compute_forces_enabled=False,
                                    velocity_control=False)
        gripper_left.add_constraint(name='hinge_joint_left',
                                    end_effector_dof=EndEffectorConstraint.Dof.Y_ROTATIONAL,
                                    compute_forces_enabled=False,
                                    velocity_control=False)

        args = sys.argv
        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(BendWireEnv, self).__init__(scene_path=SCENE_PATH,
                                          n_substeps=n_substeps,
                                          end_effectors=[gripper_right, gripper_left],
                                          camera=camera,
                                          args=args,
                                          distance_threshold=0.06,  # 0.16
                                          reward_type=reward_type,
                                          reward_limit=1.5,
                                          randomized_goal=False,
                                          goal_scene_path=GOAL_SCENE_PATH)
