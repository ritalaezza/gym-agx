import os
import sys
import agx
import logging
import numpy as np

from gym_agx.envs import dlo_env
from gym_agx.rl.observation import ObservationConfig
from gym_agx.rl.reward import RewardConfig
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.utils import goal_distance

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_goal.agx')
# TODO: Make scene_path and goal_scene_path be passed in kwargs. Maybe just keep one as default.

logger = logging.getLogger('gym_agx.envs')


class Reward(RewardConfig):

    def reward_function(self, achieved_goal, desired_goal, info):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'],
                                           desired_goal['dlo_curvature'])
        info['distance'] = curvature_distance
        ee_position = achieved_goal['ee_position']['pusher']
        min_distance = np.inf
        dlo_positions = achieved_goal['dlo_positions']
        for i in range(dlo_positions.shape[1]):
            ee_distance = goal_distance(ee_position, dlo_positions[:, i])
            if ee_distance < min_distance:
                min_distance = ee_distance

        if min_distance < 0.011:
            bonus = 0.01
        else:
            bonus = 0
        reward = -curvature_distance + bonus

        if self.is_success(achieved_goal, desired_goal):
            reward = self.reward_range[1]

        return reward, info

    def scale_reward(self, reward):
        return np.clip(reward, self.reward_range[0], self.reward_range[1])

    def success_condition(self, achieved_goal, desired_goal):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'],
                                           desired_goal['dlo_curvature'])
        return bool(curvature_distance < self.kwargs['dlo_curvature_threshold'])


class PushRopeEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, reward_type=RewardConfig.RewardType.DENSE, n_substeps=20, **kwargs):
        """Initializes PushRope environment
        The radius and length should be consistent with the model defined in 'SCENE_PATH'.
        :param reward_type: either 'sparse' or 'dense'
        """
        camera_distance = 0.5  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, 0, 0.5),
            center=agx.Vec3(0, 0, 0),
            up=agx.Vec3(0., 0., 0.),
            light_position=agx.Vec4(0.1, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        pusher = EndEffector(
            name='pusher',
            controllable=True,
            observable=True,
            max_velocity=5 / 100,  # m/s
            max_acceleration=10 / 100,  # m/s^2
        )
        pusher.add_constraint(name='pusher_joint_base_x',
                              end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                              compute_forces_enabled=False,
                              velocity_control=True,
                              compliance_control=False)
        pusher.add_constraint(name='pusher_joint_base_y',
                              end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                              compute_forces_enabled=False,
                              velocity_control=True,
                              compliance_control=False)
        pusher.add_constraint(name='pusher_joint_base_z',
                              end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATION,
                              compute_forces_enabled=False,
                              velocity_control=False,
                              compliance_control=False)

        observation_config = ObservationConfig(goals=[ObservationConfig.ObservationType.DLO_CURVATURE,
                                                      ObservationConfig.ObservationType.DLO_POSITIONS,
                                                      ObservationConfig.ObservationType.EE_POSITION])
        observation_config.set_dlo_frenet_curvature()
        observation_config.set_dlo_positions()
        observation_config.set_ee_position()

        reward_config = Reward(reward_type=reward_type, reward_range=(-1.5, 1.5), dlo_curvature_threshold=0.1)

        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv

        if 'show_goal' in kwargs:
            show_goal = kwargs['show_goal']
        else:
            show_goal = False

        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(PushRopeEnv, self).__init__(args=args,
                                          scene_path=SCENE_PATH,
                                          n_substeps=n_substeps,
                                          end_effectors=[pusher],
                                          observation_config=observation_config,
                                          camera_config=camera_config,
                                          reward_config=reward_config,
                                          randomized_goal=False,
                                          goal_scene_path=GOAL_SCENE_PATH,
                                          show_goal=show_goal)
