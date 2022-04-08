import os
import sys
import agx
import logging
import numpy as np

from gym_agx.envs import dlo_env
from gym_agx.rl.observation import ObservationConfig
from gym_agx.rl.reward import RewardConfig
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.sims import push_rope

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_goal.agx')
RANDOM_GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_goal_random.agx')

logger = logging.getLogger('gym_agx.envs')


class Reward(RewardConfig):

    def reward_function(self, achieved_goal, desired_goal, info):
        curvature_distance = np.linalg.norm(achieved_goal['dlo_curvature'] - desired_goal['dlo_curvature'])
        info['distance'] = curvature_distance

        if self.is_success(achieved_goal, desired_goal):
            reward = self.reward_range[1]
        else:
            reward = -curvature_distance

        return reward, info

    def scale_reward(self, reward):
        return np.clip(reward, self.reward_range[0], self.reward_range[1])

    def success_condition(self, achieved_goal, desired_goal):
        curvature_distance = np.linalg.norm(achieved_goal['dlo_curvature'] - desired_goal['dlo_curvature'])
        return bool(curvature_distance < self.kwargs['dlo_curvature_threshold'])


class PushRopeEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps, observation_config=None, pushers=None, reward_type=None, reward_config=None,
                 scene_path=None, goal_scene_path=None, **kwargs):
        """Initializes PushRope environment
        :param int n_substeps: number of simulation steps between each action step
        :param ObservationConfig: types of observations to be used
        :param list pushers: EndEffector objects
        :param RewardConfig.RewardType reward_type: type of reward
        :param RewardConfig reward_config: adds possibility to completely override reward definition
        :param str scene_path: possibility to overwrite default scene file
        :param str goal_scene_path: possibility to overwrite default goal scene file
        """
        camera_distance = 0.21  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, - camera_distance / 2, camera_distance),
            center=agx.Vec3(0, 0, 0),
            up=agx.Vec3(0., 0., 0.),
            light_position=agx.Vec4(0, -camera_distance / 2, camera_distance * 0.9, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        if not pushers:
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
            pushers = [pusher]

        if not observation_config:
            observation_config = ObservationConfig(goals=[ObservationConfig.ObservationType.DLO_CURVATURE])
            observation_config.set_dlo_frenet_curvature()
            observation_config.set_ee_position()

        if not reward_type:
            reward_type = RewardConfig.RewardType.DENSE
        if not reward_config:
            reward_config = Reward(reward_type=reward_type, reward_range=(-1.5, 1.5), dlo_curvature_threshold=0.1)

        if not scene_path:
            scene_path = SCENE_PATH
        if not goal_scene_path:
            goal_scene_path = GOAL_SCENE_PATH

        args = kwargs['agxViewer'] if 'agxViewer' in kwargs else sys.argv
        show_goal = kwargs['show_goal'] if 'show_goal' in kwargs else False
        osg_window = kwargs['osg_window'] if 'osg_window' in kwargs else False
        agx_only = kwargs['agx_only'] if 'agx_only' in kwargs else False
        randomized_goal = kwargs['randomized_goal'] if 'randomized_goal' in kwargs else False

        # Overwrite goal_scene_path with starting point for random goals
        if randomized_goal:
            goal_scene_path = RANDOM_GOAL_SCENE_PATH

        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(PushRopeEnv, self).__init__(args=args,
                                          scene_path=scene_path,
                                          n_substeps=n_substeps,
                                          end_effectors=pushers,
                                          observation_config=observation_config,
                                          camera_config=camera_config,
                                          reward_config=reward_config,
                                          randomized_goal=randomized_goal,
                                          goal_scene_path=goal_scene_path,
                                          show_goal=show_goal,
                                          osg_window=osg_window,
                                          agx_only=agx_only)

    def _sample_random_goal(self, sim):
        push_rope.sample_random_goal(self.sim)

