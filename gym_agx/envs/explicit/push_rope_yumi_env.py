import os
import sys
import agx
import agxSDK
import logging
import numpy as np

from gym_agx.envs import dlo_env
from gym_agx.rl.reward import RewardConfig
from gym_agx.utils.agx_classes import CameraConfig, ContactEventListenerRigidBody
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.rl.joint_entity import JointObjects, JointConstraint
from gym_agx.rl.observation import ObservationConfig
from gym_agx.utils.utils import goal_distance
from gym_agx.utils.agx_utils import add_collision_events

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_yumi.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'push_rope_yumi_goal.agx')
JOINT_NAMES_REV = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                   'yumi_joint_5_r', 'yumi_joint_6_r']  # names of controllable the revolute joints

logger = logging.getLogger('gym_agx.envs')


class Reward(RewardConfig):

    def reward_function(self, achieved_goal, desired_goal, info):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'],
                                           desired_goal['dlo_curvature'])
        info['distance'] = curvature_distance

        if self.is_success(achieved_goal, desired_goal):
            reward = self.reward_range[1]
        else:
            reward = -curvature_distance

        return reward, info

    def scale_reward(self, reward):
        return np.clip(reward, self.reward_range[0], self.reward_range[1])

    def success_condition(self, achieved_goal, desired_goal):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'],
                                           desired_goal['dlo_curvature'])
        return bool(curvature_distance < self.kwargs['dlo_curvature_threshold'])



class PushRopeYuMiEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps, observation_config=None, joints=None, reward_type=None, reward_config=None,
                 scene_path=None, goal_scene_path=None, **kwargs):
        """Initializes BendWire environment
        :param int n_substeps: number of simulation steps between each action step.
        :param ObservationConfig: types of observations to be used.
        :param list joints: joints objects.
        :param RewardConfig.RewardType reward_type: type of reward.
        :param RewardConfig reward_config: adds possibility to completely override reward definition.
        :param str scene_path: possibility to overwrite default scene file.
        :param str goal_scene_path: possibility to overwrite default goal scene file.
        """
        length = 0.1  # meters
        camera_distance = 0.5  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(1, 0.1, 0.3),
            center=agx.Vec3(0.3, 0, 0.3),
            up=agx.Vec3(0., 0., 1.),
            light_position=agx.Vec4(length / 2, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        # links to do collision detection event for
        self.link_names = ['gripper_r_finger_r', 'gripper_r_finger_l', 'pusher']
        self.contact_names = ['contact_' + link_name for link_name in self.link_names]
        self.ignore_names = ['dlo'] # geometry names to ignore from collision detection event

        if not joints:
            yumi = JointObjects(
                name='yumi',
                controllable=True,
                observable=True,
            )
            for i in range(len(JOINT_NAMES_REV)):
                yumi.add_constraint(name=JOINT_NAMES_REV[i],
                                    type_of_joint=JointConstraint.Type.REVOLUTE,
                                    velocity_control=True)
            joints = [yumi]

        if not observation_config:
            observation_config = ObservationConfig(goals=[ObservationConfig.ObservationType.DLO_CURVATURE])
            observation_config.set_dlo_frenet_curvature()
            observation_config.set_joint_position()
            observation_config.set_joint_velocity()
            observation_config.set_rb_contact_listener(self.contact_names)

        if not reward_config:
            if not reward_type:
                reward_type = RewardConfig.RewardType.DENSE
            reward_config = Reward(reward_type=reward_type, reward_range=(-1.5, 1.5), set_done_on_success=False,
                                   dlo_curvature_threshold=0.05)
        if not scene_path:
            scene_path = SCENE_PATH
        if not goal_scene_path:
            goal_scene_path = GOAL_SCENE_PATH

        args = kwargs['agxViewer'] if 'agxViewer' in kwargs else sys.argv
        show_goal = kwargs['show_goal'] if 'show_goal' in kwargs else False
        osg_window = kwargs['osg_window'] if 'osg_window' in kwargs else False
        agx_only = kwargs['agx_only'] if 'agx_only' in kwargs else False

        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(PushRopeYuMiEnv, self).__init__(args=args,
                                          scene_path=scene_path,
                                          n_substeps=n_substeps,
                                          end_effectors=joints,
                                          observation_config=observation_config,
                                          camera_config=camera_config,
                                          reward_config=reward_config,
                                          randomized_goal=False,
                                          goal_scene_path=goal_scene_path,
                                          show_goal=show_goal,
                                          osg_window=osg_window,
                                          agx_only=agx_only)

    def reset(self):
        logger.info("reset")
        super(PushRopeYuMiEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
            add_collision_events(self.link_names, self.contact_names, self.ignore_names, ContactEventListenerRigidBody,
                                 self.sim)
        self.goal = self._sample_goal().copy()
        obs = self._get_observation()
        return obs

