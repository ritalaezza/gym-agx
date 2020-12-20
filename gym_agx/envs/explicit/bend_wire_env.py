import os
import sys
import agx
import logging
import numpy as np

import agxIO
import agxSDK

from gym_agx.envs import dlo_env
from gym_agx.rl.reward import RewardConfig
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.rl.observation import ObservationConfig
from gym_agx.utils.utils import goal_distance
from gym_agx.sims import bend_wire_random_goal

FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.split(FILE_DIRECTORY)[0]
SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire_hinge.agx')
GOAL_SCENE_PATH = os.path.join(PACKAGE_DIRECTORY, 'assets', 'bend_wire_hinge_goal.agx')

logger = logging.getLogger('gym_agx.envs')


class Reward(RewardConfig):

    def reward_function(self, achieved_goal, desired_goal, info):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'], desired_goal['dlo_curvature'])
        info['distance'] = curvature_distance
        if not self.is_success(achieved_goal, desired_goal):
            # penalize large distances to goal
            reward = np.clip(-curvature_distance, self.reward_range[0], self.reward_range[1])
        else:
            # reward achieving goal, but penalize non-zero velocity
            velocity_scale = 100
            ee_velocity = np.linalg.norm(achieved_goal['ee_velocity']['gripper_right'])
            reward = np.clip(self.reward_range[1] - velocity_scale*ee_velocity, 0, self.reward_range[1])
        return reward, info

    def scale_reward(self, reward):
        return np.round(reward, decimals=4)

    def success_condition(self, achieved_goal, desired_goal):
        curvature_distance = goal_distance(achieved_goal['dlo_curvature'], desired_goal['dlo_curvature'])
        return bool(curvature_distance < self.kwargs['dlo_curvature_threshold'])


class BendWireEnv(dlo_env.DloEnv):
    """Subclass which inherits from DLO environment."""

    def __init__(self, n_substeps, observation_config=None, grippers=None, reward_type=None, reward_config=None,
                 scene_path=None, goal_scene_path=None, **kwargs):
        """Initializes BendWire environment
        :param int n_substeps: number of simulation steps between each action step.
        :param ObservationConfig: types of observations to be used.
        :param list grippers: EndEffector objects.
        :param RewardConfig.RewardType reward_type: type of reward.
        :param RewardConfig reward_config: adds possibility to completely override reward definition.
        :param str scene_path: possibility to overwrite default scene file.
        :param str goal_scene_path: possibility to overwrite default goal scene file.
        """
        length = 0.1  # meters
        camera_distance = 0.5  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(length / 2, -3 * length, 0),
            center=agx.Vec3(length / 2, 0, 0),
            up=agx.Vec3(0., 0., 1.),
            light_position=agx.Vec4(length / 2, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.)
        )

        if not grippers:
            gripper_right = EndEffector(
                name='gripper_right',
                controllable=True,
                observable=True,
                max_velocity=10 / 1000,  # m/s
                max_acceleration=10 / 1000,  # m/s^2
            )
            gripper_right.add_constraint(name='gripper_right_joint_base_x',
                                         end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                         compute_forces_enabled=False,
                                         velocity_control=True,
                                         compliance_control=False)
            gripper_right.add_constraint(name='gripper_right_joint_base_y',
                                         end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                                         compute_forces_enabled=False,
                                         velocity_control=False,
                                         compliance_control=False)
            gripper_right.add_constraint(name='gripper_right_joint_base_z',
                                         end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATION,
                                         compute_forces_enabled=False,
                                         velocity_control=False,
                                         compliance_control=False)
            gripper_right.add_constraint(name='hinge_joint_right',
                                         end_effector_dof=EndEffectorConstraint.Dof.Y_ROTATION,
                                         compute_forces_enabled=False,
                                         velocity_control=False,
                                         compliance_control=False)

            gripper_left = EndEffector(
                name='gripper_left',
                controllable=False,
                observable=False,
            )
            gripper_left.add_constraint(name='prismatic_joint_left',
                                        end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                        compute_forces_enabled=False,
                                        velocity_control=False)
            gripper_left.add_constraint(name='hinge_joint_left',
                                        end_effector_dof=EndEffectorConstraint.Dof.Y_ROTATION,
                                        compute_forces_enabled=False,
                                        velocity_control=False)
            grippers = [gripper_right, gripper_left]

        if not observation_config:
            observation_config = ObservationConfig(goals=[ObservationConfig.ObservationType.DLO_CURVATURE,
                                                          ObservationConfig.ObservationType.EE_VELOCITY])
            observation_config.set_dlo_frenet_curvature()

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
        randomized_goal = kwargs['randomized_goal'] if 'randomized_goal' in kwargs else False

        if not os.path.exists(SCENE_PATH):
            raise IOError("File %s does not exist" % SCENE_PATH)
        logger.info("Fetching environment from {}".format(SCENE_PATH))

        super(BendWireEnv, self).__init__(args=args,
                                          scene_path=scene_path,
                                          n_substeps=n_substeps,
                                          end_effectors=grippers,
                                          observation_config=observation_config,
                                          camera_config=camera_config,
                                          reward_config=reward_config,
                                          randomized_goal=randomized_goal,
                                          goal_scene_path=goal_scene_path,
                                          show_goal=show_goal,
                                          osg_window=osg_window,
                                          agx_only=agx_only)

    def _sample_goal(self):

        if self.randomized_goal:
            goal_cable_length, goal_cable_segments = bend_wire_random_goal.add_goal(self.sim, logger)
            logger.info(f"Added goal cable consisting of {goal_cable_segments} segments "
                        f"with a total length of {goal_cable_length}.")

        else:
            scene = agxSDK.Assembly()  # Create a new empty Assembly
            scene.setName("goal_assembly")

            if not agxIO.readFile(self.goal_scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
                raise RuntimeError("Unable to open goal file \'" + self.goal_scene_path + "\'")

            self.sim.add(scene)

        goal = self.observation_config.get_observations(self.sim, self.render_to_image, self.end_effectors, cable="DLO",
                                                        goal_only=True)

        if self.show_goal:
            self._add_rendering()
        else:
            self._reset_sim()

        return goal
