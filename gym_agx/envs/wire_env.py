import sys
import math
import logging
import numpy as np

import agxSDK
import agxOSG
import agxCable
import agxRender

from gym_agx.envs import agx_env
from gym_agx.utils.agx_utils import get_cable_state
from gym_agx.utils.utils import get_cable_curvature, sinusoidal_trajectory

logger = logging.getLogger('gym_agx.envs')


def goal_distance(achieved_goal, goal, norm="l2"):
    assert achieved_goal.shape == goal.shape
    if norm == "l1":
        return np.sum(abs(achieved_goal - goal))
    elif norm == "l2":
        return np.linalg.norm(achieved_goal - goal)
    elif norm == 'inf':
        return np.linalg.norm(achieved_goal - goal,  np.inf)
    elif norm == '-inf':
        return np.linalg.norm(achieved_goal - goal,  -np.inf)
    else:
        logger.error("Unexpected norm.")


class WireEnv(agx_env.AgxEnv):
    """Superclass for all Wire environments.
    """
    def __init__(self, scene_path, n_substeps, grippers, camera, args, distance_threshold,
                 reward_type, reward_limit, randomized_goal, goal_scene_path):
        """Initializes a WireEnv object
        :param scene_path: path to binary file containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step()
        :param grippers: list of Gripper objects
        :param camera: dictionary containing EYE, CENTER, UP information for rendering together with lighting info
        :param args: sys.argv
        :param distance_threshold: threshold for reward function
        :param reward_type: reward type, i.e. 'sparse' or 'dense'
        :param reward_limit: reward limit to bound reward
        :param randomized_goal: boolean deciding if a new goal is sampled for each episode
        :param goal_scene_path: path to goal scene file
        """
        self.distance_threshold = distance_threshold
        self.goal_scene_path = goal_scene_path
        self.randomized_goal = randomized_goal
        self.reward_type = reward_type
        self.reward_limit = reward_limit
        self.light_pose = camera.light_pose
        self.grippers = grippers

        n_actions = 0
        for gripper in grippers:
            if gripper.controllable:
                for key, constraint in gripper.constraints.items():
                    if constraint.velocity_control:
                        n_actions += 1
                    if constraint.compliance_control:
                        n_actions += 1

        super(WireEnv, self).__init__(scene_path=scene_path, n_substeps=n_substeps, n_actions=n_actions,
                                      camera_pose=camera.camera_pose, args=args)

        # Keep track of latest action
        self.last_action = self.action_space.sample()*0

    @property
    def done(self):
        return False

    def render(self, mode="human"):
        return super(WireEnv, self).render(mode)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal
        curvature_distance = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            if self._is_success(achieved_goal, goal):
                return self.reward_limit
            else:
                return -self.reward_limit
        else:
            reward = 0
            if not self._is_success(achieved_goal, goal):
                # penalize large distances to goal
                reward = np.clip(-curvature_distance, -self.reward_limit, self.reward_limit)
            else:
                # reward achieving goal
                total_action = abs(self.last_action).sum()
                reward = np.clip(self.reward_limit - total_action, 0, self.reward_limit)
            return reward

    # AgxEnv methods
    # ----------------------------

    def _add_rendering(self, mode='osg'):
        self.app.setAutoStepping(False)
        if mode == 'osg':
            self.app.setEnableDebugRenderer(False)
            self.app.setEnableOSGRenderer(True)
        elif mode == 'debug':
            self.app.setEnableDebugRenderer(True)
            self.app.setEnableOSGRenderer(False)
        else:
            logger.error("Unexpected rendering mode: {}".format(mode))

        self.root = self.app.getRoot()
        rbs = self.sim.getRigidBodies()
        for rb in rbs:
            if rb.getName() == "ground":
                ground_node = agxOSG.createVisual(rb, self.root)
                agxOSG.setDiffuseColor(ground_node, agxRender.Color(1.0, 1.0, 1.0, 1.0))
            elif rb.getName() == "gripper_left":
                gripper_left_node = agxOSG.createVisual(rb, self.root)
                agxOSG.setDiffuseColor(gripper_left_node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
            elif rb.getName() == "gripper_right":
                gripper_right_node = agxOSG.createVisual(rb, self.root)
                agxOSG.setDiffuseColor(gripper_right_node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
            else:  # Cable segments
                cable_node = agxOSG.createVisual(rb, self.root)
                agxOSG.setDiffuseColor(cable_node, agxRender.Color(0.0, 1.0, 0.0, 1.0))

        scene_decorator = self.app.getSceneDecorator()
        light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
        light_source_0.setPosition(self.light_pose['light_position'])
        light_source_0.setDirection(self.light_pose['light_direction'])
        scene_decorator.setEnableLogo(False)

    def _reset_sim(self):
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_ALL):
            logger.error("Unable to restore simulation!")
            return False

        self._add_rendering(mode='osg')
        return True

    def _get_obs(self):
        obs = dict.fromkeys({'observation', 'achieved_goal', 'desired_goal'})

        cable = agxCable.Cable.find(self.sim, "DLO")
        cable_state = get_cable_state(cable)
        cable_curvature = get_cable_curvature(cable_state)

        grippers_state = []
        for i, gripper in enumerate(self.grippers):
            if gripper.observable:
                grippers_state.append(gripper.get_state(self.sim).ravel(order='F'))
        grippers_state = np.asarray(grippers_state).flatten()

        if len(grippers_state) > 0:
            observation = np.concatenate((cable_curvature, grippers_state))
        else:
            observation = cable_state
        obs['observation'] = observation.ravel(order='F')
        obs['achieved_goal'] = cable_curvature

        obs['desired_goal'] = self.goal
        return obs

    def _set_action(self, action):
        for i, gripper in enumerate(self.grippers):
            if gripper.controllable:
                gripper.apply_control(self.sim, action, self.dt)

        self.last_action = action

    def _is_success(self, achieved_goal, desired_goal):
        curvature_distance = goal_distance(achieved_goal, desired_goal)
        return curvature_distance < self.distance_threshold

    def _sample_goal(self):
        if self.randomized_goal:
            n_steps = 1000
            dt = self.sim.getTimeStep()
            min_period = n_steps*dt

            # Define initial linear and angular velocities
            gripper_right = self.sim.getRigidBody('gripper_right')
            amplitude = np.random.uniform(low=0.0, high=0.025)
            period = np.random.uniform(low=min_period, high=2 * min_period)
            rad_frequency = 2 * math.pi * (1 / period)
            for k in range(n_steps):
                velocity_x = sinusoidal_trajectory(amplitude, rad_frequency, k * dt)
                gripper_right.setVelocity(velocity_x, 0, 0)
                self.sim.stepForward()
        else:
            self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
            if not self.sim.restore(self.goal_scene_path, agxSDK.Simulation.READ_ALL):
                logger.error("Unable to restore goal!")

        cable = agxCable.Cable.find(self.sim, "DLO")
        goal_state = get_cable_state(cable)
        goal_curvature = get_cable_curvature(goal_state)
        self._reset_sim()

        return goal_curvature

    def _step_callback(self):
        t = self.sim.getTimeStamp()

        t_0 = t
        while t < t_0 + self.dt:
            try:
                self.sim.stepForward()
                t = self.sim.getTimeStamp()
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])

    def _render_callback(self):
        if not self.app.breakRequested():
            self.app.executeOneStepWithGraphics()
        else:
            self._init_app(True)
            self.app.executeOneStepWithGraphics()
