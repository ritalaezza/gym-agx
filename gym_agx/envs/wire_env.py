import sys
import math
import logging
import numpy as np

import agx
import agxSDK
import agxOSG
import agxCable
import agxRender

from gym_agx.envs import agx_env
from gym_agx.utils.agx_utils import get_cable_state, get_gripper_state, to_agx_list
from gym_agx.utils.agx_utils import compute_linear_distance, compute_angular_distance

logger = logging.getLogger(__name__)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


def goal_distance(goal_a, goal_b):
    logger.debug("goal distance")
    assert goal_a.shape == goal_b.shape
    goal_a_reshaped = goal_a.reshape(7, int(len(goal_a)/7), order='F')
    goal_b_reshaped = goal_b.reshape(7, int(len(goal_a)/7), order='F')
    linear_distance = np.zeros(int(len(goal_a)/7))
    angular_distance = np.zeros(int(len(goal_a)/7))
    for i in range(0, len(linear_distance)):
        p_a = goal_a_reshaped[:3, i]
        p_b = goal_b_reshaped[:3, i]
        q_a = goal_a_reshaped[3:, i]
        q_b = goal_b_reshaped[3:, i]
        linear_distance[i] = compute_linear_distance(p_a, p_b)
        angular_distance[i] = compute_angular_distance(q_a, q_b)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(goal_a_reshaped[0, :], goal_a_reshaped[1, :], goal_a_reshaped[2, :])
    # ax.scatter(goal_b_reshaped[0, :], goal_b_reshaped[1, :], goal_b_reshaped[2, :])
    # plt.show()

    return linear_distance, angular_distance


def sinusoidal_trajectory(amp, w, t):
    return -amp * w * math.sin(w * t)


class WireEnv(agx_env.AgxEnv):
    """Superclass for all Wire environments.
    """
    def __init__(self, scene_path, n_substeps, grippers, length, n_actions, camera, args, distance_threshold,
                 reward_type, terminate_when_unhealthy, damage_threshold, observation_type, randomized_goal,
                 goal_scene_path):
        """Initializes a WireEnv object
        :param scene_path: path to binary file containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step()
        :param grippers: dictionary containing gripper names
        :param length: length of wire
        :param n_actions: number of actions (DoF)
        :param camera: dictionary containing EYE, CENTER, UP information for rendering
        :param args: sys.argv
        :param distance_threshold: threshold for reward function
        :param reward_type: reward type, i.e. 'sparse' or 'dense'
        :param terminate_when_unhealthy: boolean to determine early stopping when too much damage occurs
        :param damage_threshold: damage threshold used when 'terminate_when_unhealthy' is True
        :param observation_type: either 'vector' or 'matrix'
        :param randomized_goal: boolean deciding if a new goal is sampled for each episode
        :param goal_scene_path: path to goal scene file
        """
        # TODO: may want to move some of these to parent class
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.distance_threshold = distance_threshold
        self.damage_threshold = damage_threshold
        self.observation_type = observation_type
        self.goal_scene_path = goal_scene_path
        self.randomized_goal = randomized_goal
        self.reward_type = reward_type
        self.grippers = grippers
        self.length = length

        super(WireEnv, self).__init__(scene_path=scene_path, n_substeps=n_substeps, grippers=grippers,
                                      n_actions=n_actions, camera=camera, args=args)

    @property
    def is_healthy(self):
        largest = agxCable.SegmentDamage()
        cable = agxCable.Cable.find(self.sim, "DLO")
        cable_damage = agxCable.CableDamage.getCableDamage(cable)
        damages = cable_damage.getCurrentDamages()

        for d in damages:
            for e in range(agxCable.NUM_CABLE_DAMAGE_TYPES):
                largest[e] = max(largest[e], d[e])
        if largest.total() > self.damage_threshold:
            return False
        else:
            return True

    @property
    def done(self):
        done = (not self.is_healthy if self.terminate_when_unhealthy else False)
        return done

    def render(self, mode="human"):
        return super(WireEnv, self).render(mode)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        logger.debug("compute reward")
        # Compute distance between goal and the achieved goal
        distance_per_segment, __ = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -1 * float(any(dist >= self.distance_threshold for dist in distance_per_segment))
        else:
            if any(dist >= self.distance_threshold for dist in distance_per_segment):
                reward = -1 * distance_per_segment.mean()
            else:
                reward = 0
            return reward

    # AgxEnv methods
    # ----------------------------

    def _add_rendering(self, mode='osg'):
        logger.debug("add rendering")
        camera_distance = 0.5
        light_pos = agx.Vec4(self.length / 2, - camera_distance, camera_distance, 1.)
        light_dir = agx.Vec3(0., 0., -1.)

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
        light_source_0.setPosition(light_pos)
        light_source_0.setDirection(light_dir)
        scene_decorator.setEnableLogo(False)

    def _reset_sim(self):
        logger.debug("reset sim")
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_DEFAULT):
            logger.error("Unable to restore simulation!")
            return False

        self._add_rendering(mode='osg')
        return True

    def _get_obs(self):
        logger.debug("get obs")
        obs = dict.fromkeys({'observation', 'achieved_goal', 'desired_goal'})

        cable = agxCable.Cable.find(self.sim, "DLO")
        padded_cable_state = np.zeros(shape=(14, cable.getNumSegments()))
        cable_state = get_cable_state(cable)
        padded_cable_state[:7, :] = cable_state
        gripper_state = get_gripper_state(self.sim, self.grippers)

        if self.observation_type == 'vector':
            padded_cable_state[7:, :] = self.goal.reshape(7, cable.getNumSegments(), order='F')
            observation = np.concatenate((gripper_state, padded_cable_state), axis=1)
            obs['observation'] = observation.ravel(order='F')
            obs['achieved_goal'] = cable_state.ravel(order='F')
        elif self.observation_type == 'matrix':
            observation = np.concatenate((gripper_state, padded_cable_state), axis=1)
            obs['observation'] = observation
            obs['achieved_goal'] = cable_state
        else:
            logger.error("Unsupported observation type!")

        obs['desired_goal'] = self.goal
        return obs

    def _set_action(self, stacked_action):
        logger.debug("set action")
        n_grippers = len(self.grippers)
        action = np.reshape(stacked_action, newshape=(int(len(stacked_action)/n_grippers), n_grippers))
        for i, key in enumerate(self.grippers):
            gripper = self.sim.getRigidBody(key)
            velocity = np.zeros(3)
            velocity[0] = action[0, i]
            linear_velocity = to_agx_list(velocity, agx.Vec3)
            linear_velocity *= 0.001  # from meters per second to millimeters per second

            if self.is_healthy:
                gripper.setVelocity(linear_velocity)
            else:
                gripper.setVelocity(0, 0, 0)
            # angular_velocity = to_agx_list(action[3:, i], agx.Vec3)
            # angular_velocity *= 0.5  # half of input rad/s
            # gripper.setAngularVelocity(angular_velocity)

    def _is_success(self, achieved_goal, desired_goal):
        logger.info("is success")
        distance_per_segment, __ = goal_distance(achieved_goal, desired_goal)
        return all(dist < self.distance_threshold for dist in distance_per_segment)

    def _sample_goal(self):
        logger.debug("sample goal")
        if self.randomized_goal:
            n_steps = 1000
            dt = self.sim.getTimeStep()
            min_period = n_steps*dt

            valid_goal = False
            while not valid_goal:
                # Define initial linear and angular velocities
                gripper_right = self.sim.getRigidBody('gripper_right')
                amplitude = np.random.uniform(low=0.0, high=self.length/4)
                period = np.random.uniform(low=min_period, high=2*min_period)
                rad_frequency = 2 * math.pi * (1 / period)
                for k in range(n_steps):
                    velocity_x = sinusoidal_trajectory(amplitude, rad_frequency, k*dt)
                    gripper_right.setVelocity(velocity_x, 0, 0)
                    self.sim.stepForward()

                    # Check for cable damage
                    if not self.is_healthy:
                        self._reset_sim()
                        logger.info("Too much damage!")
                        break

                valid_goal = True
        else:
            self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
            if not self.sim.restore(self.goal_scene_path, agxSDK.Simulation.READ_DEFAULT):
                logger.error("Unable to restore goal!")

        cable = agxCable.Cable.find(self.sim, "DLO")
        goal = get_cable_state(cable)
        if self.observation_type == 'vector':
            goal = goal.ravel(order="F")
        self._reset_sim()

        return goal

    def _step_callback(self):
        logger.debug("step callback")
        t = self.sim.getTimeStamp()

        t_0 = t
        while t < t_0 + self.dt:
            try:
                self.sim.stepForward()
                t = self.sim.getTimeStamp()
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])

    def _render_callback(self):
        logger.debug("render callback")
        if not self.app.breakRequested():
            self.app.executeOneStepWithGraphics()
        else:
            self._init_app(True)
            self.app.executeOneStepWithGraphics()
