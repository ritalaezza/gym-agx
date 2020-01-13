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
from gym_agx.utils.agx_utils import get_cable_state, get_force_torque, to_numpy_array, to_agx_list
from gym_agx.utils.agx_utils import compute_linear_distance, compute_angular_distance


def goal_distance(goal_a, goal_b):
    logger.debug("goal distance")
    assert goal_a.shape == goal_b.shape
    linear_distance = np.zeros(shape=goal_a.shape)
    angular_distance = np.zeros(shape=goal_a.shape)
    for i in range(0, len(linear_distance), 7):
        v_a = goal_a[i:i+3]
        v_b = goal_b[i:i+3]
        q_a = goal_a[i+3:i+7]
        q_b = goal_b[i+3:i+7]
        linear_distance[i] = compute_linear_distance(v_a, v_b)
        angular_distance[i] = compute_angular_distance(q_a, q_b)
    return linear_distance, angular_distance


logger = logging.getLogger(__name__)


class WireEnv(agx_env.AgxEnv):
    """Superclass for all Wire environments.
    """
    def __init__(self, scene_path, n_substeps, grippers, length, n_actions, camera, args, distance_threshold,
                 reward_type, terminate_when_unhealthy, damage_threshold):
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
        """
        # TODO: may want to move some of these to parent class
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.distance_threshold = distance_threshold
        self.damage_threshold = damage_threshold
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
        # Compute distance between goal and the achieved goal.
        distance_per_segment, __ = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -1 * float(any(i >= self.distance_threshold for i in distance_per_segment))
        else:
            return -1 * distance_per_segment.mean()

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
        padded_cable_state = np.zeros(shape=(7, cable.getNumSegments(), 2), dtype=float)
        cable_state = get_cable_state(cable)
        padded_cable_state[:, :, 0] = cable_state

        gripper_state = np.zeros(shape=(7, len(self.grippers), 2), dtype=float)
        for i, key in enumerate(self.grippers):
            gripper = self.sim.getRigidBody(key)
            gripper_state[:3, i, 0] = to_numpy_array(gripper.getPosition())
            gripper_state[3:, i, 0] = to_numpy_array(gripper.getRotation())
            gripper_state[:3, i, 1], gripper_state[3:6, i, 1] = get_force_torque(self.sim, gripper, key + '_constraint')
            gripper_state[6, i, 1] = 1  # fill empty space. Boolean indicating gripper and not segment.

        observation = np.concatenate((gripper_state, padded_cable_state), axis=1)
        logger.debug("Observation: {}".format(observation))

        obs['observation'] = observation.ravel()
        obs['achieved_goal'] = cable_state.ravel()
        obs['desired_goal'] = self.goal.ravel()
        return obs

    def _set_action(self, stacked_action):
        logger.debug("set action")
        n_grippers = len(self.grippers)
        action = np.reshape(stacked_action, newshape=(int(len(stacked_action)/n_grippers), n_grippers))
        for i, key in enumerate(self.grippers):
            gripper = self.sim.getRigidBody(key)
            velocity = to_agx_list(action[:3, i], agx.Vec3)
            velocity *= 0.001  # from meters per second to millimeters per second
            gripper.setVelocity(velocity)
            angular_velocity = to_agx_list(action[3:, i], agx.Vec3)
            angular_velocity *= 0.5  # half of input rad/s
            gripper.setAngularVelocity(angular_velocity)

    def _is_success(self, achieved_goal, desired_goal):
        logger.debug("is success")
        distance_per_segment, __ = goal_distance(achieved_goal, desired_goal)
        return all(i < self.distance_threshold for i in distance_per_segment)

    def _sample_goal(self):
        logger.debug("sample goal")
        n_steps = 1000
        valid_goal = False
        while not valid_goal:
            # Define initial linear and angular velocities
            gripper_right = self.sim.getRigidBody('gripper_right')
            velocity_x = np.random.uniform(-0.002, 0)
            velocity_y = np.random.uniform(-0.002, 0.002)
            velocity_z = np.random.uniform(-0.002, 0.002)
            gripper_right.setVelocity(velocity_x, velocity_y, velocity_z)
            angular_velocity_y = np.random.uniform(-math.pi / 4, math.pi / 4)
            angular_velocity_z = np.random.uniform(-math.pi / 4, math.pi / 4)
            gripper_right.setAngularVelocity(0, angular_velocity_y, angular_velocity_z)

            largest = agxCable.SegmentDamage()
            for k in range(n_steps):
                angular_velocity = gripper_right.getAngularVelocity()
                gripper_right.setAngularVelocity(0, angular_velocity[1] * 0.99, angular_velocity[2] * 0.99)
                self.sim.stepForward()

                # Check for cable damage
                if not self.is_healthy:
                    logger.debug("Too much damage!")
                    break

            valid_goal = True

        cable = agxCable.Cable.find(self.sim, "DLO")
        goal = get_cable_state(cable)
        self._reset_sim()

        return goal.ravel()

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
