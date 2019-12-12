import sys
import math
import logging
import numpy as np

import agx
import agxIO
import agxSDK
import agxCable

from gym_agx.envs import agx_env
from gym_agx.utils.agx_utils import add_wire_rendering, get_cable_state, to_numpy_array, to_agx_list


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


logger = logging.getLogger(__name__)


class WireEnv(agx_env.AgxEnv):
    def __init__(self, scene_path, n_substeps, grippers, length, n_actions, camera, args, distance_threshold,
                 reward_type):
        """Superclass for all Wire environments.

        Args:
            n_substeps (int): number of substeps the simulation runs on every call to step
            n_actions (int): number of DoF of kinematic object
            distance_threshold (float): the threshold after which a goal is considered achieved
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        """
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.grippers = grippers
        self.length = length
        self.camera = camera
        self.args = args  # TODO: may want to move this to parent class

        super(WireEnv, self).__init__(scene_path=scene_path, n_substeps=n_substeps, grippers=grippers,
                                      n_actions=n_actions)

        self.app, self.root = add_wire_rendering(self.sim, self.length)
        self.app.init(agxIO.ArgumentParser([sys.executable] + self.args))
        self.app.setCameraHome(self.camera['eye'], self.camera['center'], self.camera['up'])  # only after app.init
        self.app.initSimulation(self.sim, True)

    def render(self, mode='osg'):
        return super(WireEnv, self).render(mode)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # AgxEnv methods
    # ----------------------------

    def _set_action(self, action):
        print("set action")
        for i, key in enumerate(self.grippers):
            gripper = self.sim.getRigidBody(key)
            velocity = to_agx_list(action[:3, i], agx.Vec3)
            gripper.setVelocity(velocity)
            angular_velocity = to_agx_list(action[3:, i], agx.Vec3)
            gripper.setAngularVelocity(angular_velocity)

    def _reset_sim(self):
        print("reset sim")
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_ALL):
            print("Unable to restore simulation!")
            return False

        return True

    def _get_obs(self):
        print("get obs")
        obs = dict.fromkeys({'achieved_goal', 'observation'}, None)

        cable = agxCable.Cable.find(self.sim, "DLO")
        cable_state = get_cable_state(cable)

        gripper_state = np.zeros(shape=(7, len(self.grippers)), dtype=float)
        for i, key in enumerate(self.grippers):
            gripper = self.sim.getRigidBody(key)
            gripper_state[:3, i] = to_numpy_array(gripper.getPosition())
            gripper_state[3:, i] = to_numpy_array(gripper.getRotation())

        obs['achieved_goal'] = cable_state
        obs['observation'] = np.concatenate((gripper_state, cable_state), axis=1)
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _sample_goal(self):
        print("sample goal")
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
                cable = agxCable.Cable.find(self.sim, "DLO")
                cable_damage = agxCable.CableDamage.getCableDamage(cable)
                damages = cable_damage.getCurrentDamages()

                for d in damages:
                    for e in range(agxCable.NUM_CABLE_DAMAGE_TYPES):
                        largest[e] = max(largest[e], d[e])
                if largest.total() > 1e7:
                    print("Too much damage!")
                    break

            valid_goal = True

        goal = get_cable_state(cable)
        self._reset_sim()

        return goal.copy()

    def _render_callback(self):
        print("render callback")

    def _step_callback(self):
        print("step callback")
