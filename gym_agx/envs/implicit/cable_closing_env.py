import sys
import logging
import os
import numpy as np

import agx
import agxCable
import agxOSG
import agxRender
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z

from gym_agx.envs import agx_env
from gym_agx.rl.observation import get_cable_segment_positions
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array
from agxPythonModules.utils.numpy_utils import create_numpy_array

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/cable_closing.agx")

GOAL_MAX_Z = 0.0125
OBSTACLE_POSITIONS = [[0.0, 0.0], [0.075, 0.075], [-0.075, 0.075], [0.075, -0.075], [-0.075, -0.075]]


class CableClosingEnv(agx_env.AgxEnv):
    """Cable closing environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="state", headless=False, **kwargs):
        """Initializes a CableClosingEnv object
        :param args: arguments for agxViewer.
        :param scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step().
        :param end_effectors: list of EndEffector objects, defining controllable constraints.
        :param observation_config: ObservationConfig object, defining the types of observations.
        :param camera_config: dictionary containing EYE, CENTER, UP information for rendering, with lighting info.
        :param reward_config: reward configuration object, defines success condition and reward function.
        """

        self.reward_type = reward_type
        self.segment_pos_old = None
        self.headless = headless

        camera_distance = 0.1  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, 0.0, 0.65),
            center=agx.Vec3(0, 0, 0.0),
            up=agx.Vec3(0., 0., 0.0),
            light_position=agx.Vec4(0, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.))

        gripper_0 = EndEffector(
            name='gripper_0',
            controllable=True,
            observable=True,
            max_velocity=3,  # m/s
            max_acceleration=3  # m/s^2
        )
        gripper_0.add_constraint(name='gripper_0_joint_base_x',
                                 end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                 compute_forces_enabled=False,
                                 velocity_control=True,
                                 compliance_control=False)
        gripper_0.add_constraint(name='gripper_0_joint_base_y',
                                 end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                                 compute_forces_enabled=False,
                                 velocity_control=True,
                                 compliance_control=False)

        gripper_1 = EndEffector(
            name='gripper_1',
            controllable=True,
            observable=True,
            max_velocity=3,  # m/s
            max_acceleration=3 # m/s^2
        )
        gripper_1.add_constraint(name='gripper_1_joint_base_x',
                                 end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                                 compute_forces_enabled=False,
                                 velocity_control=True,
                                 compliance_control=False)
        gripper_1.add_constraint(name='gripper_1_joint_base_y',
                                 end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                                 compute_forces_enabled=False,
                                 velocity_control=True,
                                 compliance_control=False)

        self.end_effectors = [gripper_0, gripper_1]

        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv

        # Change window size
        args.extend(["--window", "600", "600"])

        no_graphics = headless and observation_type not in ("rgb", "depth", "rgb_and_depth")

        # Disable rendering in headless mode
        if headless:
            args.extend(["--osgWindow", "0"])

        if headless and observation_type == "gt":
            # args.extend(["--osgWindow", "0"])
            args.extend(["--agxOnly", "1", "--osgWindow", "0"])

        super(CableClosingEnv, self).__init__(scene_path=SCENE_PATH,
                                              n_substeps=n_substeps,
                                              observation_type=observation_type,
                                              n_actions=4,
                                              camera_pose=camera_config.camera_pose,
                                              image_size=(64, 64),
                                              no_graphics=no_graphics,
                                              args=args)

    def render(self, mode="human"):
        return super(CableClosingEnv, self).render(mode)

    def step(self, action):
        logger.info("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        info = self._set_action(action)
        self._step_callback()

        if not self.headless or self.observation_type in ("rgb", "depth", "rgb_and_depth"):
            self._render_callback()

        # Get segments positions
        segment_pos = self._compute_segments_pos()

        # Compute rewards
        if self.reward_type == "dense":
            reward, goal_reached = self._compute_dense_reward_and_check_goal(segment_pos, self.segment_pos_old)
        else:
            goal_reached = self._is_goal_reached(segment_pos)
            reward = float(goal_reached)

        # Set old segment pos for next time step
        self.segment_pos_old = segment_pos

        info['is_success'] = goal_reached
        done = info['is_success']

        obs = self._get_observation()

        return obs, reward, done, info

    def reset(self):
        logger.info("reset")
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        # Randomly initialize cylinder position
        goal_pos_new = np.random.uniform([-0.1, -0.1], [0.1, -0.025])
        self.sim.getRigidBody("obstacle_goal").setPosition(agx.Vec3(goal_pos_new[0], goal_pos_new[1] ,0.005))

        # TODO Find good initialization strategy for this task which covers a larger area of the state space
        n_inital_random = 50
        for k in range(n_inital_random):
            if k == 0 or not k % 25:
                action = self.action_space.sample()
            self._set_action(action)
            self.sim.stepForward()

        # Wait several steps after initalization
        n_inital_wait = 10
        for k in range(n_inital_wait):
            self.sim.stepForward()

        self.segment_pos_old = self._compute_segments_pos()

        obs = self._get_observation()
        return obs

    def _compute_segments_pos(self):
        segments_pos = []
        dlo = agxCable.Cable.find(self.sim, "DLO")
        segment_iterator = dlo.begin()
        n_segments = dlo.getNumSegments()
        for i in range(n_segments):
            if not segment_iterator.isEnd():
                pos = segment_iterator.getGeometry().getPosition()
                segments_pos.append(to_numpy_array(pos))
                segment_iterator.inc()

        return segments_pos

    def _is_goal_reached(self, segment_pos):
        """
        Goal is reached if the cable is closed around the center obstacle. This is the case if the segments
        are on the ground, the grippers are close to each other and the center pole is within the
        dlo polygon.
        :return:
        """

        # Get position of goal
        goal_pos = self.sim.getRigidBody("obstacle_goal").getPosition()

        # Check if goal obstacle in enclosed by dlo
        is_within_polygon = point_inside_polygon(np.array(segment_pos)[:, 0:2], goal_pos)

        # Check if cable has correct height
        is_correct_height = all_points_below_z(segment_pos, max_z=GOAL_MAX_Z)

        # Check if grippers are close enough to each other
        position_g0 = to_numpy_array(self.sim.getRigidBody("gripper_0").getPosition())
        position_g1 = to_numpy_array(self.sim.getRigidBody("gripper_1").getPosition())
        is_grippers_close = np.linalg.norm(position_g1 - position_g0) < 0.01

        if is_within_polygon and is_correct_height and is_grippers_close:
            return True
        return False

    def _compute_dense_reward_and_check_goal(self, segments_pos_0, segments_pos_1):

        # Get position of goal
        goal_pos = self.sim.getRigidBody("obstacle_goal").getPosition()

        pole_enclosed_0 = point_inside_polygon(np.array(segments_pos_0)[:, 0:2], goal_pos)
        pole_enclosed_1 = point_inside_polygon(np.array(segments_pos_1)[:, 0:2], goal_pos)
        poles_enclosed_diff = pole_enclosed_0 - pole_enclosed_1

        # Check if final goal is reached
        is_correct_height = all_points_below_z(segments_pos_0, max_z=GOAL_MAX_Z)

        # Check if grippers are close enough to each other
        position_g0 = to_numpy_array(self.sim.getRigidBody("gripper_0").getPosition())
        position_g1 = to_numpy_array(self.sim.getRigidBody("gripper_1").getPosition())
        is_grippers_close = np.linalg.norm(position_g1 - position_g0) < 0.01

        final_goal_reached = pole_enclosed_0 and is_correct_height and is_grippers_close

        return poles_enclosed_diff + 5 * float(final_goal_reached), final_goal_reached

    def _add_rendering(self, mode='osg'):
        # Set renderer
        self.app.setAutoStepping(True)
        self.app.setEnableDebugRenderer(False)
        self.app.setEnableOSGRenderer(True)

        # Create scene graph for rendering
        root = self.app.getSceneRoot()
        rbs = self.sim.getRigidBodies()
        for rb in rbs:
            node = agxOSG.createVisual(rb, root)
            if rb.getName() == "ground":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.8, 0.8, 0.8, 1.0))
            elif rb.getName() == "walls":
                agxOSG.setDiffuseColor(node, agxRender.Color.Burlywood())
            elif rb.getName() == "cylinder":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
            elif rb.getName() == "cylinder_inner":
                agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
            elif rb.getName() == "gripper_0" or rb.getName() == "gripper_1":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.1, 0.1, 0.1, 1.0))
            elif "dlo" in rb.getName():  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.1, 0.5, 0.0, 1.0))
                agxOSG.setAmbientColor(node, agxRender.Color(0.2, 0.5, 0.0, 1.0))
            elif rb.getName() == "obstacle":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.5, 0.5, 0.5, 1.0))
            elif rb.getName() == "obstacle_goal":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
            else:
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                agxOSG.setAlpha(node, 0.0)

        # Set rendering options
        scene_decorator = self.app.getSceneDecorator()
        scene_decorator.setEnableLogo(False)
        scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0, 1.0, 1.0))

    def _get_observation(self):
        rgb_buffer = None
        depth_buffer = None
        for buffer in self.render_to_image:
            name = buffer.getName()
            if name == 'rgb_buffer':
                rgb_buffer = buffer
            elif name == 'depth_buffer':
                depth_buffer = buffer

        if self.observation_type == "rgb":
            image_ptr = rgb_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
            obs = np.flipud(image_data)
        elif self.observation_type == "depth":
            image_ptr = depth_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1]), np.float32)
            obs = np.flipud(image_data)
        elif self.observation_type == "rgb_and_depth":

            obs = np.zeros((self.image_size[0], self.image_size[1], 4), dtype=np.float32)

            image_ptr = rgb_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
            obs[:, :, 0:3] = np.flipud(image_data.astype(np.float32)) / 255

            image_ptr = depth_buffer.getImageData()
            image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1]), np.float32)
            obs[:, :, 3] = np.flipud(image_data)
        else:
            seg_pos = get_cable_segment_positions(cable=agxCable.Cable.find(self.sim, "DLO")).flatten()
            gripper_0 = self.sim.getRigidBody("gripper_0")
            gripper_1 = self.sim.getRigidBody("gripper_1")
            obstacle_goal = self.sim.getRigidBody("obstacle_goal")
            gripper_0_pos = to_numpy_array(gripper_0.getPosition())[0:2]
            gripper_1_pos = to_numpy_array(gripper_1.getPosition())[0:2]
            goal_pos = to_numpy_array(obstacle_goal.getPosition())[0:2]

            obs = np.concatenate([gripper_0_pos, gripper_1_pos, seg_pos, goal_pos])

        return obs

    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info
