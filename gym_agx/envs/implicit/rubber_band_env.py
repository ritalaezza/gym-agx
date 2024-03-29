import sys
import logging
import os
import numpy as np

import agx
import agxCable
import agxOSG
import agxRender
from agxPythonModules.utils.numpy_utils import create_numpy_array
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z

from gym_agx.envs import agx_env
from gym_agx.rl.observation import get_cable_segment_positions, get_cable_segment_positions_and_velocities
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/rubber_band.agx")

GOAL_MAX_Z = 0.0125
POLE_OFFSET = [[0.0, 0.01], [-0.01, -0.01], [0.01, -0.01]]
MAX_X = 0.01
MAX_Y = 0.01


class RubberBandEnv(agx_env.AgxEnv):
    """Rubber band environment."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="state", headless=False,
                 image_size=[64, 64], **kwargs):
        """Initializes a RubberBandEnv object
        :param args: arguments for agxViewer
        :param scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step()
        :param end_effectors: list of EndEffector objects, defining controllable constraints
        :param observation_config: ObservationConfig object, defining the types of observations
        :param camera_config: dictionary containing EYE, CENTER, UP information for rendering, with lighting info
        :param reward_config: reward configuration object, defines success condition and reward function
        """

        self.reward_type = reward_type
        self.observation_type = observation_type
        self.segments_pos_old = None
        self.headless = headless

        camera_distance = 0.15  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, -0.1, camera_distance),
            center=agx.Vec3(0, 0, 0.0),
            up=agx.Vec3(0., 0., 0.0),
            light_position=agx.Vec4(0, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.))

        gripper = EndEffector(
            name='gripper',
            controllable=True,
            observable=True,
            max_velocity=0.4,  # m/s
            max_acceleration=0.2  # m/s^2
        )
        gripper.add_constraint(name='gripper_joint_base_x',
                               end_effector_dof=EndEffectorConstraint.Dof.X_TRANSLATION,
                               compute_forces_enabled=False,
                               velocity_control=True,
                               compliance_control=False)
        gripper.add_constraint(name='gripper_joint_base_y',
                               end_effector_dof=EndEffectorConstraint.Dof.Y_TRANSLATION,
                               compute_forces_enabled=False,
                               velocity_control=True,
                               compliance_control=False)
        gripper.add_constraint(name='gripper_joint_base_z',
                               end_effector_dof=EndEffectorConstraint.Dof.Z_TRANSLATION,
                               compute_forces_enabled=False,
                               velocity_control=True,
                               compliance_control=False)

        self.end_effectors = [gripper]

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

        super(RubberBandEnv, self).__init__(scene_path=SCENE_PATH,
                                            n_substeps=n_substeps,
                                            observation_type=observation_type,
                                            n_actions=3,
                                            image_size=image_size,
                                            camera_pose=camera_config.camera_pose,
                                            no_graphics=no_graphics,
                                            args=args)

    def render(self, mode="human"):
        return super(RubberBandEnv, self).render(mode)

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
            reward, goal_reached = self._compute_dense_reward_and_check_goal(segment_pos, self.segments_pos_old)
        else:
            goal_reached = self._is_goal_reached(segment_pos)
            reward = float(goal_reached)

        # Set old segment pos for next time step
        self.segments_pos_old = segment_pos

        info['is_success'] = goal_reached
        done = info['is_success']

        obs = self._get_observation()

        return obs, reward, done, info

    def reset(self):
        logger.info("reset")
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        # Wait several steps after initialization
        n_initial_wait = 10
        for k in range(n_initial_wait):
            self.sim.stepForward()

        n_initial_random = 10
        for k in range(n_initial_random):
            self._set_action(self.action_space.sample())
            self.sim.stepForward()

        # Set random obstacle position
        center_pos = np.random.uniform([-MAX_X, -MAX_Y], [MAX_X, MAX_Y])
        self._set_obstacle_center(center_pos)

        self.segments_pos_old = self._compute_segments_pos()

        obs = self._get_observation()

        return obs

    def _set_obstacle_center(self, center_pos):
        ground = self.sim.getRigidBody("ground")
        ground.setPosition(agx.Vec3(center_pos[0], center_pos[1], -0.005))

        for i in range(3):
            self.sim.getRigidBody("cylinder_low_" + str(i)).setPosition(agx.Vec3(center_pos[0] + POLE_OFFSET[i][0],
                                                                                 center_pos[1] + POLE_OFFSET[i][1],
                                                                                 0.0))

            self.sim.getRigidBody("cylinder_inner_" + str(i)).setPosition(agx.Vec3(center_pos[0] + POLE_OFFSET[i][0],
                                                                                   center_pos[1] + POLE_OFFSET[i][1],
                                                                                   0.005))

            self.sim.getRigidBody("cylinder_top_" + str(i)).setPosition(agx.Vec3(center_pos[0] + POLE_OFFSET[i][0],
                                                                                 center_pos[1] + POLE_OFFSET[i][1],
                                                                                 0.01))

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

    def _get_poles_enclosed(self, segments_pos):
        """Check how many poles the rubber band encloses
        :param segments_pos:
        :return:
        """
        poles_enclosed = np.zeros(3)
        ground_pos = self.sim.getRigidBody("ground").getPosition()
        pole_pos = np.array([ground_pos[0], ground_pos[1]]) + POLE_OFFSET
        for i in range(0, 3):
            segments_xy = np.array(segments_pos)[:, 0:2]
            is_within_polygon = point_inside_polygon(segments_xy, pole_pos[i])
            poles_enclosed[i] = int(is_within_polygon)

        return poles_enclosed

    def _compute_dense_reward_and_check_goal(self, segments_pos_0, segments_pos_1):
        """Compute reward for transition between two timesteps and check goal condition
        :return:
        """
        poles_enclosed_0 = self._get_poles_enclosed(segments_pos_0)
        poles_enclosed_1 = self._get_poles_enclosed(segments_pos_1)
        poles_enclosed_diff = poles_enclosed_0 - poles_enclosed_1

        # Check if final goal is reached
        is_correct_height = all_points_below_z(segments_pos_0, max_z=GOAL_MAX_Z)
        n_enclosed_0 = np.sum(poles_enclosed_0)
        final_goal_reached = n_enclosed_0 >= 3 and is_correct_height

        return np.sum(poles_enclosed_diff) + 5 * float(final_goal_reached), final_goal_reached

    def _is_goal_reached(self, segments_pos):
        """Goal is reached if the centers of all three poles are contained in the dlo polygon and the segments
        are below a certain height.
        :return:
        """
        n_enclosed = self._get_poles_enclosed(segments_pos)
        if np.sum(n_enclosed) >= 3 and all_points_below_z(segments_pos, max_z=GOAL_MAX_Z):
            return True
        return False

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
            name = rb.getName()
            if name == "ground":
                agxOSG.setDiffuseColor(node, agxRender.Color.SlateGray())
            elif name == "cylinder_top_0" or name == "cylinder_top_1" or name == "cylinder_top_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
            elif name == "cylinder_inner_0" or name == "cylinder_inner_1" or name == "cylinder_inner_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
            elif name == "cylinder_low_0" or name == "cylinder_low_1" or name == "cylinder_low_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
            elif name == "gripper":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkBlue())
            elif "dlo" in name:  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.8, 0.2, 0.2, 1.0))
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

        assert self.observation_type in ("rgb", "depth", "rgb_and_depth", "pos", "pos_and_vel")

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
        elif self.observation_type == "pos":
            goal_pos  = to_numpy_array(self.sim.getRigidBody("ground").getPosition())[0:2]
            seg_pos = get_cable_segment_positions(cable=agxCable.Cable.find(self.sim, "DLO")).flatten()
            gripper = self.sim.getRigidBody("gripper")
            gripper_pos = to_numpy_array(gripper.getPosition())[0:3]
            obs = np.concatenate([gripper_pos, seg_pos, goal_pos])

        elif self.observation_type == "pos_and_vel":
            goal_pos = to_numpy_array(self.sim.getRigidBody("ground").getPosition())[0:2]
            seg_pos, seg_vel = get_cable_segment_positions_and_velocities(cable=agxCable.Cable.find(self.sim, "DLO"))
            seg_pos = seg_pos.flatten()
            seg_vel = seg_vel.flatten()
            gripper = self.sim.getRigidBody("gripper")
            gripper_pos = to_numpy_array(gripper.getPosition())[0:3]
            gripper_vel = to_numpy_array(gripper.getVelocity())[0:3]

            obs = np.concatenate([gripper_pos, gripper_vel, seg_pos, seg_vel, goal_pos])

        return obs

    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info
