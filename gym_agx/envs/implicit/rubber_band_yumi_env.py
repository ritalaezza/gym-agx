import sys
import logging
import os
import numpy as np

import agx
import agxCable
import agxOSG
import agxRender

from gym_agx.envs import agx_env
from gym_agx.rl.observation import get_cable_segment_positions, get_cable_segment_positions_and_velocities
from gym_agx.utils.agx_classes import CameraConfig, ContactEventListenerRigidBody
from gym_agx.utils.agx_utils import to_numpy_array
from agxPythonModules.utils.numpy_utils import create_numpy_array
from gym_agx.utils.agx_utils import add_collision_events
from gym_agx.rl.joint_entity import JointObjects, JointConstraint
from gym_agx.rl.observation import get_joint_velocity, get_joint_position
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/rubber_band_yumi.agx")

# Meshes and textures
TEXTURE_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/textures/texture_gripper.png")
MESH_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_gripper.obj")
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_hollow_cylinder.obj")

SCALE = 5  # scale up from original rubber band env
MAX_X = 0.01 * SCALE
MAX_Y = 0.01 * SCALE
POLE_OFFSETS = [[0.3 + 0.0*SCALE, 0.01*SCALE], [0.3 + -0.01*SCALE, -0.01*SCALE], [0.3 + 0.01*SCALE, -0.01*SCALE]]
GOAL_MAX_Z = 0.0125 * SCALE

JOINT_NAMES_REV = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                   'yumi_joint_5_r', 'yumi_joint_6_r']  # names of controllable the revolute joints

class RubberBandYuMiEnv(agx_env.AgxEnv):
    """Peg-in-hole environment."""

    def __init__(self, n_substeps=10, reward_type="dense", observation_type="state", headless=False, image_size=[64, 64], joints=None,
                 **kwargs):
        """Initializes a PegInHoleEnv object
        :param args: arguments for agxViewer.
        :param scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step().
        :param end_effectors: list of EndEffector objects, defining controllable constraints.
        :param observation_config: ObservationConfig object, defining the types of observations.
        :param camera_config: dictionary containing EYE, CENTER, UP information for rendering, with lighting info.
        :param reward_config: reward configuration object, defines success condition and reward function.
        """
        # links to do collision detection event for
        self.link_names = ['gripper_r_finger_r', 'gripper_r_finger_l', 'pusher']
        self.contact_names = ['contact_' + link_name for link_name in self.link_names]
        self.ignore_names = ['dlo']  # geometry names to ignore from collision detection event

        self.reward_type = reward_type
        self.segment_pos_old = None
        self.n_segments = None
        self.headless = headless

        camera_distance = 0.1  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(1, 0.2, 0.2),
            center=agx.Vec3(0.3, 0, 0.1),
            up=agx.Vec3(0., 0., 1.0),
            light_position=agx.Vec4(0, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.))

        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv

        # Change window size
        args.extend(["--window", "400", "600"])

        no_graphics = headless and observation_type not in ("rgb", "depth", "rgb_and_depth")

        # Disable rendering in headless mode
        if headless:
            args.extend(["--osgWindow", "0"])

        if headless and observation_type == "gt":
            # args.extend(["--osgWindow", "0"])
            args.extend(["--agxOnly", "1", "--osgWindow", "0"])

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

        self.end_effectors = joints
        n_actions = 0

        for end_effector in self.end_effectors:
            if end_effector.controllable:
                for key, constraint in end_effector.constraints.items():
                    if constraint.velocity_control:
                        n_actions += 1
                    if constraint.compliance_control:
                        n_actions += 1

        super(RubberBandYuMiEnv, self).__init__(scene_path=SCENE_PATH,
                                           n_substeps=n_substeps,
                                           observation_type=observation_type,
                                           n_actions=n_actions,
                                           camera_pose=camera_config.camera_pose,
                                           no_graphics=no_graphics,
                                           image_size=image_size,
                                           args=args)

    def render(self, mode="human"):
        return super(RubberBandYuMiEnv, self).render(mode)

    def step(self, action):
        logger.info("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #action_mutliplier = np.array([0.2, 0.2, 0.4, 3])
        #action *= action_mutliplier
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
            add_collision_events(self.link_names, self.contact_names, self.ignore_names, ContactEventListenerRigidBody,
                                 self.sim)
        # Wait several steps after initalization
        n_inital_wait = 10
        for k in range(n_inital_wait):
            self.sim.stepForward()

        n_inital_random = 10
        for k in range(n_inital_random):
            self._set_action(self.action_space.sample())
            self.sim.stepForward()

        # Set random obstacle position
        center_pos = np.random.uniform([-MAX_X, -MAX_Y], [MAX_X, MAX_Y])
        self._set_obstacle_center(center_pos)

        self.segment_pos_old = self._compute_segments_pos()

        obs = self._get_observation()
        return obs

    def _set_obstacle_center(self, center_pos):

        for i in range(3):
            self.sim.getRigidBody("cylinder_low_" + str(i)).setPosition(agx.Vec3(center_pos[0] + POLE_OFFSETS[i][0],
                                                                            center_pos[1] + POLE_OFFSETS[i][1],
                                                                            0.0))

            self.sim.getRigidBody("cylinder_inner_" + str(i)).setPosition(
                agx.Vec3(center_pos[0] + POLE_OFFSETS[i][0],
                         center_pos[1] + POLE_OFFSETS[i][1],
                         0.005 * SCALE))

            self.sim.getRigidBody("cylinder_top_" + str(i)).setPosition(agx.Vec3(center_pos[0] + POLE_OFFSETS[i][0],
                                                                            center_pos[1] + POLE_OFFSETS[i][1],
                                                                            0.01 * SCALE))

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
        """
        Check how many poles the rubber band encloses
        :param segments_pos:
        :return:
        """
        poles_enclosed = np.zeros(3)
        ground_pos = self.sim.getRigidBody("ground").getPosition()
        pole_pos = np.array( [ground_pos[0], ground_pos[1]]) + POLE_OFFSETS
        for i in range(0, 3):
            segments_xy = np.array(segments_pos)[:, 0:2]
            is_within_polygon = point_inside_polygon(segments_xy, pole_pos[i])
            poles_enclosed[i] = int(is_within_polygon)

        return poles_enclosed

    def _compute_dense_reward_and_check_goal(self, segments_pos_0, segments_pos_1):
        """
        Compute reward for transition between two timesteps and check goal condition
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
        """
        Goal is reached if the centers of all three poles are contained in the dlo polygon and the segments
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

        file_directory = os.path.dirname(os.path.abspath(__file__))
        package_directory = os.path.split(file_directory)[0]
        gripper_texture = os.path.join(package_directory, TEXTURE_GRIPPER_FILE)

        # Create scene graph for rendering
        root = self.app.getSceneRoot()
        rbs = self.sim.getRigidBodies()
        for rb in rbs:
            node = agxOSG.createVisual(rb, root)
            if rb.getName() == "hollow_cylinder":
                agxOSG.setDiffuseColor(node, agxRender.Color_SteelBlue())
                agxOSG.setShininess(node, 15)
            elif rb.getName() == "cylinder_top_0" or rb.getName() == "cylinder_top_1" or rb.getName() == "cylinder_top_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
            elif rb.getName() == "cylinder_inner_0" or rb.getName() == "cylinder_inner_1" or rb.getName() == "cylinder_inner_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
            elif rb.getName() == "cylinder_low_0" or rb.getName() == "cylinder_low_1" or rb.getName() == "cylinder_low_2":
                agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
            elif "dlo" in rb.getName():  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
            elif "ground" in rb.getName():
                agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
            else:
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                agxOSG.setAlpha(node, 0.0)

        # Set rendering options
        scene_decorator = self.app.getSceneDecorator()
        scene_decorator.setEnableLogo(False)
        scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0, 1.0, 1.0))

    def _get_observation(self):
        # TODO update observation
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
            seg_pos = get_cable_segment_positions(cable=agxCable.Cable.find(self.sim, "DLO")).flatten()


            # get joint position
            joint_position = []
            for joint_entity in self.end_effectors:
                if joint_entity.observable and joint_entity.type == 'JointObject':
                    for key, constraint in joint_entity.constraints.items():
                        joint_position.append(get_joint_position(self.sim, key))
            joint_position = np.hstack(joint_position)

            obs = np.concatenate([joint_position, seg_pos])

        elif self.observation_type == "pos_and_vel":
            seg_pos, seg_vel = get_cable_segment_positions_and_velocities(cable=agxCable.Cable.find(self.sim, "DLO"))
            seg_pos = seg_pos.flatten()
            seg_vel = seg_vel.flatten()


            # get joint position
            joint_position = []
            for joint_entity in self.end_effectors:
                if joint_entity.observable and joint_entity.type == 'JointObject':
                    for key, constraint in joint_entity.constraints.items():
                        joint_position.append(get_joint_position(self.sim, key))
            joint_position = np.hstack(joint_position)

            # get joint vel
            joint_velocity = []
            for joint_entity in self.end_effectors:
                if joint_entity.observable and joint_entity.type == 'JointObject':
                    for key, constraint in joint_entity.constraints.items():
                        joint_velocity.append(get_joint_velocity(self.sim, key))
            joint_velocity = np.hstack(joint_velocity)
            obs = np.concatenate([joint_position, joint_velocity, seg_pos, seg_vel])

        return obs


    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info
