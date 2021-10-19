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

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/peg_in_hole_yumi.agx")

# Meshes and textures
TEXTURE_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/textures/texture_gripper.png")
MESH_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_gripper.obj")
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_hollow_cylinder.obj")

JOINT_NAMES_REV = ['yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                   'yumi_joint_5_r', 'yumi_joint_6_r']  # names of controllable the revolute joints

class PegInHoleYuMiEnv(agx_env.AgxEnv):
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

        super(PegInHoleYuMiEnv, self).__init__(scene_path=SCENE_PATH,
                                           n_substeps=n_substeps,
                                           observation_type=observation_type,
                                           n_actions=n_actions,
                                           camera_pose=camera_config.camera_pose,
                                           no_graphics=no_graphics,
                                           image_size=image_size,
                                           args=args)

    def render(self, mode="human"):
        return super(PegInHoleYuMiEnv, self).render(mode)

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

        # Randomly initialize cylinder position
        cylinder_pos_new = np.random.uniform([0.2, -0.15], [0.35, 0.15])
        self.sim.getRigidBody("hollow_cylinder").setPosition(agx.Vec3(cylinder_pos_new[0], cylinder_pos_new[1], 0.05))

        cable = agxCable.Cable.find(self.sim, "DLO")
        self.n_segments = cable.getNumSegments()
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
        Checks if positions of cable segments on lower end are within goal region. Returns True if cable is partially
        inserted and False otherwise.
        """

        cylinder_pos = self.sim.getRigidBody("hollow_cylinder").getPosition()
        for p in segment_pos[int(self.n_segments / 2):]:
                # Return False if a segment is ouside bounds
                if not (cylinder_pos[0]-0.015 <= p[0] <= cylinder_pos[0]+0.015 and
                        cylinder_pos[1]-0.015 <= p[1] <= cylinder_pos[1]+0.015 and
                        cylinder_pos[2]-0.1 <= p[2] <=cylinder_pos[2]+0.07):
                    return False

        return True

    def _determine_n_segments_inserted(self, segment_pos, cylinder_pos):
        """
        Determine number of segments that are inserted into the hole.
        :param segment_pos:
        :return:
        """
        n_inserted = 0
        for p in segment_pos:
            if (cylinder_pos[0]-0.015 <= p[0] <= cylinder_pos[0]+0.015 and
                    cylinder_pos[1]-0.015 <= p[1] <= cylinder_pos[1]+0.015 and
                    -0.1 <= p[2] <=0.07):
                n_inserted += 1
        return n_inserted

    def _compute_dense_reward_and_check_goal(self, segment_pos_0, segment_pos_1):
        cylinder_pos = self.sim.getRigidBody("hollow_cylinder").getPosition()
        n_segs_inserted_0 = self._determine_n_segments_inserted(segment_pos_0, cylinder_pos)
        n_segs_inserted_1 = self._determine_n_segments_inserted(segment_pos_1, cylinder_pos)
        n_segs_inserted_diff = n_segs_inserted_0 - n_segs_inserted_1

        # Check if final goal is reached
        final_goal_reached = n_segs_inserted_0 >= self.n_segments / 2

        return np.sum(n_segs_inserted_diff) + 5 * float(final_goal_reached), final_goal_reached

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

            cylinder = self.sim.getRigidBody("hollow_cylinder")
            cylinder_pos = cylinder.getPosition()

            # get joint position
            joint_position = []
            for joint_entity in self.end_effectors:
                if joint_entity.observable and joint_entity.type == 'JointObject':
                    for key, constraint in joint_entity.constraints.items():
                        joint_position.append(get_joint_position(self.sim, key))
            joint_position = np.hstack(joint_position)

            obs = np.concatenate([joint_position, seg_pos, [cylinder_pos[0], cylinder_pos[1]]])

        elif self.observation_type == "pos_and_vel":
            seg_pos, seg_vel = get_cable_segment_positions_and_velocities(cable=agxCable.Cable.find(self.sim, "DLO"))
            seg_pos = seg_pos.flatten()
            seg_vel = seg_vel.flatten()
            cylinder = self.sim.getRigidBody("hollow_cylinder")
            cylinder_pos = cylinder.getPosition()

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
            obs = np.concatenate([joint_position, joint_velocity, seg_pos, seg_vel, [cylinder_pos[0], cylinder_pos[1]]])

        return obs


    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info
