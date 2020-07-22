import sys
import logging
import os
import numpy as np

import agx
import agxCable
import agxOSG
import agxRender

from gym_agx.envs import agx_task_env
from gym_agx.rl.observation import get_cable_segment_positions
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/peg_in_hole.agx")

# Meshes and textures
TEXTURE_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/textures/texture_gripper.png")
MESH_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_gripper.obj")
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "assets/meshes/mesh_hollow_cylinder.obj")


class PegInHoleEnv(agx_task_env.AgxTaskEnv):
    """Superclass for all DLO environments."""

    def __init__(self, n_substeps=1, reward_type="dense", observation_type="state", headless=False, **kwargs):
        """Initializes a DloEnv object
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
        self.n_segments = None

        camera_distance = 0.1  # meters
        camera_config = CameraConfig(
            eye=agx.Vec3(0, -0.1, 0.05),
            center=agx.Vec3(0, 0, 0.02),
            up=agx.Vec3(0., 0., 1.0),
            light_position=agx.Vec4(0, - camera_distance, camera_distance, 1.),
            light_direction=agx.Vec3(0., 0., -1.))

        if 'agxViewer' in kwargs:
            args = sys.argv + kwargs['agxViewer']
        else:
            args = sys.argv

        # Change window size
        args.extend(["--window", "400", "600"])

        # TODO does -agxOnly made a difference?
        # # Disable rendering in headless mode
        # if headless:
        #     args.extend(["--osgWindow", False])
        #
        # if headless and observation_type == "state":
        #     args.extend(["-agxOnly", "--osgWindow", False])

        super(PegInHoleEnv, self).__init__(scene_path=SCENE_PATH,
                                           n_substeps=n_substeps,
                                           observation_config=None,
                                           n_actions=4,
                                           camera_pose=camera_config.camera_pose,
                                           args=args)

    def render(self, mode="human"):
        return super(PegInHoleEnv, self).render(mode)

    def step(self, action):
        logger.info("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action[0:3] *= 0.2
        action[3] *= 1
        info = self._set_action(action)

        self._step_callback()

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

        # Wait several steps after initalization
        n_inital_wait = 10
        for k in range(n_inital_wait):
            self.sim.stepForward()

        # Randomize starting configuration
        n_steps_rand = 50
        action = np.random.uniform(-0.05,0.05,4)
        for k in range(n_steps_rand):
            self.sim.getConstraint1DOF("gripper_joint_base_x").getMotor1D().setSpeed(action[0])
            self.sim.getConstraint1DOF("gripper_joint_base_y").getMotor1D().setSpeed(action[1])
            self.sim.getConstraint1DOF("gripper_joint_base_z").getMotor1D().setSpeed(action[2])
            self.sim.getConstraint1DOF("gripper_joint_rot_y").getMotor1D().setSpeed(action[3]*50)
            self.sim.stepForward()

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

        for p in segment_pos[int(self.n_segments/2):]:
            # Return False if segment is ouside bounds
            if not (-0.003 <= p[0] <= 0.003 and -0.003 <= p[1] <= 0.003 and -0.01 <= p[2] <=0.006):
                return False
        return True

    def _determine_n_segments_inserted(self, segment_pos):
        """
        Determine number of segments that are inserted into the hole.
        :param segment_pos:
        :return:
        """
        n_inserted = 0
        for p in segment_pos:
            # Return False if segment is ouside bounds
            if -0.003 <= p[0] <= 0.003 and -0.003 <= p[1] <= 0.003 and -0.01 <= p[2] <=0.006:
                n_inserted +=1
        return n_inserted

    def _compute_dense_reward_and_check_goal(self, segment_pos_0, segment_pos_1):
        n_segs_inserted_0 = self._determine_n_segments_inserted(segment_pos_0)
        n_segs_inserted_1 = self._determine_n_segments_inserted(segment_pos_1)
        n_segs_inserted_diff = n_segs_inserted_0 - n_segs_inserted_1

        # Check if final goal is reached
        final_goal_reached = n_segs_inserted_0 >= self.n_segments/2

        return np.sum(n_segs_inserted_diff) + 5*float(final_goal_reached), final_goal_reached

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
                agxOSG.setDiffuseColor(node,agxRender.Color_SteelBlue())
                agxOSG.setShininess(node, 15)
            elif rb.getName() == "gripper_body":
                agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 1.0, 1.0, 1.0))
                agxOSG.setTexture(node, gripper_texture, False, agxOSG.DIFFUSE_TEXTURE)
                agxOSG.setShininess(node,2)
            elif "dlo" in  rb.getName():  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
            else:
                agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
                agxOSG.setAlpha(node, 0.0)

        # Set rendering options
        scene_decorator = self.app.getSceneDecorator()
        scene_decorator.setEnableLogo(False)
        scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0,1.0, 1.0))

    def _get_observation(self):
        # TODO use modular strucutre for observations and allow different type of observations
        seg_pos = get_cable_segment_positions(cable=agxCable.Cable.find(self.sim, "DLO")).flatten()
        gripper = self.sim.getRigidBody("gripper_body")
        gripper_pos = to_numpy_array(gripper.getPosition())[0:3]
        ea = agx.EulerAngles().set(gripper.getRotation())
        gripper_rot = ea.y()

        obs = np.concatenate([gripper_pos, [gripper_rot], seg_pos])

        return obs

    def _set_action(self, action):
        info = dict()
        action = np.array(action, dtype=np.double)
        self.sim.getConstraint1DOF("gripper_joint_base_x").getMotor1D().setSpeed(action[0])
        self.sim.getConstraint1DOF("gripper_joint_base_y").getMotor1D().setSpeed(action[1])
        self.sim.getConstraint1DOF("gripper_joint_base_z").getMotor1D().setSpeed(action[2])
        self.sim.getConstraint1DOF("gripper_joint_rot_y").getMotor1D().setSpeed(action[3])

        return info
