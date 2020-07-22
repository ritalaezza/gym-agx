import sys
import logging
import os
import numpy as np

import agx
import agxCable
import agxOSG
import agxRender
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z

from gym_agx.envs import agx_task_env
from gym_agx.rl.observation import get_cable_segment_positions
from gym_agx.rl.end_effector import EndEffector, EndEffectorConstraint
from gym_agx.utils.agx_classes import CameraConfig
from gym_agx.utils.agx_utils import to_numpy_array

logger = logging.getLogger('gym_agx.envs')

# Set paths
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
SCENE_PATH = os.path.join(PACKAGE_DIR, "assets/clip_closing.agx")

GOAL_MAX_Z = 0.0125
OBSTACLE_POSITIONS = [ [0.0,0.0], [0.075,0.075],[-0.075,0.075], [0.075,-0.075], [-0.075,-0.075]]


class ClipClosingEnv(agx_task_env.AgxTaskEnv):
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
            max_velocity= 3,  # m/s
            max_acceleration= 1 # m/s^2
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
            max_velocity= 3,  # m/s
            max_acceleration= 1 # m/s^2
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

        # TODO does -agxOnly made a difference?
        # # Disable rendering in headless mode
        # if headless:
        #     args.extend(["--osgWindow", False])
        #
        # if headless and observation_type == "state":
        #     args.extend(["-agxOnly", "--osgWindow", False])

        super(ClipClosingEnv, self).__init__(scene_path=SCENE_PATH,
                                             n_substeps=n_substeps,
                                             observation_config=None,
                                             n_actions=4,
                                             camera_pose=camera_config.camera_pose,
                                             args=args)

    def render(self, mode="human"):
        return super(ClipClosingEnv, self).render(mode)

    def step(self, action):
        logger.info("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
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

        # TODO Find good initialization strategy for this task which covers a larger area of the state space
        n_inital_random = 100
        for k in range(n_inital_random):
            if k==0 or not k%25:
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
        Goal is reached if the clip is closed around the center obstacle. This is the case if the segments
        are on the ground, the grippers are close to each other and the center pole is within the
        dlo polygon.
        :return:
        """

        # Check if goal obstacle in enclosed by dlo
        is_within_polygon = point_inside_polygon(np.array(segment_pos)[:,0:2], OBSTACLE_POSITIONS[0])

        # Check if clip has correct height
        is_correct_height = all_points_below_z(segment_pos, max_z=GOAL_MAX_Z)

        # Check if grippers are close enough to each other
        position_g0 = to_numpy_array(self.sim.getRigidBody("gripper_0").getPosition())
        position_g1 = to_numpy_array(self.sim.getRigidBody("gripper_1").getPosition())
        is_grippers_close = np.linalg.norm(position_g1-position_g0) < 0.01

        if is_within_polygon and is_correct_height and is_grippers_close:
            return True
        return False

    def _compute_dense_reward_and_check_goal(self, segments_pos_0, segments_pos_1):

        pole_enclosed_0 = point_inside_polygon(np.array(segments_pos_0)[:,0:2], OBSTACLE_POSITIONS[0])
        pole_enclosed_1 = point_inside_polygon(np.array(segments_pos_1)[:,0:2], OBSTACLE_POSITIONS[0])
        poles_enclosed_diff = pole_enclosed_0 - pole_enclosed_1

        # Check if final goal is reached
        is_correct_height = all_points_below_z(segments_pos_0, max_z=GOAL_MAX_Z)

        # Check if grippers are close enough to each other
        position_g0 = to_numpy_array(self.sim.getRigidBody("gripper_0").getPosition())
        position_g1 = to_numpy_array(self.sim.getRigidBody("gripper_1").getPosition())
        is_grippers_close = np.linalg.norm(position_g1-position_g0) < 0.01

        final_goal_reached = pole_enclosed_0 and is_correct_height and is_grippers_close

        return poles_enclosed_diff + 5*float(final_goal_reached), final_goal_reached

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
            elif "dlo" in  rb.getName():  # Cable segments
                agxOSG.setDiffuseColor(node, agxRender.Color(0.1, 0.5, 0.0, 1.0))
                agxOSG.setAmbientColor(node, agxRender.Color(0.2, 0.5, 0.0, 1.0))
            elif rb.getName() == "obstacle":
                agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
            elif rb.getName() == "obstacle_center":
                agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
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
        gripper_0 = self.sim.getRigidBody("gripper_0")
        gripper_1 = self.sim.getRigidBody("gripper_1")
        gripper_0_pos = to_numpy_array(gripper_0.getPosition())[0:2]
        gripper_1_pos = to_numpy_array(gripper_1.getPosition())[0:2]

        obs = np.concatenate([gripper_0_pos, gripper_1_pos, seg_pos])

        return obs

    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info
