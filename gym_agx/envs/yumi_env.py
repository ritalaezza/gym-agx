import logging

import agxIO
import agxSDK
import agxOSG
import agxRender

from gym_agx.envs import agx_goal_env

logger = logging.getLogger('gym_agx.envs')


class YuMiEnv(agx_goal_env.AgxGoalEnv):
    """Superclass for all explicit shape control environments with DLOs."""

    def __init__(self, args, scene_path, n_substeps, controllable_joints, observation_config, camera_config, reward_config,
                 randomized_goal, goal_scene_path, show_goal, osg_window=True, agx_only=False):
        """Initializes a Dlo object
        :param list args: arguments for agxViewer.
        :param str scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/
        folder
        :param int n_substeps: number os simulation steps per call to step().
        :param list controllable_joints: list of controllable joints, defining controllable constraints.
        :param gym_agx.rl.observation.ObservationConfig observation_config: ObservationConfig object, defining the types
        of observations.
        :param gym_agx.utils.agx_classes.CameraConfig camera_config: dictionary containing EYE, CENTER, UP information
        for rendering, with lighting info.
        :param gym_agx.rl.reward.RewardConfig reward_config: reward configuration object, defines success condition and
        reward function.
        :param bool randomized_goal: boolean deciding if a new goal is sampled for each episode.
        :param str goal_scene_path: path to goal scene file.
        :param bool show_goal: boolean determining whether goal is rendered or not.
        :param bool osg_window: boolean which enables/disables window rendering (useful for training).
        :param bool agx_only: boolean which disables all rendering, including for observations of type RGB or depth
        images.
        """
        self.goal_scene_path = goal_scene_path
        self.randomized_goal = randomized_goal
        self.reward_config = reward_config
        self.light_pose = camera_config.light_pose
        self.controllable_joints = controllable_joints
        self.show_goal = show_goal

        n_actions = 0
        for controllable_joint in controllable_joints:
            if controllable_joint.controllable:
                for key, constraint in controllable_joint.constraints.items():
                    if constraint.velocity_control:
                        n_actions += 1
                    if constraint.compliance_control:
                        n_actions += 1

        super(YuMiEnv, self).__init__(scene_path=scene_path, n_substeps=n_substeps, n_actions=n_actions,
                                     observation_config=observation_config, camera_pose=camera_config.camera_pose,
                                     osg_window=osg_window, agx_only=agx_only, args=args)

    def render(self, mode="human"):
        return super(YuMiEnv, self).render(mode)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_config.compute_reward(achieved_goal, desired_goal, info)

    # AgxEnv methods
    # ----------------------------

    def _add_rendering(self):
        self.app.setAutoStepping(False)
        if not self.root:
            self.root = self.app.getRoot()

        rbs = self.sim.getRigidBodies()
        for rb in rbs:
            name = rb.getName()
            node = agxOSG.createVisual(rb, self.root)
            if name == "ground":
                agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
            elif "yumi" in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
            elif "gripper_left" in name and "base" not in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.Red())
            elif "gripper_right" in name and "base" not in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.Blue())
            elif "pusher" in name and "base" not in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.Yellow())
            elif "dlo" in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.Green())
            elif "obstacle" in name or "cylinder" in name:
                agxOSG.setDiffuseColor(node, agxRender.Color.DimGray())
            elif name == "bounding_box":
                agxOSG.setDiffuseColor(node, agxRender.Color.White())
                agxOSG.setAlpha(node, 0.2)
            else:
                agxOSG.setAlpha(node, 0)
                logger.info("No color set for {}.".format(name))

            if "goal" in name:
                agxOSG.setAlpha(node, 0.2)
        scene_decorator = self.app.getSceneDecorator()
        light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
        light_source_0.setPosition(self.light_pose['light_position'])
        light_source_0.setDirection(self.light_pose['light_direction'])
        scene_decorator.setEnableLogo(False)
        scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0, 1.0, 1.0))

    # Extension methods
    # ----------------------------

    def _get_observation(self):
        observation, achieved_goal = self.observation_config.get_observations(self.sim, self.render_to_image,
                                                                              self.controllable_joints, cable="DLO")

        # Note that this structure is necessary for GoalEnv environments
        observation_goal = dict.fromkeys({'observation', 'achieved_goal', 'desired_goal'})
        observation_goal['observation'] = observation
        observation_goal['achieved_goal'] = achieved_goal
        observation_goal['desired_goal'] = self.goal
        return observation_goal

    def _set_action(self, action):
        info = dict()
        for end_effector in self.controllable_joints:
            if end_effector.controllable:
                logger.debug("action: {}".format(action))
                info[end_effector.name] = end_effector.apply_control(self.sim, action, self.dt)

        return info

    def _is_success(self, achieved_goal, desired_goal):
        done = False
        success = self.reward_config.is_success(achieved_goal, desired_goal)
        if self.reward_config.set_done_on_success:
            done = success
        return success, done

    def _sample_goal(self):
        if self.randomized_goal:
            raise NotImplementedError
        else:
            scene = agxSDK.Assembly()  # Create a new empty Assembly
            scene.setName("goal_assembly")

            if not agxIO.readFile(self.goal_scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
                raise RuntimeError("Unable to open goal file \'" + self.goal_scene_path + "\'")

        self.sim.add(scene)
        goal = self.observation_config.get_observations(self.sim, self.render_to_image, self.controllable_joints, cable="DLO",
                                                        goal_only=True)

        if self.show_goal:
            self._add_rendering()
        else:
            self._reset_sim()

        return goal
