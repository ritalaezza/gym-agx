import sys
import logging

import agxIO
import agxSDK
import agxOSG
import agxRender

from gym_agx.envs import agx_env

logger = logging.getLogger('gym_agx.envs')


class DloEnv(agx_env.AgxEnv):
    """Superclass for all DLO environments."""

    def __init__(self, args, scene_path, n_substeps, end_effectors, observation_config, camera_config, reward_config,
                 randomized_goal, goal_scene_path, show_goal):
        """Initializes a DloEnv object
        :param args: arguments for agxViewer.
        :param scene_path: path to binary file in assets/ folder containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step().
        :param end_effectors: list of EndEffector objects, defining controllable constraints.
        :param observation_config: ObservationConfig object, defining the types of observations.
        :param camera_config: dictionary containing EYE, CENTER, UP information for rendering, with lighting info.
        :param reward_config: reward configuration object, defines success condition and reward function.
        :param randomized_goal: boolean deciding if a new goal is sampled for each episode.
        :param goal_scene_path: path to goal scene file.
        :param show_goal: boolean determining whether goal is rendered or not.
        """
        self.goal_scene_path = goal_scene_path
        self.randomized_goal = randomized_goal
        self.reward_config = reward_config
        self.light_pose = camera_config.light_pose
        self.end_effectors = end_effectors
        self.show_goal = show_goal

        n_actions = 0
        for end_effector in end_effectors:
            if end_effector.controllable:
                for key, constraint in end_effector.constraints.items():
                    if constraint.velocity_control:
                        n_actions += 1
                    if constraint.compliance_control:
                        n_actions += 1

        super(DloEnv, self).__init__(scene_path=scene_path, n_substeps=n_substeps, n_actions=n_actions,
                                     observation_config=observation_config, camera_pose=camera_config.camera_pose,
                                     args=args)

    def render(self, mode="human"):
        return super(DloEnv, self).render(mode)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.reward_config.compute_reward(achieved_goal, desired_goal, info)

    # AgxEnv methods
    # ----------------------------

    def _add_rendering(self, mode='osg'):
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
            name = rb.getName()
            node = agxOSG.createVisual(rb, self.root)
            if name == "ground":
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

    def _reset_sim(self):
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_ALL):
            logger.error("Unable to restore simulation!")
            return False

        self._add_rendering(mode='osg')
        return True

    def _get_observation(self):
        observation, achieved_goal = self.observation_config.get_observations(self.sim, self.end_effectors, cable="DLO")

        # Note that this structure is necessary for GoalEnv environments
        observation_goal = dict.fromkeys({'observation', 'achieved_goal', 'desired_goal'})
        observation_goal['observation'] = observation
        observation_goal['achieved_goal'] = achieved_goal
        observation_goal['desired_goal'] = self.goal
        return observation_goal

    def _set_action(self, action):
        info = dict()
        for end_effector in self.end_effectors:
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
        goal = self.observation_config.get_observations(self.sim, self.end_effectors, cable="DLO", goal_only=True)

        if not self.show_goal:
            self._reset_sim()

        return goal

    def _step_callback(self):
        t = self.sim.getTimeStamp()

        t_0 = t
        while t < t_0 + self.dt:
            try:
                self.sim.stepForward()
                t = self.sim.getTimeStamp()
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])

    def _render_callback(self):
        if not self.app.breakRequested():
            self.app.executeOneStepWithGraphics()
        else:
            self._init_app(True)
            self.app.executeOneStepWithGraphics()
