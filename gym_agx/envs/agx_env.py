import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from agxPythonModules.utils.numpy_utils import create_numpy_array

import gym
from gym import error, spaces
from gym.utils import seeding

from gym_agx.utils.utils import construct_space

try:
    import agx
    import agxPython
    import agxCollide
    import agxOSG
    import agxIO
    import agxSDK
    import agxUtil
    import agxRender
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install AGX Dynamics, "
                                       "have a valid license and run 'setup_env.bash'.)".format(e))

logger = logging.getLogger('gym_agx.envs')


class AgxEnv(gym.GoalEnv):
    """Superclass for all AGX Dynamics environments. Initializes AGX, loads scene from file and builds it."""
    metadata = {'render.modes': ['osg', 'debug', 'human', 'depth']}

    def __init__(self, scene_path, n_substeps, n_actions, observation_config, camera_pose, osg_window, agx_only, args):
        """Initializes a AgxEnv object
        :param str scene_path: path to binary file containing serialized simulation defined in sim/ folder
        :param int n_substeps: number os simulation steps per call to step()
        :param int n_actions: number of actions (DoF)
        :param gym_agx.rl.observation.ObservationConfig observation_config: observation configuration object.
        :param dict camera_pose: dictionary containing EYE, CENTER, UP information for rendering
        :param bool osg_window: enables/disables window rendering (useful for training).
        :param bool agx_only: enables/disables all rendering (useful for training which does not use images).
        :param list args: arguments for agxViewer
        """
        self.render_mode = 'osg'
        self.scene_path = scene_path
        self.n_substeps = n_substeps

        self.gravity = None
        self.time_step = None
        self.init = agx.AutoInit()
        self.sim = agxSDK.Simulation()
        self._build_simulation()

        self.app = None
        self.root = None
        self.camera_pose = camera_pose
        self.osg_window = osg_window
        self.agx_only = agx_only

        self.n_actions = n_actions
        self.observation_config = observation_config

        self.args = args
        if not self.agx_only:
            self.app = agxOSG.ExampleApplication(self.sim)
            self.args = self.args + ['--window', 2 * observation_config.image_size[1],
                                     2 * observation_config.image_size[0]]
            if self.osg_window:
                print("WARNING: OSG window is enabled!")
                if self.observation_config.depth_in_obs or self.observation_config.rgb_in_obs:
                    print("=======> Observations contain image data (OSG rendering cannot be disabled).")
                    print("=======> Rendering is done inside step(). No need to call render().")
                else:
                    print("=======> Observations do not contain image data.")
                    print("=======> Rendering is done inside render(), do not use 'human' or 'depth' mode.")
            else:
                print("WARNING: OSG window is disabled!")
                self.args = self.args + ['--osgWindow', '0']
                if self.observation_config.depth_in_obs or self.observation_config.rgb_in_obs:
                    print("=======> Observations contain image data (OSG rendering cannot be enabled).")
                    print("=======> Rendering is done inside step(), only 'human' or 'depth' modes available.")
                else:
                    print("=======> Observations do not contain image data.")
                    print("=======> Rendering is done inside render(), only 'human' or 'depth' modes available.")

        self.render_to_image = []
        self.img_object = None

        # TODO: Is this needed?
        self.fps = int(np.round(1.0 / self.dt))
        self.np_random = None
        self.seed()

        self.goal = self._sample_goal()
        obs = self._get_observation()

        self.obs_space = construct_space(obs['observation'])
        self.goal_space = construct_space(obs['desired_goal'])

        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=self.goal_space,
            achieved_goal=self.goal_space,
            observation=self.obs_space
        ))

    @property
    def dt(self):
        return self.sim.getTimeStep() * self.n_substeps

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        raise NotImplementedError()

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        logger.info("seed")
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        logger.info("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        info = self._set_action(action)

        self._step_callback()
        if self.observation_config.rgb_in_obs or self.observation_config.rgb_in_obs:
            self._render_callback(self.observation_config.rgb_in_obs, self.observation_config.rgb_in_obs)

        obs = self._get_observation()
        info['is_success'], done = self._is_success(obs['achieved_goal'], self.goal)
        reward, info = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        logger.info("reset")
        super(AgxEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        self.goal = self._sample_goal().copy()
        obs = self._get_observation()
        return obs

    def close(self):
        logger.info("close")
        if self.app is not None:
            self.app = None
            agx.shutdown()

    def render(self, mode='human'):
        logger.info("render")
        assert not self.agx_only, "Rendering is disabled when agx_only is True."
        if mode is not self.render_mode:
            assert mode in self.metadata['render.modes'], "Invalid render mode! ['osg', 'debug', 'human', 'depth']"
            if mode in ['osg', 'debug']:
                self._set_render_mode(mode)
            self.render_mode = mode

        if self.osg_window and mode not in ['human', 'depth']:
            self._render_callback()
        elif self.osg_window and mode in ['human', 'depth']:
            print("Matplotlib 'human' and 'depth' rendering is disabled, use 'osg' or 'debug'.")
        elif not self.osg_window and mode in ['osg', 'debug']:
            print("OSG window is disabled.")
        elif not self.osg_window and mode == 'human':
            if not self.observation_config.rgb_in_obs or not self.render_to_image:
                self._render_callback(add_background_rgb=True)
            self._matplotlib_rendering(buffer='rgb_buffer')
        elif not self.osg_window and mode == 'depth':
            if not self.observation_config.depth_in_obs or not self.render_to_image:
                self._render_callback(add_background_depth=True)
            self._matplotlib_rendering(buffer='depth_buffer')

    # AGX Dynamics methods
    # ----------------------------

    def _build_simulation(self):
        """Build simulation
        """
        logger.info("build simulation")
        scene = agxSDK.Assembly()  # Create a new empty Assembly

        if not agxIO.readFile(self.scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
            raise RuntimeError("Unable to open file \'" + self.scene_path + "\'")
        scene.setName("main_assembly")
        self.sim.add(scene)

        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def _init_app(self, add_background_rgb=False, add_background_depth=False):
        """Initialize OSG Example Application. Needed for rendering graphics.
        :param bool add_background_rgb: flag to determine if type of background rendering is RGB.
        :param bool add_background_depth: flag to determine if type of background rendering is depth.
        """
        logger.info("init app")
        self.app.init(agxIO.ArgumentParser([sys.executable] + self.args))
        self.app.setCameraHome(self.camera_pose['eye'],
                               self.camera_pose['center'],
                               self.camera_pose['up'])  # only after app.init
        self.app.initSimulation(self.sim, True)  # initialize graphics

        if self.observation_config.rgb_in_obs or add_background_rgb:
            self._add_background_rendering()
        if self.observation_config.depth_in_obs or add_background_depth:
            self._add_background_rendering(depth=True)

        self.sim.setUniformGravity(self.gravity)
        self.sim.setTimeStep(self.time_step)
        logger.debug("Timestep after initSimulation is: {}".format(self.sim.getTimeStep()))
        logger.debug("Gravity after initSimulation is: {}".format(self.sim.getUniformGravity()))

    def _reset_sim(self):
        """Resets the simulation.
        """
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_ALL):
            logger.error("Unable to restore simulation!")
            return False

        if not self.agx_only:
            self._add_rendering()
        return True

    def _step_callback(self):
        """Steps the simulation, until n_substeps have passed.
        """
        t = self.sim.getTimeStamp()

        t_0 = t
        while t < t_0 + self.dt:
            try:
                self.sim.stepForward()
                t = self.sim.getTimeStamp()
            except:
                logger.error("Unexpected error:", sys.exc_info()[0])

    def _render_callback(self, add_background_rgb=False, add_background_depth=False):
        """Executes one step with graphics rendering.
        :param bool add_background_rgb: flag to determine if type of background rendering is RGB.
        :param bool add_background_depth: flag to determine if type of background rendering is depth.
        """
        assert not self.agx_only, "Rendering is disabled when agx_only is True. No image observations are possible."
        if self.app.breakRequested():
            self._init_app(add_background_rgb, add_background_depth)
        self.app.executeOneStepWithGraphics()

    def _set_render_mode(self, mode):
        """Change OSG render mode.
        :param str mode: rendering mode ('osg', 'debug')
        """
        if mode == 'osg':
            self.app.setEnableDebugRenderer(False)
            self.app.setEnableOSGRenderer(True)
        elif mode == 'debug':
            self.app.setEnableDebugRenderer(True)
            self.app.setEnableOSGRenderer(False)
        else:
            logger.error("Unexpected rendering mode: {}".format(mode))

    def _add_background_rendering(self, depth=False):
        """Add rendering buffer to application. Needed for image observations
        :param bool depth: Boolean to define if type of rendering is RGB or depth.
        """
        image_size = self.observation_config.image_size
        if depth:
            rti = agxOSG.RenderToImage(image_size[0], image_size[1], agxOSG.RenderTarget.DEPTH_BUFFER,
                                       8, agxOSG.RenderTarget.DEPTH_COMPONENT)
            rti.setName("depth_buffer")
        else:
            rti = agxOSG.RenderToImage(image_size[0], image_size[1], agxOSG.RenderTarget.COLOR_BUFFER,
                                       8, agxOSG.RenderTarget.RGB)
            rti.setName("rgb_buffer")

        camera = self.app.getCamera()
        rti.setReferenceCamera(camera)

        self.root = self.app.getRoot()
        self.app.addRenderTarget(rti, self.root)
        self.render_to_image.append(rti)

    def _matplotlib_rendering(self, buffer='rgb_buffer'):
        """Matplotlib rendering shows RGB and depth images. Useful when OSG window is disabled.
        :param str buffer: type of buffer for matplotlib rendering: 'rgb_buffer' or 'depth_buffer'.
        """
        rgb_buffer = None
        depth_buffer = None
        for rti in self.render_to_image:
            name = rti.getName()
            if name == 'rgb_buffer' and name == buffer:
                rgb_buffer = rti
            elif name == 'depth_buffer' and name == buffer:
                depth_buffer = rti

        if rgb_buffer:
            image_ptr = rgb_buffer.getImageData()
            image_size = self.observation_config.image_size
            image_data = create_numpy_array(image_ptr, (image_size[0], image_size[1], 3), np.uint8)
        if depth_buffer:
            image_ptr = depth_buffer.getImageData()
            image_size = self.observation_config.image_size
            image_data = create_numpy_array(image_ptr, (image_size[0], image_size[1]), np.float32)
        if any(buffer for buffer in [rgb_buffer, depth_buffer]):
            img = np.flipud(image_data)
            if not self.img_object:
                fig, ax = plt.subplots(1, figsize=(10, 10))
                self.img_object = ax.imshow(img)
                plt.ion()
                plt.show()
            else:
                self.img_object.set_data(img)
                plt.draw()
                plt.pause(1e-5)

    def _add_rendering(self):
        """Add environment specific rendering information, to OSG ExampleApplication.
        Implement this in each subclass.
        """
        raise NotImplementedError()

    # Extension methods
    # ----------------------------

    def _get_observation(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully reached the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()
