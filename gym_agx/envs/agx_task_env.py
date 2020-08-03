import sys
import numpy as np
import logging

import gym
from gym import error, spaces
from gym.utils import seeding

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


class AgxTaskEnv(gym.Env):
    """Superclass for all AGX Dynamics environments. Initializes AGX, loads scene from file and builds it."""
    metadata = {'render.modes': ['osg', 'debug']}

    def __init__(self, scene_path, n_substeps, n_actions, observation_type, image_size, camera_pose, no_graphics, args):
        """Initializes a AgxEnv object
        :param scene_path: path to binary file containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step()
        :param n_actions: number of actions (DoF)
        :param camera_pose: dictionary containing EYE, CENTER, UP information for rendering
        :param args: arguments for agxViewer
        """
        self.scene_path = scene_path
        self.n_substeps = n_substeps
        self.render_to_image = []
        self.image_size = image_size

        # Initialize AGX simulation
        self.gravity = None
        self.time_step = None
        self.init = agx.AutoInit()
        self.sim = agxSDK.Simulation()
        self._build_simulation()

        # Initialize OSG ExampleApplication
        self.app = agxOSG.ExampleApplication(self.sim)
        self.args = args
        self.root = None
        self.camera_pose = camera_pose
        if not no_graphics:
            self._add_rendering(mode='osg')

        # TODO: Is this needed?
        self.fps = int(np.round(1.0 / self.dt))

        self.np_random = None
        self.seed()

        self.n_actions = n_actions
        self.observation_type = observation_type
        if not no_graphics:
            self._render_callback()

        obs = self._get_observation()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs.shape), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')

    @property
    def dt(self):
        return self.sim.getTimeStep() * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        logger.info("seed")
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def close(self):
        logger.info("close")
        if self.app is not None:
            self.app = None
            agx.shutdown()

    def render(self, mode='human'):
        logger.info("render")
        # while this rendering mode is not used, it has to be defined for compatibility
        self._render_callback()

    # AGX Dynamics methods
    # ----------------------------

    def _build_simulation(self):
        scene = agxSDK.Assembly()  # Create a new empty Assembly

        if not agxIO.readFile(self.scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
            raise RuntimeError("Unable to open file \'" + self.scene_path + "\'")
        scene.setName("main_assembly")
        self.sim.add(scene)
        self.gravity = self.sim.getUniformGravity()
        self.time_step = self.sim.getTimeStep()
        logger.debug("Timestep after readFile is: {}".format(self.time_step))
        logger.debug("Gravity after readFile is: {}".format(self.gravity))

    def _add_background_rendering(self, depth=False):
        """Add rendering buffer to application. Needed for image observations
        :param bool depth: Boolean to define if type of rendering is RGB or depth.
        """

        if depth:
            rti = agxOSG.RenderToImage(self.image_size[0], self.image_size[1], agxOSG.RenderTarget.DEPTH_BUFFER,
                                       8, agxOSG.RenderTarget.DEPTH_COMPONENT)
            rti.setName("depth_buffer")
        else:
            rti = agxOSG.RenderToImage(self.image_size[0], self.image_size[1], agxOSG.RenderTarget.COLOR_BUFFER,
                                       8, agxOSG.RenderTarget.RGB)
            rti.setName("rgb_buffer")

        camera = self.app.getCamera()
        rti.setReferenceCamera(camera)

        self.root = self.app.getRoot()
        self.app.addRenderTarget(rti, self.root)
        self.render_to_image.append(rti)

    def _init_app(self):
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

        if self.observation_type == "rgb" or self.observation_type == "rgb_and_depth":
            self._add_background_rendering()
        if self.observation_type == "depth" or self.observation_type == "rgb_and_depth":
            self._add_background_rendering(depth=True)

        self.sim.setUniformGravity(self.gravity)
        self.sim.setTimeStep(self.time_step)
        logger.debug("Timestep after initSimulation is: {}".format(self.sim.getTimeStep()))
        logger.debug("Gravity after initSimulation is: {}".format(self.sim.getUniformGravity()))

    def _add_rendering(self, mode='osg'):
        """Create ExampleApplication instance and add rendering information.
        Implement this in each subclass
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

    def _reset_sim(self):
        self.sim.cleanup(agxSDK.Simulation.CLEANUP_ALL, True)
        if not self.sim.restore(self.scene_path, agxSDK.Simulation.READ_ALL):
            logger.error("Unable to restore simulation!")
            return False

        self._add_rendering(mode='osg')
        return True

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
        """Executes one step with graphics rendering.
        :param bool add_background_rgb: flag to determine if type of background rendering is RGB.
        :param bool add_background_depth: flag to determine if type of background rendering is depth.
        """
        # assert not self.agx_only, "Rendering is disabled when agx_only is True. No image observations are possible."
        if self.app.breakRequested():
            self._init_app()
        self.app.executeOneStepWithGraphics()
