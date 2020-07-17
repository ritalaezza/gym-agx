import sys
import numpy as np
import logging

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


class AgxTaskEnv(gym.Env):
    """Superclass for all AGX Dynamics environments. Initializes AGX, loads scene from file and builds it."""
    metadata = {'render.modes': ['osg', 'debug']}

    def __init__(self, scene_path, n_substeps, n_actions, observation_config, camera_pose, args):
        """Initializes a AgxEnv object
        :param scene_path: path to binary file containing serialized simulation defined in sim/ folder
        :param n_substeps: number os simulation steps per call to step()
        :param n_actions: number of actions (DoF)
        :param camera_pose: dictionary containing EYE, CENTER, UP information for rendering
        :param args: arguments for agxViewer
        """
        self.scene_path = scene_path
        self.n_substeps = n_substeps

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
        self._add_rendering(mode='osg')

        # TODO: Is this needed?
        self.fps = int(np.round(1.0 / self.dt))

        self.np_random = None
        self.seed()

        self.n_actions = n_actions
        self.observation_config = observation_config

        obs = self._get_observation()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(obs.shape[0],), dtype=np.float32)
        self.action_space = spaces.Box(-1., 1., shape=(self.n_actions,), dtype='float32')

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

    def _init_app(self, start_rendering):
        logger.info("init app")
        if start_rendering:
            self._add_rendering()
        self.app.init(agxIO.ArgumentParser([sys.executable] + self.args))
        self.app.setCameraHome(self.camera_pose['eye'], self.camera_pose['center'],
                               self.camera_pose['up'])  # only after app.init
        self.app.initSimulation(self.sim, start_rendering)
        self.sim.setUniformGravity(self.gravity)
        self.sim.setTimeStep(self.time_step)
        logger.debug("Timestep after initSimulation is: {}".format(self.sim.getTimeStep()))
        logger.debug("Gravity after initSimulation is: {}".format(self.sim.getUniformGravity()))

    def _add_rendering(self, mode='osg'):
        """Create ExampleApplication instance and add rendering information.
        Implement this in each subclass
        """
        raise NotImplementedError()

    def _reset_sim(self):
        """Resets the simulation.
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
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

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
        if not self.app.breakRequested():
            self.app.executeOneStepWithGraphics()
        else:
            self._init_app(True)
            self.app.executeOneStepWithGraphics()