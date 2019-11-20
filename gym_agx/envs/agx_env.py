from collections import OrderedDict
import os
import sys

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

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
    raise error.DependencyNotInstalled("{}. (HINT: you need to install AGX Dynamics, have a valid license and run setup_env.bash.)".format(e))

DEFAULT_EYE = agx.Vec3(10.4129, -16.5642, 12.7635)
DEFAULT_CENTER = agx.Vec3(10.0255, -15.7991, 12.2491)
DEFAULT_UP = agx.Vec3(-0.232383, 0.458895, 0.857563)


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space


class AgxEnv(gym.GoalEnv):
    """Superclass for all AGX Dynamics environments. Initializes AGX, loads scene
    from file and builds it.
    """

    def __init__(self, scene_path, n_substeps, n_actions):
        if (not scene_path.startswith("/")):
             scene_fullpath = os.path.join(os.path.dirname(__file__), "assets", scene_path)
        if not path.exists(scene_fullpath):
            raise IOError("File %s does not exist" % self.scene_path)
        self.scene_path = scene_fullpath
        self.n_substeps = n_substeps

        # TODO: Understand these steps:
        if agxPython.getContext() is None:
            init = agx.AutoInit()
        self.app = agxOSG.ExampleApplication()
        self.argParser = agxIO.ArgumentParser([sys.executable] + sys.argv)
        self.app.addScene(self.argParser.getArgumentName(1), "_build_scene", ord('1'), True)
        if self.app.init(self.argParser):
            self.app.run()
        else:
            print("An error occurred while initializing ExampleApplication.")

        self.metadata = {
            'render.modes': ['osg', 'debug'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()

        # TODO: Determine if these steps are needed
        # self._env_setup(initial_qpos=initial_qpos)
        # self.initial_state = copy.deepcopy(self.sim.get_state())

        # TODO: Get first goal and observation
        self.goal = self._sample_goal()
        obs = self._get_obs()

        # TODO: Define observation and state spaces
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.getTimeStep() * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.stepForward()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        print("Resetting...")
        # TODO: How to reset a simulation in AGX Dynamics?
        # return obs

    def close(self):
        if self.app is not None:
            # self.viewer.finish()
            self.app = None
            agx.shutdown();

    def render(self, mode='osg'):
        self._init_camera(self.app)
        if mode == 'osg':
            self.app.setEnableOSGRenderer(True)
        elif mode == 'debug':
            self.app.setEnableDebugRenderer(True)
        # TODO: What else can be moved into this function?

    # AGX Dynamics methods
    # ----------------------------

    def _init_camera(self, eye=DEFAULT_EYE, center=DEFAULT_CENTER, up=DEFAULT_UP):
        """Initializes camera. Maybe should be called in subclasses?
        """
        self.app.setCameraHome(eye, center, up)

    def _load_scene_from_file(self, file_path):
        """Loads scene from file. Read all the objects in the file and add to
        the scene (not the simulation!) (and the scenegraph root)
        """
        scene = agxSDK.Assembly()  # Create a new empty Assembly

        if not agxOSG.readFile(file_path, self.sim, self.root, scene):
            raise RuntimeError("Unable to open file \'" + file_path + "\'")

        return scene

    def _build_scene(self):
        """Initializes simulation, application and scene root objects:
        sim - A pointer to an instance of a agxSDK::Simulation
        app - A pointer to an instance of a agxOSG::ExampleApplication
        root - A pointer to an instance of agxOSG::Group
        Adds scene to simulation.
        """
        self.sim = agxPython.getContext().environment.getSimulation()
        self.app = agxPython.getContext().environment.getApplication()
        self.root = agxPython.getContext().environment.getSceneRoot()

        scene = self._load_scene_from_file(self.scene_path)
        self.sim.add(scene)

    def _reset_scene(self):
        """Resets the scene.
        Implement this in each subclass.
        """
        raise NotImplementedError()

    # Extension methods
    # ----------------------------

    def _get_obs(self):
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

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
