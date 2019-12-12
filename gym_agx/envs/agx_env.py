import numpy as np
from collections import OrderedDict

from gym import error, spaces
from gym.utils import seeding
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
    raise error.DependencyNotInstalled("{}. (HINT: you need to install AGX Dynamics, "
                                       "have a valid license and run setup_env.bash.)".format(e))

from gym_agx.utils.agx_utils import get_state


def convert_observation_to_space(observation):
    """Convert an observation to Gym space. An observation object can be either a dictionary or a ndarray.
    :param observation:
    :return: Gym space object with observations
    """
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
    metadata = {'render.modes': ['osg', 'debug']}

    def __init__(self, scene_path, n_substeps, grippers, n_actions):
        self.scene_path = scene_path
        self.n_substeps = n_substeps

        # Initialize AGX simulation
        init = agx.AutoInit()
        self.sim = agxSDK.Simulation()
        self._build_simulation()

        # Needed for rendering OSG ExampleApplication
        self.app = None
        self.root = None

        # set frames per second
        self.fps = int(np.round(1.0 / self.dt))

        self.np_random = None
        self.seed()

        # TODO: Determine if these steps are needed
        # self._env_setup(camera=camera)
        # should deepcopy be used here?
        self.initial_state = get_state(self.sim)

        # TODO: Get first goal and observation
        self.goal = self._sample_goal()
        obs = self._get_obs()

        # TODO: Define observation and action spaces
        self.action_space = spaces.Box(-1., 1., shape=(n_actions, len(grippers)), dtype='float32')
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

        t = self.sim.getTimeStamp()
        print("Timestep: {}".format(t))
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
        # TODO: What should be done here and not in _reset_sim?
        # return obs

    def close(self):
        if self.app is not None:
            # self.viewer.finish()
            self.app = None
            agx.shutdown()

    def render(self, mode='osg'):
        self._render_callback()
        if not self.app.breakRequested():
            self.app.executeOneStepWithGraphics()

    # AGX Dynamics methods
    # ----------------------------

    def _build_simulation(self):
        """Initializes simulation:
        sim - A pointer to an instance of a agxSDK::Simulation
        Adds scene to simulation.
        """
        scene = agxSDK.Assembly()  # Create a new empty Assembly

        if not agxIO.readFile(self.scene_path, self.sim, scene, agxSDK.Simulation.READ_ALL):
            raise RuntimeError("Unable to open file \'" + self.scene_path + "\'")
        self.sim.add(scene)

    def _reset_sim(self):
        """Resets the simulation.
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
