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


class AgxEnv(gym.GoalEnv):
    """Superclass for all AGX Dynamics environments. Initializes AGX, loads scene
    from file and builds it.
    """
    metadata = {'render.modes': ['osg', 'debug']}

    def __init__(self, scene_path, n_substeps, grippers, n_actions):
        self.scene_path = scene_path
        self.n_substeps = n_substeps

        # Initialize AGX simulation
        self.init = agx.AutoInit()
        self.sim = agxSDK.Simulation()
        self._build_simulation()

        # Needed for rendering OSG ExampleApplication
        self.app = None
        self.root = None

        # set frames per second
        self.fps = int(np.round(1.0 / self.dt))

        self.np_random = None
        self.seed()

        self.goal = self._sample_goal()
        obs = self._get_obs()

        self.action_space = spaces.Box(-1., 1., shape=(n_actions, len(grippers)), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        print("dt")
        return self.sim.getTimeStep() * self.n_substeps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        print("seed")
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        print("step")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._step_callback()

        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        print("reset")
        super(AgxEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        print("close")
        if self.app is not None:
            # self.viewer.finish()
            self.app = None
            agx.shutdown()

    def render(self, mode='human'):
        print("render")
        # while this rendering mode is not used, it has to be defined for compatibility
        self._render_callback()

    # AGX Dynamics methods
    # ----------------------------

    def _init_app(self, start_rendering):
        """Start rendering using OSG's ExampleApplication.
        """
        raise NotImplementedError()

    def _add_rendering(self, mode='osg'):
        """Create ExampleApplication instance and add rendering information.
        """
        raise NotImplementedError()

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
