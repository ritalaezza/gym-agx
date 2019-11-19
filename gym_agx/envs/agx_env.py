from collections import OrderedDict
import os


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


class AgxEnv(gym.Env):
    """
    Superclass for all AGX Dynamics environments. Initializes AGX, loads scene
    from file and builds it.
    """

    def __init__(self, scene_path, frame_skip):
        self.scene_path = scene_path
        self.frame_skip = frame_skip

        if agxPython.getContext() is None:
            init = agx.AutoInit()

        if scene_path.startswith("/"):
            fullpath = self.scene_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", self.scene_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.scene = self._load_scene_from_file(fullpath)

        self.app = agxOSG.ExampleApplication()
        self.argParser = agxIO.ArgumentParser([sys.executable] + sys.argv)
        self.app.addScene(argParser.getArgumentName(1), "_build_scene", ord('1'), True)

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        #self.data = self.sim.data
        #self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def init_camera(self, eye=DEFAULT_EYE,center=DEFAULT_CENTER,up=DEFAULT_UP):
        self.app.setCameraHome(eye, center, up)

    def _load_scene_from_file(self, file_path):
        """
        Load scene from file. Read all the objects in the file and add to
        the scene (not the simulation!) (and the scenegraph root)
        """
        scene = agxSDK.Assembly()  # Create a new empty Assembly

        if not agxOSG.readFile(file_path, self.sim, self.root, scene):
            raise RuntimeError("Unable to open file \'" + file_path + "\'")

        return scene

    def _build_scene():
        # sim - A pointer to an instance of a agxSDK::Simulation
        # app - A pointer to an instance of a agxOSG::ExampleApplication
        # root - A pointer to an instance of agxOSG::Group
        self.sim = agxPython.getContext().environment.getSimulation()
        self.app = agxPython.getContext().environment.getApplication()
        self.root = agxPython.getContext().environment.getSceneRoot()

        self.sim.add(self.scene)

        init_camera()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy()
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_scene(self):
        """
        Resets the scene.
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        #return self.model.opt.timestep * self.frame_skip
        return self.sim.getTimeStep() * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        # Not sure what this does, it seems to be a way of giving a control sequence
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.stepForward()

    def render(self,
               mode='osg',
               camera_id=None,
               camera_name=None):
        if mode == 'osg':
            self.app.setEnableOSGRenderer(True)
        elif mode == 'debug':
            self.app.setEnableDebugRenderer(True)

        # Call the init method of ExampleApplication
        # It will setup the viewer, windows etc.
        if self.app.init(self.argParser):
            self.app.run()
        else:
            print("An error occurred while initializing ExampleApplication.")

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
