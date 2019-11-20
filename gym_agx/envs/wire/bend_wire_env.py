from gym_agx.envs import wire_env

SCENE_PATH = 'bend_wire.aagx'


class BendWireEnv(wire_env.WireEnv):
    def __init__(self, reward_type='sparse'):
        """Initialize BendWire environment.
        """
        wire_env.WireEnv.__init__(
            self, SCENE_PATH, n_substeps=5, n_actions=6, distance_threshold=0.05,
            reward_type=reward_type)
