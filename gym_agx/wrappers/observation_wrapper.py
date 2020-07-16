import copy
from gym import ObservationWrapper, spaces


class GoalEnvFlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens each dictionary entry of a GoalEnv observation ('observation', 'achieved_goal',
     'desired_goal'), into an OpenAI Gym space of type Box.
     """
    def __init__(self, env):
        super(GoalEnvFlattenObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, spaces.Dict), (
            "FilterObservationWrapper is only usable with dict observations.")

        self._goal_keys = ['observation', 'achieved_goal', 'desired_goal']
        self.observation_space = type(wrapped_observation_space)([
            (name, copy.deepcopy(spaces.flatten_space(space)))
            for name, space in wrapped_observation_space.spaces.items()
            if name in self._goal_keys
        ])

    def observation(self, observation):
        return self._flatten_observation(observation)

    def _flatten_observation(self, observation):
        observation = type(observation)([
            (name, spaces.flatten(self.env.observation_space[name], value))
            for name, value in observation.items()
            if name in self._goal_keys
        ])
        return observation
