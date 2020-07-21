import copy
import numpy as np
from gym import ObservationWrapper, spaces


class GoalEnvObservation(ObservationWrapper):
    """Observation wrapper that extracts only observation from dictionary entry of a GoalEnv 'observation',
    into an OpenAI Gym space of type Box.
     """

    def __init__(self, env):
        super(GoalEnvObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, spaces.Dict), (
            "GoalEnvObservation is only usable with dict observations.")

        unwrapped_observation_space = wrapped_observation_space['observation']
        assert isinstance(unwrapped_observation_space, spaces.Box), (
            "GoalEnvObservation is only usable with a single Box observation inside Dict.")
        self.observation_space = unwrapped_observation_space

    def observation(self, observation):
        return observation['observation']


class GoalEnvRGBObservation(ObservationWrapper):
    """Observation wrapper that extracts image observation from dictionary entry of a GoalEnv 'observation',
    into an OpenAI Gym space of type Box. Leaves other observations intact.
     """

    def __init__(self, env):
        super(GoalEnvRGBObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, spaces.Dict), (
            "GoalEnvRGBObservation is only usable with dict observations.")

        unwrapped_observation_space = wrapped_observation_space['observation']
        assert 'img_rgb' in unwrapped_observation_space.spaces.keys(), (
            "GoalEnvRGBObservation is only usable with a 'img_rgb' observation inside Dict.")

        rgb_image_shape = unwrapped_observation_space['img_rgb'].shape
        rgb_image_space = spaces.Box(shape=(3, rgb_image_shape[0], rgb_image_shape[1]), low=-np.inf, high=np.inf)
        self.observation_space = spaces.Dict([('observation', rgb_image_space),
                                              ('achieved_goal', wrapped_observation_space.spaces['achieved_goal']),
                                              ('desired_goal', wrapped_observation_space.spaces['desired_goal'])])

    def observation(self, observation):
        observation['observation'] = np.moveaxis(observation['observation']['img_rgb'], 2, 0).copy()
        return observation


class GoalEnvFlattenObservation(ObservationWrapper):
    """Observation wrapper that flattens the 'observation' dictionary entry of a GoalEnv observation space, from any
    space type into an OpenAI Gym space of type Box.
     """

    def __init__(self, env):
        super(GoalEnvFlattenObservation, self).__init__(env)

        wrapped_observation_space = env.observation_space
        assert isinstance(wrapped_observation_space, spaces.Dict), (
            "GoalEnvFlattenObservation is only usable with dict observations.")

        unwrapped_observation_space = wrapped_observation_space['observation']

        self.observation_space = spaces.Dict([
            ('observation', copy.deepcopy(spaces.flatten_space(unwrapped_observation_space))),
            ('achieved_goal', copy.deepcopy(wrapped_observation_space.spaces['achieved_goal'])),
            ('desired_goal', copy.deepcopy(wrapped_observation_space.spaces['desired_goal']))])

    def observation(self, observation):
        return self._flatten_observation(observation)

    def _flatten_observation(self, observation):
        observation['observation'] = spaces.flatten(self.env.observation_space['observation'],
                                                    observation['observation'])
        return observation
