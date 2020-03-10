from rlpyt.envs.base import EnvSpaces, EnvStep
from rlpyt.spaces.gym_wrapper import GymSpaceWrapper
from rlpyt.envs.gym import build_info_tuples, info_to_nt

from gym import spaces, Wrapper
from gym.wrappers.time_limit import TimeLimit
from collections import OrderedDict
from enum import Enum
import numpy as np
import copy

# Important: gym mixes up ordered and unordered keys
# and the Dict space may return a different order of keys that the actual one
KEY_ORDER = ['observation', 'achieved_goal', 'desired_goal']


class GoalEnvWrapper(Wrapper):
    """A wrapper that allow to use dict observation space, coming from GoalEnv.
    It assumes that all the spaces of the dict space are of the same type.
    :param env: (gym.GoalEnv)
    """
    def __init__(self, env, act_null_value=0, obs_null_value=0, force_float32=True):
        super(GoalEnvWrapper, self).__init__(env)

        o = self.env.reset()
        o, r, d, info = self.env.step(self.env.action_space.sample())
        env_ = self.env

        time_limit = isinstance(self.env, TimeLimit)
        while not time_limit and hasattr(env_, "env"):
            env_ = env_.env
            time_limit = isinstance(self.env, TimeLimit)
        if time_limit:
            info["timeout"] = False  # gym's TimeLimit.truncated invalid name.
        self._time_limit = time_limit

        self.metadata = self.env.metadata
        self.action_space = GymSpaceWrapper(
            space=self.env.action_space,
            name="act",
            null_value=act_null_value,
            force_float32=force_float32,
        )

        self.goal_spaces = list(env.observation_space.spaces.values())
        # Check that all observation spaces are of the same type (limitation):
        space_types = [type(env.observation_space.spaces[key]) for key in KEY_ORDER]
        assert len(set(space_types)) == 1, "The spaces for goal and observation must be of the same type"

        # Set obs_dim and goal_dim:
        if isinstance(self.goal_spaces[0], spaces.Discrete):
            self.obs_dim = 1
            self.goal_dim = 1
        else:
            goal_space_shape = env.observation_space.spaces['achieved_goal'].shape
            self.obs_dim = env.observation_space.spaces['observation'].shape[0]
            self.goal_dim = goal_space_shape[0]

        # Set observation space:
        if isinstance(self.goal_spaces[0], spaces.MultiBinary):
            total_dim = self.obs_dim + 2 * self.goal_dim
            observation_space_ = spaces.MultiBinary(total_dim)
        elif isinstance(self.goal_spaces[0], spaces.Box):
            observation_space_ = env.observation_space.spaces['observation']
        elif isinstance(self.goal_spaces[0], spaces.Discrete):
            dimensions = [env.observation_space.spaces[key].n for key in KEY_ORDER]
            observation_space_ = spaces.MultiDiscrete(dimensions)
        else:
            raise NotImplementedError("{} space is not supported".format(type(self.spaces[0])))

        self.observation_space = GymSpaceWrapper(
            space=observation_space_,
            name="obs",
            null_value=obs_null_value,
            force_float32=force_float32,
        )

        build_info_tuples(info)

    def step(self, action):
        a = self.action_space.revert(action)
        o, r, d, info = self.env.step(a)
        goal_obs = self.convert_dict_to_obs(o)
        obs = self.observation_space.convert(goal_obs)
        if self._time_limit:
            if "TimeLimit.truncated" in info:
                info["timeout"] = info.pop("TimeLimit.truncated")
            else:
                info["timeout"] = False
        info = info_to_nt(info)
        return EnvStep(obs, r, d, info)

    def reset(self):
        o = self.env.reset()
        goal_obs = self.convert_dict_to_obs(o)
        obs = self.observation_space.convert(goal_obs)
        return obs

    def seed(self, seed=None):
        return self.env.seed(seed)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    def convert_dict_to_obs(self, obs_dict):
        """
        :param obs_dict: (dict<np.ndarray>)
        :return: (np.ndarray)
        """
        # Note: achieved goal is not removed from the observation
        # this is helpful to have a revertible transformation
        if isinstance(self.observation_space, spaces.MultiDiscrete):
            # Special case for multidiscrete
            return np.concatenate([[int(obs_dict[key])] for key in KEY_ORDER])
        # return np.concatenate([obs_dict[key] for key in KEY_ORDER])
        return obs_dict['observation']

    def convert_obs_to_dict(self, observations):
        """
        Inverse operation of convert_dict_to_obs

        :param observations: (np.ndarray)
        :return: (OrderedDict<np.ndarray>)
        """
        return OrderedDict([
            ('observation', observations[:self.obs_dim]),
            ('achieved_goal', observations[self.obs_dim:self.obs_dim + self.goal_dim]),
            ('desired_goal', observations[self.obs_dim + self.goal_dim:]),
        ])

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )


class GoalSelectionStrategy(Enum):
    """
    The strategies for selecting new goals when
    creating artificial transitions.
    """
    # Select a goal that was achieved
    # after the current step, in the same episode
    FUTURE = 0
    # Select the goal that was achieved
    # at the end of the episode
    FINAL = 1
    # Select a goal that was achieved in the episode
    EPISODE = 2
    # Select a goal that was achieved
    # at some point in the training procedure
    # (and that is present in the replay buffer)
    RANDOM = 3


# For convenience
# that way, we can use string to select a strategy
KEY_TO_GOAL_STRATEGY = {
    'future': GoalSelectionStrategy.FUTURE,
    'final': GoalSelectionStrategy.FINAL,
    'episode': GoalSelectionStrategy.EPISODE,
    'random': GoalSelectionStrategy.RANDOM
}


class HindsightExperienceReplayWrapper(object):
    """
    Wrapper around a replay buffer in order to use HER.
    This implementation is inspired by to the one found in https://github.com/NervanaSystems/coach/.

    :param replay_buffer: (ReplayBuffer)
    :param n_sampled_goal: (int) The number of artificial transitions to generate for each actual transition
    :param goal_selection_strategy: (GoalSelectionStrategy) The method that will be used to generate
        the goals for the artificial transitions.
    :param wrapped_env: (HERGoalEnvWrapper) the GoalEnv wrapped using HERGoalEnvWrapper,
        that enables to convert observation to dict, and vice versa
    """

    def __init__(self, replay_buffer, n_sampled_goal, goal_selection_strategy, wrapped_env):
        super(HindsightExperienceReplayWrapper, self).__init__()

        assert isinstance(goal_selection_strategy, GoalSelectionStrategy), "Invalid goal selection strategy," \
                                                                           "please use one of {}".format(
            list(GoalSelectionStrategy))

        self.n_sampled_goal = n_sampled_goal
        self.goal_selection_strategy = goal_selection_strategy
        self.env = wrapped_env
        # Buffer for storing transitions of the current episode
        self.episode_transitions = []
        self.replay_buffer = replay_buffer

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer

        :param obs_t: (np.ndarray) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (np.ndarray) the new observation
        :param done: (bool) is the episode done
        """
        assert self.replay_buffer is not None
        # Update current episode buffer
        self.episode_transitions.append((obs_t, action, reward, obs_tp1, done))
        if done:
            # Add transitions (and imagined ones) to buffer only when an episode is over
            self._store_episode()
            # Reset episode buffer
            self.episode_transitions = []

    def sample(self, *args, **kwargs):
        return self.replay_buffer.sample(*args, **kwargs)

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return self.replay_buffer.can_sample(n_samples)

    def __len__(self):
        return len(self.replay_buffer)

    def _sample_achieved_goal(self, episode_transitions, transition_idx):
        """
        Sample an achieved goal according to the sampling strategy.

        :param episode_transitions: ([tuple]) a list of all the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        if self.goal_selection_strategy == GoalSelectionStrategy.FUTURE:
            # Sample a goal that was observed in the same episode after the current step
            selected_idx = np.random.choice(np.arange(transition_idx + 1, len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.FINAL:
            # Choose the goal achieved at the end of the episode
            selected_transition = episode_transitions[-1]
        elif self.goal_selection_strategy == GoalSelectionStrategy.EPISODE:
            # Random goal achieved during the episode
            selected_idx = np.random.choice(np.arange(len(episode_transitions)))
            selected_transition = episode_transitions[selected_idx]
        elif self.goal_selection_strategy == GoalSelectionStrategy.RANDOM:
            # Random goal achieved, from the entire replay buffer
            selected_idx = np.random.choice(np.arange(len(self.replay_buffer)))
            selected_transition = self.replay_buffer.storage[selected_idx]
        else:
            raise ValueError("Invalid goal selection strategy,"
                             "please use one of {}".format(list(GoalSelectionStrategy)))
        return self.env.convert_obs_to_dict(selected_transition[0])['achieved_goal']

    def _sample_achieved_goals(self, episode_transitions, transition_idx):
        """
        Sample a batch of achieved goals according to the sampling strategy.

        :param episode_transitions: ([tuple]) list of the transitions in the current episode
        :param transition_idx: (int) the transition to start sampling from
        :return: (np.ndarray) an achieved goal
        """
        return [
            self._sample_achieved_goal(episode_transitions, transition_idx)
            for _ in range(self.n_sampled_goal)
        ]

    def _store_episode(self):
        """
        Sample artificial goals and store transition of the current
        episode in the replay buffer.
        This method is called only after each end of episode.
        """
        # For each transition in the last episode,
        # create a set of artificial transitions
        for transition_idx, transition in enumerate(self.episode_transitions):

            obs_t, action, reward, obs_tp1, done = transition

            # Add to the replay buffer
            self.replay_buffer.add(obs_t, action, reward, obs_tp1, done)

            # We cannot sample a goal from the future in the last step of an episode
            if (transition_idx == len(self.episode_transitions) - 1 and
                    self.goal_selection_strategy == GoalSelectionStrategy.FUTURE):
                break

            # Sampled n goals per transition, where n is `n_sampled_goal`
            # this is called k in the paper
            sampled_goals = self._sample_achieved_goals(self.episode_transitions, transition_idx)
            # For each sampled goals, store a new transition
            for goal in sampled_goals:
                # Copy transition to avoid modifying the original one
                obs, action, reward, next_obs, done = copy.deepcopy(transition)

                # Convert concatenated obs to dict, so we can update the goals
                obs_dict, next_obs_dict = map(self.env.convert_obs_to_dict, (obs, next_obs))

                # Update the desired goal in the transition
                obs_dict['desired_goal'] = goal
                next_obs_dict['desired_goal'] = goal

                # Update the reward according to the new desired goal
                reward = self.env.compute_reward(goal, next_obs_dict['achieved_goal'], None)
                # Can we use achieved_goal == desired_goal?
                done = False

                # Transform back to ndarrays
                obs, next_obs = map(self.env.convert_dict_to_obs, (obs_dict, next_obs_dict))

                # Add artificial transition to the replay buffer
                self.replay_buffer.add(obs, action, reward, next_obs, done)


class OrnsteinUhlenbeckProcess:
    def __init__(self, theta, mu, sigma, time_scale=1e-1, size=1, initial_value=None):
        """Taken from: https://github.com/dchetelat/acer"""
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.time_scale = time_scale
        self.size = size
        self.initial_value = initial_value if initial_value is not None else np.zeros(size)
        self.previous_value = self.initial_value

    def sample(self):
        value = self.previous_value
        value += self.theta * (self.mu - self.previous_value) * self.time_scale
        value += self.sigma * np.sqrt(self.time_scale) * np.random.normal(size=self.size)
        return value

    def reset(self):
        self.previous_value = self.initial_value

    def sampling_parameters(self):
        mean = self.previous_value + self.theta * (self.mu - self.previous_value) * self.time_scale
        sd = self.sigma * np.sqrt(self.time_scale) * np.ones((self.size,))
        return mean, sd