import logging
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger('gym_agx.rl')


class RewardType(Enum):
    SPARSE = "sparse"
    DENSE = "dense"
    

class RewardConfig(ABC):

    def __init__(self, reward_type, reward_range, set_done_on_success=True, **kwargs):
        """Initialize RewardConfig object. Defines reward type, limit and function
        :param RewardConfig.RewardType reward_type: defines the type of reward
        :param tuple reward_range: tuple of floats, determining the bounds of the reward
        :param bool set_done_on_success: boolean which determines if task is episodic, and should be terminated once
        success condition is achieved
        :param kwargs: keyword arguments that may be used in reward_function or success_condition.
        """
        self.reward_type = reward_type
        self.reward_range = reward_range
        self.set_done_on_success = set_done_on_success
        self.kwargs = kwargs
        super().__init__()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """This function should return a reward computed based on the achieved_goal and desired_goal dictionaries. These
         may contain more than a single observation, which means the reward can weight different parts of the goal
         differently. The info dictionary should be populated and returned with any relevant information useful for
         analysing results
        :param dict achieved_goal: dictionary of observations of achieved state
        :param dict desired_goal: dictionary of observations of desired state
        :param dict info: information dictionary, which should be updated, and can be used to include more information
        needed for reward computations
        :return: float reward
        """
        if self.reward_type == RewardType.SPARSE:
            success = self.is_success(achieved_goal, desired_goal)
            if success:
                reward = self.reward_range[1]
            else:
                reward = self.reward_range[0]
        elif self.reward_type == RewardType.DENSE:
            reward, info = self.reward_function(achieved_goal, desired_goal, info)
            reward = self.scale_reward(reward)
        else:
            raise Exception("Reward type must be either SPARSE or DENSE!")

        return reward

    def is_success(self, achieved_goal, desired_goal):
        """This function should return a boolean based on the achieved_goal and desired_goal dictionaries
        :param dict achieved_goal: dictionary of observations from achieved state
        :param dict desired_goal: dictionary of observations from desired state
        :return: success
        """
        success = self.success_condition(achieved_goal, desired_goal)
        assert type(success) is bool, "Success condition must return boolean. Returned type {}".format(type(success))
        return success

    @abstractmethod
    def reward_function(self, achieved_goal, desired_goal, info):
        """This abstract method should define how the reward is computed
        :param dict achieved_goal: dictionary of observations from achieved state
        :param dict desired_goal: dictionary of observations from desired state
        :param dict info: information dictionary, which should be updated, and can be used to include more information
        needed for reward computations
        :return: reward, info
        """
        pass

    @abstractmethod
    def scale_reward(self, reward):
        """This abstract method should define how the dense reward is scaled. This function is always called, for dense
        rewards, after the reward_function returns a reward value
        :param reward: reward output from reward_function
        :return: scaled reward
        """
        pass

    @abstractmethod
    def success_condition(self, achieved_goal, desired_goal):
        """This abstract method returns a boolean indicating if the desired_goal is achieved. Since the goals may be
        composed of several observations, different conditions can be checked, at the same time
        :param dict achieved_goal: dictionary of observations from achieved state
        :param dict desired_goal: dictionary of observations from desired state
        :return boolean
        """
        pass
