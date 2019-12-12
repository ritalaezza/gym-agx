import gym
import numpy as np
from gym_agx import envs

env = gym.make('BendWire-v0')
env.reset()
print(env.action_space.sample())
zero_action = np.zeros((6, 1), dtype=float)
for _ in range(10):
    env.render()
    env.step(zero_action)  # take neutral action
env.close()
