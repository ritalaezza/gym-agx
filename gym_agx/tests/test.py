import gym
from gym_agx import envs

env = gym.make('BendWire-v0')
env.reset()
for _ in range(10):
    # env.render()
    env.step(env.action_space.sample())  # take a random action
env.close()
