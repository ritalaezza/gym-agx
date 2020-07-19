import gym
from gym_agx import envs
from gym_agx.wrappers import GoalEnvFlattenObservation

env = gym.make("BendWire-v0", osg_window=False)
env = GoalEnvFlattenObservation(env)
observation = env.reset()

for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
