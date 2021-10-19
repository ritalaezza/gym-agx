import gym
from gym_agx import envs

env = gym.make("RubberBandYuMi-v0", reward_type="dense", observation_type="pos", headless=0)
observation = env.reset()

for _ in range(1000):
    env.render("osg")
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
