import gym
from gym_agx import envs
from gym_agx.wrappers import GoalEnvFlattenObservation

env = gym.make("PushRope-v0")
env = GoalEnvFlattenObservation(env)
observation = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    print(observation)

    if done:
        observation = env.reset()
env.close()
