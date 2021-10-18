import gym
from gym_agx import envs
from gym_agx.wrappers import GoalEnvFlattenObservation

env = gym.make("BendWireObstacle-v0", osg_window=True, show_goal=True)
env = GoalEnvFlattenObservation(env)
observation = env.reset()

for _ in range(300):
    env.render("osg")
    action = env.action_space.sample()  # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
