import gym
from gym_agx import envs

env = gym.make("ClipClosing-v0")


for _ in range(10):
    done = False
    obs = env.reset()
    for i in range(250):

        env.render()
        action = env.action_space.sample()  # your agent here (this takes random actions)
        obs, reward, done, info = env.step(action)

        if info["is_success"]:
            print("Success")

        if done:
            break

env.close()
