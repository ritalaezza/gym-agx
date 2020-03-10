import gym
import math
from gym_agx import envs
from gym_agx.utils.utils import sinusoidal_trajectory

LENGTH = 0.1
N_EPISODES = 1
EPISODE_LENGTH = 1500   # 1 minute (1 step is 0.04 seconds)


def main():
    n_seconds = 10  # seconds
    period = 12  # seconds
    amplitude = LENGTH / 4
    rad_frequency = 2 * math.pi * (1 / period)
    # compliance = 1e12
    # decay = 0.1

    env = gym.make('BendWireDense-v0')
    for _ in range(N_EPISODES):
        env.reset()
        n_steps = int(n_seconds / env.dt)
        for t in range(n_steps):
            env.render()
            # compliance = compliance * math.exp(-decay * t*env.dt)
            velocity = sinusoidal_trajectory(amplitude, rad_frequency, t*env.dt)
            obs, reward, done, info = env.step(velocity)  # env.action_space.sample()
            print(reward)
    env.close()


if __name__ == "__main__":
    main()
