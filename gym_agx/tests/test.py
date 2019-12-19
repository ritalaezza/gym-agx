import gym
from gym_agx import envs

N_EPISODES = 1
EPISODE_LENGTH = 1200  # 1 minute (1 step is 0.05 seconds)


def main():
    env = gym.make('BendWire-v0')
    for _ in range(N_EPISODES):
        env.reset()
        for _ in range(EPISODE_LENGTH):
            env.render()
            obs, reward, done, info = env.step(env.action_space.sample())
            if done:
                if info['is_success']:
                    print("Goal reached!", "reward=", reward)
                else:
                    print("Stopped early.")
                break
    env.close()


if __name__ == "__main__":
    main()
