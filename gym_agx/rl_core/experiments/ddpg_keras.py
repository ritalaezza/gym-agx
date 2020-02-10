import os
import numpy as np
from datetime import datetime

import gym
from gym_agx import envs
from gym_agx.utils.gym_utils import HERGoalEnvWrapper

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Convolution2D
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


class AgxProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


def main(env_name, train=True, path=None):
    # Get the environment and extract the number of actions.
    env = gym.make(env_name)
    env = HERGoalEnvWrapper(env)
    # np.random.seed(123)
    # env.seed(123)
    assert len(env.action_space.shape) == 1
    nb_actions = env.action_space.shape[0]

    # Next, we build a very simple model.
    actor = Sequential()
    actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    actor.add(Dense(400))
    actor.add(Activation('relu'))
    actor.add(Dense(300))
    actor.add(Activation('relu'))
    actor.add(Dense(nb_actions))
    actor.add(Activation('tanh'))
    print(actor.summary())

    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Dense(400)(flattened_observation)
    x = Activation('relu')(x)
    x = Concatenate()([x, action_input])
    x = Dense(300)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)
    print(critic.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and even the metrics!
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
    agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                      memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                      random_process=random_process, gamma=.99, target_model_update=1e-3,
                      processor=AgxProcessor())
    agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    # We can now train the agent.
    if train:
        now = datetime.now()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, 'runs/keras/{}/{}_weights.h5f'.format(env_name, now))
        agent.fit(env, nb_steps=10000, visualize=False, verbose=1)

        # After training is done, we save the final weights.
        agent.save_weights(file_path, overwrite=True)
    else:
        agent.load_weights(path)

    # Finally, evaluate our algorithm for 5 episodes.
    history = agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=2000, verbose=1)

    env.close()


if __name__ == "__main__":
    env_name = "BendWireDense-v0"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(dir_path, 'runs/keras/{}/2020-01-22 16:03:09.754750_weights.h5f'.format(env_name))
    main(env_name, False, file_path)
