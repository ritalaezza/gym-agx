# Gym AGX

This project contains a set of environments, using AGX Dynamics as the simulation software.

## Getting Started

To install the environment, just go to the folder containing the gym-agx folder and run the command:

```
pip install -e gym-agx
```

This will install the gym environment. Now, you can use your gym environment with the following:

```
import gym
from gym_agx import envs

env = gym.make('BendWire-v0')
```
