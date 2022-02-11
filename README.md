# Gym AGX

OpenAI Gym interface for the [AGX Dynamics](https://www.algoryx.se/agx-dynamics/?utm_term=agx%20dynamics&utm_campaign=AGX&utm_source=adwords&utm_medium=ppc&hsa_acc=3676762440&hsa_cam=10062947755&hsa_grp=102831328442&hsa_ad=435384703433&hsa_src=g&hsa_tgt=kwd-906704179615&hsa_kw=agx%20dynamics&hsa_mt=b&hsa_net=adwords&hsa_ver=3&gclid=Cj0KCQjwlvT8BRDeARIsAACRFiXDJcSOP7NdKyqL4_VEyteCktN5P2D58gd0qkDZFetd2rrbhlN1gcIaAuO4EALw_wcB) simulation software.

## ReForm

Consists of a robot learning sandbox for deformable linear objects (DLOs), e.g. ropes, cables. ReForm contains six DLO manipulation environments. We categorize two types of manipulation in the context of de deformable objects:

1) Explicit shape control - the goal is to deform an object to a specific shape
2) Implicit shape control - the goal is to move the object into a specific configuration, indirectly needing to be deformed

### Explicit shape control

BendWire                   |  BendWireObstacle         |  PushRope
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1oa98fspwVYnNulq5SEgSYkrrTLXTz-_Y" width="200"> | <img src="https://drive.google.com/uc?export=view&id=16qEuWRvFr9B5u46lRJM77_tCy22VEoNz" width="200"> | <img src="https://drive.google.com/uc?export=view&id=1IuuDTLa-7hNP373yuFJfEfU3hZt9dG5Q" width="200">

### Implicit shape control


PegInHole                   |  CableClosing         |  RubberBand
:-------------------------:|:-------------------------:|:-------------------------:
<img src="https://drive.google.com/uc?export=view&id=1oENntc7lSrSpt09vUci-xWGkyPGgRqIp" width="200"> | <img src="https://drive.google.com/uc?export=view&id=1ZEuetvAu7JIiFZ4qBMtKgsnQ9lqkyeAp" width="200"> | <img src="https://drive.google.com/uc?export=view&id=1-Yij-Fqg48IL3u4VgPVXUo5wuFyxfSVL" width="200">

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
