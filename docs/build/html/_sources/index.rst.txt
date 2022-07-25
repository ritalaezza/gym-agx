Welcome to gym-agx's documentation!
===================================

This is a Python library for deformable object manipulation research. This libraly depends on `AGX Dynamics <https://www.algoryx.se/agx-dynamics/>`_, a real-time physics simulation engine. Currently, the package is focused on deformable linear object manipulation tasks e.g. ropes and cables. It includes both implicit and explicit shape control environments.

**Explicit Shape Control**: Tasks where the goal is to achieve a desired shape of the object, without necessarily caring about object's pose.

**Implicit Shape Control**: Tasks where the goal is to achieve a desired configuration of the object (relative to itself or other objects), without necessarily caring about object's exact shape.


Most of the documentation is auto-generated using `Sphinx <https://www.sphinx-doc.org/en/master/>`_. Below you can find the main modules which make up this library.

.. toctree::
   :maxdepth: 3

   gym_agx.envs
   gym_agx.rl
   gym_agx.sims
   gym_agx.utils
