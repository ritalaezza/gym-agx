"""Simulation for BendWire environment

This module creates the simulation files which will them be used in BendWire environments.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Section breaks are created by resuming unindented text. Section breaks
are also implicitly created anytime a new section starts.

Attributes:
    module_level_variable1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Todo:
    * For module TODOs
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxCable
import agxIO
import agxOSG

# Python modules
import math
import sys

# Local modules
from gym_agx.utils.agx_utils import create_body, save_simulation
from gym_agx.utils.utils import sinusoidal_trajectory


FILE_NAME = 'bend_wire'
# Simulation Parameters
TIMESTEP = 1 / 100     # seconds (eq. 100 Hz)
LENGTH = 0.1           # meters
RADIUS = LENGTH / 100  # meters
LENGTH += 2*RADIUS     # meters
RESOLUTION = 1000      # segments per meter
GRAVITY = False
# Aluminum Parameters
POISSON_RATIO = 0.35   # no unit
YOUNG_MODULUS = 69e9   # Pascals
YIELD_POINT = 5e7      # Pascals
# Rendering Parameters
GROUND_WIDTH = 0.0001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO*RADIUS
EYE = agx.Vec3(LENGTH / 2, -5 * LENGTH, 0)
CENTER = agx.Vec3(LENGTH / 2, 0, 0)
UP = agx.Vec3(0., 0., 1.)


def add_rendering(sim, length):
    camera_distance = 0.5
    light_pos = agx.Vec4(length / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)
    app.setAutoStepping(False)

    app.setEnableDebugRenderer(True)
    app.setEnableOSGRenderer(False)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)

    return app


def build_simulation():
    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # By default the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
    # too by creating an agx.PointGravityField for example).
    # AGX uses a right-hand coordinate system (That is Z defines UP. X is right, and Y is into the screen)
    if not GRAVITY:
        print("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    print("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    print("new dt = {}".format(dt))

    # Create a ground plane for reference
    ground, ground_geom = create_body(sim, name="ground", shape=agxCollide.Box(LENGTH, LENGTH, GROUND_WIDTH),
                                      position=agx.Vec3(LENGTH/2, 0, -(GROUND_WIDTH + SIZE_GRIPPER/2 + LENGTH)),
                                      motionControl=agx.RigidBody.STATIC)

    # Create cable
    cable = agxCable.Cable(RADIUS, RESOLUTION)

    # Create two grippers one static one kinematic
    gripper_left, gripper_left_geom = create_body(sim, name="gripper_left",
                                                  shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                                  position=agx.Vec3(0, 0, 0), motionControl=agx.RigidBody.STATIC)

    gripper_right, gripper_right_geom = create_body(sim, name="gripper_right",
                                                    shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                                    position=agx.Vec3(LENGTH, 0, 0),
                                                    motionControl=agx.RigidBody.KINEMATICS)

    # Create LockJoints for each gripper:
    # Cables are attached passing through the attachment point along the Z axis of the body's coordinate frame.
    # The translation specified in the transformation is relative to the body and not the world
    left_transform = agx.AffineMatrix4x4()
    left_transform.setTranslate(SIZE_GRIPPER + RADIUS, 0, 0)
    left_transform.setRotate(agx.Vec3.Z_AXIS(), agx.Vec3.X_AXIS())  # Rotation matrix which switches Z with X
    cable.add(agxCable.BodyFixedNode(gripper_left, left_transform))  # Fix cable to gripper_left

    right_transform = agx.AffineMatrix4x4()
    right_transform.setTranslate(- SIZE_GRIPPER - RADIUS, 0, 0)
    right_transform.setRotate(agx.Vec3.Z_AXIS(), agx.Vec3.X_AXIS())  # Rotation matrix which switches Z with X
    cable.add(agxCable.BodyFixedNode(gripper_right, right_transform))  # Fix cable to gripper_right

    # Set cable name and properties
    cable.setName("DLO")
    # properties = cable.getCableProperties()
    # properties.setYoungsModulus(YOUNG_MODULUS, agxCable.BEND)
    # properties.setYoungsModulus(YOUNG_MODULUS, agxCable.TWIST)
    # properties.setYoungsModulus(YOUNG_MODULUS, agxCable.STRETCH)

    # Add cable plasticity
    plasticity = agxCable.CablePlasticity()
    plasticity.setYieldPoint(YIELD_POINT, agxCable.BEND)  # set torque required for permanent deformation
    cable.addComponent(plasticity)  # NOTE: Stretch direction is always elastic

    # Define material
    material = agx.Material("Aluminum")
    bulk_material = material.getBulkMaterial()
    bulk_material.setPoissonsRatio(POISSON_RATIO)
    bulk_material.setYoungsModulus(YOUNG_MODULUS)
    cable.setMaterial(material)

    # Set cable damage name and weights
    damage = agxCable.CableDamage()
    damage.setName("DLO_damage")
    damage.setStretchDeformationWeight(10.0)
    damage.setBendDeformationWeight(0.0)
    damage.setTwistDeformationWeight(1.0)

    # Add cable damage
    cable.addComponent(damage)

    # Try to initialize cable
    report = cable.tryInitialize()
    if report.successful():
        print("Successful cable initialization.")
    else:
        print(report.getActualError())

    # Add cable to simulation
    sim.add(cable)

    # Rename constraints and set enable force computation
    constraints = sim.getConstraints()
    for i, c in enumerate(constraints):
        right_attachment = c.getAttachment(gripper_right)
        left_attachment = c.getAttachment(gripper_left)
        if right_attachment:
            c.setName('gripper_right_constraint')
            c.setEnableComputeForces(True)
        elif left_attachment:
            c.setName('gripper_left_constraint')
            c.setEnableComputeForces(True)
        else:
            print('Constraint #{}'.format(i))
    return sim


# Build and save scene to file
def main(args):
    # Build simulation object
    sim = build_simulation()

    # Print list of objects in terminal
    rbs = sim.getRigidBodies()

    for i, rb in enumerate(rbs):
        name = rbs[i].getName()
        if name == "":
            print("Object: segment_{}".format(i-2))
        else:
            print("Object: {}".format(rbs[i].getName()))
        print("Position:")
        print(rbs[i].getPosition())
        print("Velocity:")
        print(rbs[i].getVelocity())
        print("Rotation:")
        print(rbs[i].getRotation())
        print("Angular velocity:")
        print(rbs[i].getAngularVelocity())

    # Save simulation to file
    save_simulation(sim, FILE_NAME)

    # Render simulation
    app = add_rendering(sim, LENGTH)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(sim, True)       # This changes timestep!
    sim.setTimeStep(TIMESTEP)
    gripper = sim.getRigidBody('gripper_right')

    n_steps = 1000  # implies minimum period of 10 seconds.
    period = 12     # seconds
    amplitude = LENGTH / 4
    rad_frequency = 2 * math.pi * (1 / period)
    for k in range(n_steps):
        t = sim.getTimeStamp()
        velocity_x = sinusoidal_trajectory(amplitude, rad_frequency, t)
        print("k = {}, t = {}, v_x = {}".format(k, t, velocity_x))
        gripper.setVelocity(velocity_x, 0, 0)
        sim.stepForward()
        app.executeOneStepWithGraphics()

    # Save goal simulation to file
    save_simulation(sim, FILE_NAME + "_goal")


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
