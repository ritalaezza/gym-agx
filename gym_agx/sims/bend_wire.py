"""Simulation for BendWire environment

This module creates the simulation files which will be used in BendWire environments.
TODO: Instead of setting all parameters in this file, there should be a parameter file (e.g. YAML or XML).
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxRender
import agxSDK
import agxCable
import agxIO
import agxOSG

# Python modules
import logging
import math
import sys

# Local modules
from gym_agx.utils.agx_utils import create_body, save_simulation
from gym_agx.utils.utils import sinusoidal_trajectory

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = 'bend_wire_hinge'
# Simulation Parameters
N_SUBSTEPS = 2
TIMESTEP = 1 / 100  # seconds (eq. 100 Hz)
LENGTH = 0.1  # meters
RADIUS = LENGTH / 100  # meters
LENGTH += 2 * RADIUS  # meters
RESOLUTION = 1000  # segments per meter
GRAVITY = True
# Aluminum Parameters
POISSON_RATIO = 0.35  # no unit
YOUNG_MODULUS = 69e9  # Pascals
YIELD_POINT = 5e7  # Pascals
# Rendering Parameters
GROUND_WIDTH = 0.0001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
EYE = agx.Vec3(LENGTH / 2, -5 * LENGTH, 0)
CENTER = agx.Vec3(LENGTH / 2, 0, 0)
UP = agx.Vec3(0., 0., 1.)
# Control parameters
FORCE_RANGE = 2.5  # N


def add_rendering(sim):
    camera_distance = 0.5
    light_pos = agx.Vec4(LENGTH / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)

    app.setAutoStepping(False)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        if rb.getName() == "ground":
            ground_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(ground_node, agxRender.Color.Gray())
        elif rb.getName() == "gripper_left":
            gripper_left_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_left_node, agxRender.Color.Red())
        elif rb.getName() == "gripper_right":
            gripper_right_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_right_node, agxRender.Color.Blue())
        else:  # Cable segments
            cable_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(cable_node, agxRender.Color.Green())

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    return app


def build_simulation():
    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # By default the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
    # too by creating an agx.PointGravityField for example).
    # AGX uses a right-hand coordinate system (That is Z defines UP. X is right, and Y is into the screen)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    logger.debug("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    logger.debug("new dt = {}".format(dt))

    # Create a ground plane for reference
    ground = create_body(name="ground", shape=agxCollide.Box(LENGTH, LENGTH, GROUND_WIDTH),
                         position=agx.Vec3(LENGTH / 2, 0, -(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH)),
                         motion_control=agx.RigidBody.STATIC)
    sim.add(ground)

    # Create cable
    cable = agxCable.Cable(RADIUS, RESOLUTION)
    cable.setName("DLO")

    gripper_left = create_body(name="gripper_left",
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=agx.Vec3(0, 0, 0), motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_left)

    gripper_right = create_body(name="gripper_right",
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=agx.Vec3(LENGTH, 0, 0),
                                motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_right)

    logger.info("Mass of grippers: {}".format(sim.getRigidBody("gripper_right").calculateMass()))

    # Create Frames for each gripper:
    # Cables are attached passing through the attachment point along the Z axis of the body's coordinate frame.
    # The translation specified in the transformation is relative to the body and not the world
    left_transform = agx.AffineMatrix4x4()
    left_transform.setTranslate(SIZE_GRIPPER + RADIUS, 0, 0)
    left_transform.setRotate(agx.Vec3.Z_AXIS(), agx.Vec3.Y_AXIS())  # Rotation matrix which switches Z with Y
    frame_left = agx.Frame(left_transform)

    right_transform = agx.AffineMatrix4x4()
    right_transform.setTranslate(- SIZE_GRIPPER - RADIUS, 0, 0)
    right_transform.setRotate(agx.Vec3.Z_AXIS(), -agx.Vec3.Y_AXIS())  # Rotation matrix which switches Z with -Y
    frame_right = agx.Frame(right_transform)

    cable.add(agxCable.FreeNode(agx.Vec3(SIZE_GRIPPER + RADIUS, 0, 0)))  # Fix cable to gripper_left
    cable.add(agxCable.FreeNode(agx.Vec3(LENGTH - SIZE_GRIPPER - RADIUS, 0, 0)))  # Fix cable to gripper_right

    # Try to initialize cable
    report = cable.tryInitialize()
    if report.successful():
        logger.debug("Successful cable initialization.")
    else:
        logger.error(report.getActualError())

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

    # Add cable to simulation
    sim.add(cable)

    # Add segment names and get first and last segment
    count = 1
    iterator = cable.begin()
    segment_left = iterator.getRigidBody()
    segment_left.setName('dlo_' + str(count))
    while not iterator.isEnd():
        count += 1
        segment_right = iterator.getRigidBody()
        segment_right.setName('dlo_' + str(count))
        iterator.inc()

    # Add hinge constraints
    hinge_joint_left = agx.Hinge(sim.getRigidBody("gripper_left"), frame_left, segment_left)
    hinge_joint_left.setName('hinge_joint_left')
    motor_left = hinge_joint_left.getMotor1D()
    motor_left.setEnable(True)
    motor_left_param = motor_left.getRegularizationParameters()
    motor_left_param.setCompliance(1e12)
    motor_left.setLockedAtZeroSpeed(False)
    lock_left = hinge_joint_left.getLock1D()
    lock_left.setEnable(False)
    range_left = hinge_joint_left.getRange1D()
    range_left.setEnable(True)
    range_left.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    sim.add(hinge_joint_left)

    hinge_joint_right = agx.Hinge(sim.getRigidBody("gripper_right"), frame_right, segment_right)
    hinge_joint_right.setName('hinge_joint_right')
    motor_right = hinge_joint_right.getMotor1D()
    motor_right.setEnable(True)
    motor_right_param = motor_right.getRegularizationParameters()
    motor_right_param.setCompliance(1e12)
    motor_right.setLockedAtZeroSpeed(False)
    lock_right = hinge_joint_right.getLock1D()
    lock_right.setEnable(False)
    range_right = hinge_joint_right.getRange1D()
    range_right.setEnable(True)
    range_right.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    sim.add(hinge_joint_right)

    # Add prismatic constraints
    prismatic_frame_right = agx.PrismaticFrame(agx.Vec3(0, 0, 0), agx.Vec3.X_AXIS())
    prismatic_joint_right = agx.Prismatic(prismatic_frame_right, sim.getRigidBody("gripper_right"))
    prismatic_joint_right.setName('prismatic_joint_right')
    prismatic_joint_right.setEnableComputeForces(True)
    motor_right = prismatic_joint_right.getMotor1D()
    motor_right.setLockedAtZeroSpeed(False)
    motor_right.setEnable(True)
    motor_right.setForceRange(-FORCE_RANGE, FORCE_RANGE)
    lock_right = prismatic_joint_right.getLock1D()
    lock_right.setEnable(False)
    range_right = prismatic_joint_right.getRange1D()
    range_right.setEnable(True)
    range_right.setRange(agx.RangeReal(-LENGTH, 0))
    sim.add(prismatic_joint_right)

    prismatic_frame_left = agx.PrismaticFrame(agx.Vec3(LENGTH, 0, 0), agx.Vec3.X_AXIS())
    prismatic_joint_left = agx.Prismatic(prismatic_frame_left, sim.getRigidBody("gripper_left"))
    prismatic_joint_left.setName('prismatic_joint_left')
    lock = prismatic_joint_left.getLock1D()
    lock.setEnable(True)
    sim.add(prismatic_joint_left)

    return sim


# Build and save scene to file
def main(args):
    # Build simulation object
    sim = build_simulation()

    # Save simulation to file
    success = save_simulation(sim, FILE_NAME)
    if success:
        logger.debug("Simulation saved!")
    else:
        logger.debug("Simulation not saved!")

    # Render simulation
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(sim, True)  # This changes timestep and Gravity!
    sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        logger.info("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # gripper = sim.getRigidBody('gripper_right')
    prismatic_joint = sim.getConstraint1DOF('prismatic_joint_right')
    # hinge_joint = sim.getConstraint1DOF('hinge_joint_right')
    prismatic_motor = prismatic_joint.getMotor1D()
    # hinge_motor = hinge_joint.getMotor1D()
    # hinge_joint_params = hinge_motor.getRegularizationParameters()

    n_seconds = 20
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    period = 12  # seconds
    amplitude = LENGTH / 4
    rad_frequency = 2 * math.pi * (1 / period)
    # decay = 0.1
    # compliance = 1e12
    count = 0
    previous_velocity = 0
    for k in range(n_steps):
        app.executeOneStepWithGraphics()
        velocity_x = sinusoidal_trajectory(amplitude, rad_frequency, k * TIMESTEP * N_SUBSTEPS)
        if velocity_x * previous_velocity < 0:
            if count == 0:
                amplitude /= 2
            else:
                amplitude = 0
            count += 1
        previous_velocity = velocity_x
        # compliance = compliance * math.exp(-decay*t)
        # hinge_joint_params.setCompliance(compliance)
        prismatic_motor.setSpeed(velocity_x)

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # Save goal simulation to file
    success = save_simulation(sim, FILE_NAME + "_goal")
    if success:
        logger.debug("Goal simulation saved!")
    else:
        logger.debug("Goal simulation not saved!")


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
