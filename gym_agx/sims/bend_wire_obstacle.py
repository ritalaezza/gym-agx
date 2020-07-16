"""Simulation for bend_wire with obstacle environment

This module creates the simulation files which will be used in bend_wire with obstacle environments.
TODO: Instead of setting all parameters in this file, there should be a parameter file (e.g. YAML or XML).
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxCable
import agxIO
import agxOSG
import agxRender

# Python modules
import sys
import math
import logging

# Local modules
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, save_goal_simulation
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = 'bend_wire_obstacle_planar'
# Simulation parameters
N_SUBSTEPS = 2
TIMESTEP = 1 / 100
GRAVITY = True
RADIUS = 0.001
LENGTH = 0.3  # meters
LENGTH += 2 * RADIUS  # meters
RESOLUTION = 300  # segments per meter
CYLINDER_LENGTH = 0.1
CYLINDER_RADIUS = CYLINDER_LENGTH / 4

# Aluminum Parameters
POISSON_RATIO = 0.35  # no unit
YOUNG_MODULUS = 69e9  # Pascals (1e9)
YIELD_POINT = 5e7  # Pascals (1e7)
CONTACT_YOUNG_MODULUS = 67e12  # Pascals

# Rendering Parameters
GROUND_WIDTH = 0.0001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
EYE = agx.Vec3(0, -1.0, 0)
CENTER = agx.Vec3(0, 0, -2 * CYLINDER_RADIUS)
UP = agx.Vec3(0., 0., 1.)


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
            agxOSG.setDiffuseColor(gripper_left_node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
        elif rb.getName() == "gripper_right":
            gripper_right_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_right_node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
        elif rb.getName() == "obstacle":
            gripper_right_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(gripper_right_node, agxRender.Color.Gray())
        elif "dlo" in rb.getName():
            cable_node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(cable_node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
        else:
            node = agxOSG.createVisual(rb, root)
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.2)

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
                         position=agx.Vec3(0, 0, -(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH)),
                         motion_control=agx.RigidBody.STATIC)
    sim.add(ground)

    # Create two grippers one static one kinematic
    gripper_left = create_body(name="gripper_left",
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=agx.Vec3(-LENGTH / 2, 0, 0),
                               motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_left)

    gripper_right = create_body(name="gripper_right",
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=agx.Vec3(LENGTH / 2, 0, 0),
                                motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_right)

    gripper_left_body = gripper_left.getRigidBody("gripper_left")
    gripper_right_body = gripper_right.getRigidBody("gripper_right")

    # Create cable
    cable = agxCable.Cable(RADIUS, RESOLUTION)

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

    cable.add(agxCable.FreeNode(agx.Vec3(-LENGTH / 2 + SIZE_GRIPPER + RADIUS, 0, 0)))  # Fix cable to gripper_left
    cable.add(agxCable.FreeNode(agx.Vec3(LENGTH / 2 - SIZE_GRIPPER - RADIUS, 0, 0)))  # Fix cable to gripper_right

    material_cylinder = agx.Material("cylinder_material")
    bulk_material_cylinder = material_cylinder.getBulkMaterial()
    bulk_material_cylinder.setPoissonsRatio(POISSON_RATIO)
    bulk_material_cylinder.setYoungsModulus(YOUNG_MODULUS)

    cylinder = create_body(name="obstacle",
                           shape=agxCollide.Cylinder(CYLINDER_RADIUS, CYLINDER_LENGTH),
                           position=agx.Vec3(0, 0, -2 * CYLINDER_RADIUS), motion_control=agx.RigidBody.STATIC,
                           material=material_cylinder)
    sim.add(cylinder)

    # Set cable name and properties
    cable.setName("DLO")
    properties = cable.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS, agxCable.STRETCH)

    material_wire = cable.getMaterial()
    wire_material = material_wire.getBulkMaterial()
    wire_material.setPoissonsRatio(POISSON_RATIO)
    wire_material.setYoungsModulus(YOUNG_MODULUS)
    cable.setMaterial(material_wire)

    # Add cable plasticity
    plasticity = agxCable.CablePlasticity()
    plasticity.setYieldPoint(YIELD_POINT, agxCable.BEND)  # set torque required for permanent deformation
    plasticity.setYieldPoint(YIELD_POINT, agxCable.STRETCH)  # set torque required for permanent deformation
    cable.addComponent(plasticity)  # NOTE: Stretch direction is always elastic

    # Tell MaterialManager to create and return a contact material which will be used
    # when two geometries both with this material is in contact
    contact_material = sim.getMaterialManager().getOrCreateContactMaterial(material_cylinder, material_wire)
    contact_material.setYoungsModulus(CONTACT_YOUNG_MODULUS)

    # Create a Friction model, which we tell the solver to solve ITERATIVELY (faster)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contact_material.setFrictionModel(fm)

    # Try to initialize cable
    report = cable.tryInitialize()
    if report.successful():
        print("Successful cable initialization.")
    else:
        print(report.getActualError())

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
    motor_left.setEnable(False)
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
    motor_right.setEnable(False)
    motor_right_param = motor_right.getRegularizationParameters()
    motor_right_param.setCompliance(1e12)
    motor_right.setLockedAtZeroSpeed(False)
    lock_right = hinge_joint_right.getLock1D()
    lock_right.setEnable(False)
    range_right = hinge_joint_right.getRange1D()
    range_right.setEnable(True)
    range_right.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    sim.add(hinge_joint_right)

    # Create bases for gripper motors
    prismatic_base_left = create_locked_prismatic_base("gripper_left", gripper_left_body, compliance=0,
                                                       position_ranges=[(-LENGTH / 2 + CYLINDER_RADIUS,
                                                                         LENGTH / 2 - CYLINDER_RADIUS),
                                                                        (-CYLINDER_LENGTH / 3, CYLINDER_LENGTH / 3),
                                                                        (-(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH),
                                                                         0)],
                                                       lock_status=[False, True, False])
    sim.add(prismatic_base_left)
    prismatic_base_right = create_locked_prismatic_base("gripper_right", gripper_right_body, compliance=0,
                                                        position_ranges=[(-LENGTH / 2 + CYLINDER_RADIUS,
                                                                          LENGTH / 2 - CYLINDER_RADIUS),
                                                                         (-CYLINDER_LENGTH / 3, CYLINDER_LENGTH / 3),
                                                                         (-(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH),
                                                                          0)],
                                                        lock_status=[False, True, False])
    sim.add(prismatic_base_right)

    # Add keyboard listener
    left_motor_x = sim.getConstraint1DOF("gripper_left_joint_base_x").getMotor1D()
    left_motor_y = sim.getConstraint1DOF("gripper_left_joint_base_y").getMotor1D()
    left_motor_z = sim.getConstraint1DOF("gripper_left_joint_base_z").getMotor1D()
    right_motor_x = sim.getConstraint1DOF("gripper_right_joint_base_x").getMotor1D()
    right_motor_y = sim.getConstraint1DOF("gripper_right_joint_base_y").getMotor1D()
    right_motor_z = sim.getConstraint1DOF("gripper_right_joint_base_z").getMotor1D()
    key_motor_map = {agxSDK.GuiEventListener.KEY_Right: (right_motor_x, 0.1),
                     agxSDK.GuiEventListener.KEY_Left: (right_motor_x, -0.1),
                     agxSDK.GuiEventListener.KEY_Up: (right_motor_y, 0.1),
                     agxSDK.GuiEventListener.KEY_Down: (right_motor_y, -0.1),
                     65365: (right_motor_z, 0.1),
                     65366: (right_motor_z, -0.1),
                     0x64: (left_motor_x, 0.1),
                     0x61: (left_motor_x, -0.1),
                     0x32: (left_motor_y, 0.1),  # 0x77
                     0x73: (left_motor_y, -0.1),
                     0x71: (left_motor_z, 0.1),
                     0x65: (left_motor_z, -0.1)}
    sim.add(KeyboardMotorHandler(key_motor_map))

    return sim


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

    n_seconds = 30
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    for k in range(n_steps):
        app.executeOneStepWithGraphics()

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # Save goal simulation to file (but first make grippers static, remove clutter and rename)
    cable = agxCable.Cable.find(sim, "DLO")
    cable.setName("DLO_goal")
    success = save_goal_simulation(sim, FILE_NAME, ['ground', 'obstacle', 'gripper_left_prismatic_base',
                                                    'gripper_right_prismatic_base'])
    if success:
        logger.debug("Goal simulation saved!")
    else:
        logger.debug("Goal simulation not saved!")


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
