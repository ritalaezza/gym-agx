"""Simulation for ORing environment

This module creates the simulation files which will be used in ORing environments.
TODO: Instead of setting all parameters in this file, there should be a parameter file (e.g. YAML or XML).
"""
# AGX Dynamics imports
import agx
import agxPython
import agxCollide
import agxSDK
import agxRender
import agxIO
import agxOSG

# Python modules
import logging
import math
import sys

# Local modules
from gym_agx.utils.agx_utils import create_body, create_ring, create_universal_prismatic_base, KeyboardMotorHandler
from gym_agx.utils.agx_utils import save_simulation

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = 'insert_o_ring'
# Simulation Parameters
GRAVITY = True
N_SUBSTEPS = 20
TIMESTEP = 1 / 1000  # seconds
# Ring and cylinder parameters
CYLINDER_RADIUS = 0.02  # meters
CYLINDER_LENGTH = 0.08  # meters
GROOVE_DEPTH = 0.002  # meters
RING_RADIUS = CYLINDER_RADIUS - GROOVE_DEPTH  # meters
CIRCUMFERENCE = 2 * math.pi * RING_RADIUS
NUM_RING_ELEMENTS = 40  # number of segments
RING_SEGMENT_LENGTH = (CIRCUMFERENCE / NUM_RING_ELEMENTS)
RING_CROSS_SECTION = RING_SEGMENT_LENGTH / 4
GROOVE_WIDTH = RING_CROSS_SECTION * 2
RING_COMPLIANCE = [1e-5, 1.5e-3, 1e-5, 1, 1, 1e-5]
GRIPPER_COMPLIANCE = 1e-8
# Material Parameters
RUBBER_POISSON_RATIO = 0.49  # no unit
RUBBER_YOUNG_MODULUS = 0.05e9  # Pascals
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
CONTACT_YOUNG_MODULUS = 67e12
# Rendering Parameters
GROUND_WIDTH = 0.001  # meters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RING_CROSS_SECTION
EYE = agx.Vec3(CYLINDER_LENGTH / 2, -5 * CYLINDER_LENGTH, 0)
CENTER = agx.Vec3(0, 0, CYLINDER_LENGTH)
UP = agx.Vec3(0., 0., 1.)
COLOR_RING = agxRender.Color.Gold()
COLOR_CYLINDER = agxRender.Color.DimGray()
COLOR_GROUND = agxRender.Color.Black()
# Control parameters
MAX_MOTOR_FORCE = 2.5  # N
RIGHT_ELEMENT = "ring_15"
LEFT_ELEMENT = "ring_25"
BASE_CYLINDER_LENGTH = CYLINDER_LENGTH / 10
BASE_CYLINDER_RADIUS = CYLINDER_RADIUS / 10


def add_rendering(sim):
    camera_distance = 0.5
    light_pos = agx.Vec4(CYLINDER_LENGTH / 2, - camera_distance, camera_distance, 1.)
    light_dir = agx.Vec3(0., 0., -1.)

    app = agxOSG.ExampleApplication(sim)

    app.setAutoStepping(False)
    root = app.getRoot()

    rbs = sim.getRigidBodies()

    for rb in rbs:
        name = rb.getName()
        node = agxOSG.createVisual(rb, root, 2.0)
        if "ring" in name:
            agxOSG.setDiffuseColor(node, COLOR_RING)
        elif "ground" in name:
            agxOSG.setDiffuseColor(node, COLOR_GROUND)
        elif "cylinder" in name:
            agxOSG.setDiffuseColor(node, COLOR_CYLINDER)
        elif "gripper_right" == name:
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 0.0, 1.0, 1.0))
        elif "gripper_left" == name:
            agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 0.0, 0.0, 1.0))
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.2)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    # for i in range(1, NUM_RING_ELEMENTS + 2):
    #     ring_constraint = sim.getConstraint("ring_constraint_" + str(i))
    #     agxOSG.createAxes(ring_constraint, root, 0.005)

    # gripper_left_joint_rb = sim.getConstraint("gripper_left_joint_rb")
    # agxOSG.createAxes(gripper_left_joint_rb, root, 0.005)

    # gripper_right_joint_rb = sim.getConstraint("gripper_right_joint_rb")
    # agxOSG.createAxes(gripper_right_joint_rb, root, 0.005)
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
    ground = create_body(name="ground", shape=agxCollide.Box(CYLINDER_LENGTH * 2, CYLINDER_LENGTH * 2, GROUND_WIDTH),
                         position=agx.Vec3(0, 0, 0), motion_control=agx.RigidBody.STATIC)
    sim.add(ground)

    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())

    material_cylinder = agx.Material("Aluminum")
    bulk_material_cylinder = material_cylinder.getBulkMaterial()
    bulk_material_cylinder.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    bulk_material_cylinder.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # Create cylinders
    bottom_cylinder = create_body(name="bottom_cylinder",
                                  shape=agxCollide.Cylinder(CYLINDER_RADIUS, 3 / 4 * CYLINDER_LENGTH),
                                  position=agx.Vec3(0, 0, GROUND_WIDTH + (3 / 4 * CYLINDER_LENGTH) / 2),
                                  rotation=rotation_cylinder,
                                  motion_control=agx.RigidBody.STATIC,
                                  material=material_cylinder)
    sim.add(bottom_cylinder)

    middle_cylinder = create_body(name="middle_cylinder",
                                  shape=agxCollide.Cylinder(CYLINDER_RADIUS - GROOVE_DEPTH, GROOVE_WIDTH),
                                  position=agx.Vec3(0, 0, GROUND_WIDTH + (3 / 4 * CYLINDER_LENGTH) + GROOVE_WIDTH / 2),
                                  rotation=rotation_cylinder,
                                  motion_control=agx.RigidBody.STATIC,
                                  material=material_cylinder)
    sim.add(middle_cylinder)

    top_cylinder = create_body(name="top_cylinder",
                               shape=agxCollide.Cylinder(CYLINDER_RADIUS, 1 / 4 * CYLINDER_LENGTH),
                               position=agx.Vec3(0, 0, GROUND_WIDTH + (3 / 4 * CYLINDER_LENGTH) + GROOVE_WIDTH + (
                                       1 / 4 * CYLINDER_LENGTH) / 2),
                               rotation=rotation_cylinder,
                               motion_control=agx.RigidBody.STATIC,
                               material=material_cylinder)
    sim.add(top_cylinder)

    material_ring = agx.Material("Rubber")
    bulk_material_ring = material_ring.getBulkMaterial()
    bulk_material_ring.setPoissonsRatio(RUBBER_POISSON_RATIO)
    bulk_material_ring.setYoungsModulus(RUBBER_YOUNG_MODULUS)

    ring = create_ring(name="ring", radius=RING_RADIUS,
                       element_shape=agxCollide.Capsule(RING_CROSS_SECTION, RING_SEGMENT_LENGTH),
                       # element_shape=agxCollide.Cylinder(RING_CROSS_SECTION, RING_SEGMENT_LENGTH),
                       # element_shape=agxCollide.Sphere(RING_CROSS_SECTION),
                       # element_shape=agxCollide.Box(RING_CROSS_SECTION, RING_CROSS_SECTION, RING_CROSS_SECTION),
                       num_elements=NUM_RING_ELEMENTS,
                       constraint_type=agx.LockJoint,
                       rotation_shift=math.pi / 2,
                       translation_shift=RING_SEGMENT_LENGTH / 2,  # + RING_CROSS_SECTION,
                       compliance=RING_COMPLIANCE,
                       center=agx.Vec3(0, 0, CYLINDER_LENGTH + 2 * RING_RADIUS),  # normal=agx.Vec3(0, 0.5, 0.5),
                       material=material_ring)
    sim.add(ring)

    left_ring_element = sim.getRigidBody(LEFT_ELEMENT)
    right_ring_element = sim.getRigidBody(RIGHT_ELEMENT)

    gripper_left = create_body(name="gripper_left",
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=left_ring_element.getPosition(),
                               rotation=agx.OrthoMatrix3x3(left_ring_element.getRotation()),
                               motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_left)

    gripper_right = create_body(name="gripper_right",
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=right_ring_element.getPosition(),
                                rotation=agx.OrthoMatrix3x3(right_ring_element.getRotation()),
                                motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_right)

    # Disable collisions for grippers
    gripper_left_body = sim.getRigidBody("gripper_left")
    gripper_left_body.getGeometry("gripper_left").setEnableCollisions(False)
    gripper_right_body = sim.getRigidBody("gripper_right")
    gripper_right_body.getGeometry("gripper_right").setEnableCollisions(False)

    frame_element = agx.Frame()
    frame_gripper = agx.Frame()

    result = agx.Constraint.calculateFramesFromBody(agx.Vec3(RING_CROSS_SECTION, 0, 0), agx.Vec3(1, 0, 0),
                                                    left_ring_element, frame_element, gripper_left_body, frame_gripper)
    print(result)

    lock_joint_left = agx.LockJoint(gripper_left_body, frame_gripper, left_ring_element, frame_element)
    lock_joint_left.setName('lock_joint_left')
    lock_joint_left.setCompliance(GRIPPER_COMPLIANCE)
    sim.add(lock_joint_left)

    frame_gripper = agx.Frame()

    result = agx.Constraint.calculateFramesFromBody(agx.Vec3(RING_CROSS_SECTION, 0, 0), agx.Vec3(1, 0, 0),
                                                    right_ring_element, frame_element, gripper_right_body,
                                                    frame_gripper)
    print(result)

    lock_joint_right = agx.LockJoint(gripper_right_body, frame_gripper, right_ring_element, frame_element)
    lock_joint_right.setName('lock_joint_right')
    lock_joint_right.setCompliance(GRIPPER_COMPLIANCE)
    sim.add(lock_joint_right)

    # Create contact materials
    contact_material = sim.getMaterialManager().getOrCreateContactMaterial(material_cylinder, material_ring)
    contact_material.setYoungsModulus(CONTACT_YOUNG_MODULUS)

    # Create friction model
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contact_material.setFrictionModel(fm)

    # Create bases for gripper motors
    prismatic_base_left = create_universal_prismatic_base("gripper_left", gripper_left_body)
    sim.add(prismatic_base_left)
    prismatic_base_right = create_universal_prismatic_base("gripper_right", gripper_right_body)
    sim.add(prismatic_base_right)

    # Add keyboard listener
    left_motor_x = sim.getConstraint1DOF("gripper_left_joint_base_x").getMotor1D()
    left_motor_y = sim.getConstraint1DOF("gripper_left_joint_base_y").getMotor1D()
    left_motor_z = sim.getConstraint1DOF("gripper_left_joint_base_z").getMotor1D()
    left_motor_rb_1 = agx.Motor1D_safeCast(
        sim.getConstraint("gripper_left_joint_rb").getSecondaryConstraintGivenName("gripper_left_joint_rb_motor_1"))
    left_motor_rb_2 = agx.Motor1D_safeCast(
        sim.getConstraint("gripper_left_joint_rb").getSecondaryConstraintGivenName("gripper_left_joint_rb_motor_2"))
    right_motor_x = sim.getConstraint1DOF("gripper_right_joint_base_x").getMotor1D()
    right_motor_y = sim.getConstraint1DOF("gripper_right_joint_base_y").getMotor1D()
    right_motor_z = sim.getConstraint1DOF("gripper_right_joint_base_z").getMotor1D()
    right_motor_rb_1 = agx.Motor1D_safeCast(
        sim.getConstraint("gripper_right_joint_rb").getSecondaryConstraintGivenName("gripper_right_joint_rb_motor_1"))
    right_motor_rb_2 = agx.Motor1D_safeCast(
        sim.getConstraint("gripper_right_joint_rb").getSecondaryConstraintGivenName("gripper_right_joint_rb_motor_2"))
    key_motor_map = {agxSDK.GuiEventListener.KEY_Right: (right_motor_x, 0.1),
                     agxSDK.GuiEventListener.KEY_Left: (right_motor_x, -0.1),
                     agxSDK.GuiEventListener.KEY_Up: (right_motor_y, 0.1),
                     agxSDK.GuiEventListener.KEY_Down: (right_motor_y, -0.1),
                     65365: (right_motor_z, 0.1),
                     65366: (right_motor_z, -0.1),
                     0x64: (left_motor_x, 0.1),
                     0x61: (left_motor_x, -0.1),
                     0x32: (left_motor_y, 0.1),  # 0x77 W is replaced with 2, due to prior shortcut
                     0x73: (left_motor_y, -0.1),
                     0x71: (left_motor_z, 0.1),
                     0x65: (left_motor_z, -0.1),
                     0x72: (left_motor_rb_2, 0.1),  # E
                     0x74: (left_motor_rb_2, -0.1),  # R
                     0x6f: (right_motor_rb_2, 0.1),  # O
                     0x70: (right_motor_rb_2, -0.1),  # P
                     0x7a: (left_motor_rb_1, 0.1),  # Z
                     0x78: (left_motor_rb_1, -0.1),  # X
                     0x6e: (right_motor_rb_1, 0.1),  # N
                     0x6d: (right_motor_rb_1, -0.1)}  # M
    sim.add(KeyboardMotorHandler(key_motor_map))

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

    n_seconds = 50
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    for k in range(n_steps):
        app.executeOneStepWithGraphics()

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
