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
from gym_agx.utils.agx_utils import create_body, save_simulation, save_goal_simulation
from gym_agx.utils.agx_classes import ContactEventListenerRigidBody
from gym_agx.utils.utils import harmonic_trajectory
from yumi import build_yumi

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = 'bend_wire_hinge_yumi'
# Simulation Parameters
N_SUBSTEPS = 10
TIMESTEP = 1 / 400  # seconds (eq. 100 Hz)
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
YUMI_GRIPPER_OFFSET = 0.134
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
EYE = agx.Vec3(1, 0.1, 0.3)
CENTER = agx.Vec3(0.3, 0, 0.3)
UP = agx.Vec3(0., 0., 1.)
# Control parameters
FORCE_RANGE = 2.5  # N


JOINT_NAMES_REV = ['yumi_joint_1_l', 'yumi_joint_2_l', 'yumi_joint_7_l', 'yumi_joint_3_l', 'yumi_joint_4_l',
                      'yumi_joint_5_l', 'yumi_joint_6_l',
                      'yumi_joint_1_r', 'yumi_joint_2_r', 'yumi_joint_7_r', 'yumi_joint_3_r', 'yumi_joint_4_r',
                      'yumi_joint_5_r', 'yumi_joint_6_r']

JOINT_INIT_POS = [1.1368918259774035, -2.2402857573326216, -1.124104175794654, 0.6809925673250303, -1.081411999152352,
                  1.3495032182496915, -0.12787275723643968, 0.0, 0.0, -1.1406615175214208, -2.2399084451704114,
                  1.124356847534356, 0.6809493971528793, 1.0810503812000616, 1.349087587171468, 0.12816659866109675,
                  0.0, 0.0]


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

    # build yumi into scene
    build_yumi(sim, JOINT_INIT_POS)

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

    # Create cable
    cable = agxCable.Cable(RADIUS, RESOLUTION)
    cable.setName("DLO")

    # Create Frames for each gripper:
    # Cables are attached passing through the attachment point along the Z axis of the body's coordinate frame.
    # The translation specified in the transformation is relative to the body and not the world
    left_transform = agx.AffineMatrix4x4()
    left_transform.setTranslate(0, 0, YUMI_GRIPPER_OFFSET + SIZE_GRIPPER + RADIUS)
    left_transform.setRotate(agx.Vec3.Z_AXIS(), agx.Vec3.X_AXIS())  # Rotation matrix which switches Z with Y
    frame_left = agx.Frame(left_transform)

    right_transform = agx.AffineMatrix4x4()
    right_transform.setTranslate(0, 0, YUMI_GRIPPER_OFFSET + SIZE_GRIPPER + RADIUS)
    right_transform.setRotate(agx.Vec3.Z_AXIS(), agx.Vec3.X_AXIS())  # Rotation matrix which switches Z with -Y
    frame_right = agx.Frame(right_transform)

    # Cable nodes
    cable.add(agxCable.FreeNode(agx.Vec3(0.3, -LENGTH/2 + SIZE_GRIPPER + RADIUS, 0.2)))  # Fix cable to gripper_left
    cable.add(agxCable.FreeNode(agx.Vec3(0.3, LENGTH/2 - SIZE_GRIPPER - RADIUS, 0.2)))  # Fix cable to gripper_right

    # Try to initialize cable
    report = cable.tryInitialize()
    if report.successful():
        logger.debug("Successful cable initialization.")
    else:
        logger.error(report.getActualError())

    actual_length = report.getLength()

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
    segment_right = iterator.getRigidBody()
    segment_right.setName('dlo_' + str(count))
    while not iterator.isEnd():
        count += 1
        segment_left = iterator.getRigidBody()
        segment_left.setName('dlo_' + str(count))
        # set name of geometry, can be used for collision event listener
        geo = segment_left.getGeometries()
        for j in range(len(geo)):
            geo[j].setName('dlo')
        iterator.inc()

    # Add hinge constraints
    hinge_joint_left = agx.Hinge(sim.getRigidBody("gripper_l_base"), frame_left, segment_left)
    hinge_joint_left.setName('hinge_joint_left')
    motor_left = hinge_joint_left.getMotor1D()
    motor_left.setEnable(True)
    motor_left_param = motor_left.getRegularizationParameters()
    motor_left_param.setCompliance(1e12)
    motor_left.setLockedAtZeroSpeed(False)
    lock_left = hinge_joint_left.getLock1D()
    lock_left.setEnable(False)
    # range_left = hinge_joint_left.getRange1D()
    # range_left.setEnable(True)
    # range_left.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    sim.add(hinge_joint_left)

    hinge_joint_right = agx.Hinge(sim.getRigidBody("gripper_r_base"), frame_right, segment_right)
    hinge_joint_right.setName('hinge_joint_right')
    motor_right = hinge_joint_right.getMotor1D()
    motor_right.setEnable(True)
    motor_right_param = motor_right.getRegularizationParameters()
    motor_right_param.setCompliance(1e12)
    motor_right.setLockedAtZeroSpeed(False)
    lock_right = hinge_joint_right.getLock1D()
    lock_right.setEnable(False)
    # range_right = hinge_joint_right.getRange1D()
    # range_right.setEnable(True)
    # range_right.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    sim.add(hinge_joint_right)

    ignore_contact = ['dlo']
    sim.addEventListener(
        ContactEventListenerRigidBody('contact_gripper_r_finger_r', sim.getRigidBody('gripper_r_finger_r'), ignore_contact))
    sim.addEventListener(
        ContactEventListenerRigidBody('contact_gripper_r_finger_l', sim.getRigidBody('gripper_r_finger_l'), ignore_contact))

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

    yumi = sim.getAssembly('yumi')
    yumi.getConstraint1DOF(JOINT_NAMES_REV[0]).getMotor1D().setSpeed(float(-0.01))

    n_seconds = 20
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))


    count = 0
    previous_velocity = 0
    for k in range(n_steps):
        app.executeOneStepWithGraphics()

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # Save goal simulation to file (but first make grippers static, disable collisions, remove clutter and rename)
    cable = agxCable.Cable.find(sim, "DLO")
    cable.setName("DLO_goal")
    success = save_goal_simulation(sim, FILE_NAME, ['ground'])
    if success:
        logger.debug("Goal simulation saved!")
    else:
        logger.debug("Goal simulation not saved!")


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)