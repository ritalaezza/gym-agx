"""Simulation of cable pushing

This module creates the simulation files which will be used in cable_pushing environments.
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
import logging
import numpy as np

# Local modules
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, \
    dlo_encompass_point, all_segment_below_z, to_numpy_array
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "rubber_band"
# Simulation parameters
TIMESTEP = 1 / 1000
N_SUBSTEPS = 20
GRAVITY = True
# Rubber band parameters
DIAMETER = 0.0125
DLO_CIRCLE_STEPS = 20
RADIUS = 0.001  # meters
RESOLUTION = 800  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 0.5e6
YOUNG_MODULUS_TWIST = 0.5e6
YOUNG_MODULUS_STRETCH = 0.5e6

GROUND_LENGTH_X = 0.025
GROUND_LENGTH_Y = 0.025
GROUND_HEIGHT = 0.0025

# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

POLE_POSITIONS = [[0.0, 0.01], [-0.01, -0.01],[0.01, -0.01]]

# Ground Parameters
EYE = agx.Vec3(0, -0.1, 0.1)
CENTER = agx.Vec3(0, 0, 0)
UP = agx.Vec3(0., 0., 0.0)

# Control parameters
MAX_MOTOR_FORCE = 1
GOAL_MAX_Z = 0.0125


def create_pole(sim, position, material):

    x = position[0]
    y = position[1]

    # Lower part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder", shape=agxCollide.Cylinder(0.004, 0.005),
                           position=agx.Vec3(x,y,0.0),
                           rotation=rotation_cylinder,
                           motion_control=agx.RigidBody.KINEMATICS,
                           material=material)
    sim.add(cylinder)

    # Middle part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder_inner", shape=agxCollide.Cylinder(0.0035, 0.005),
                           position=agx.Vec3(x,y,0.005),
                           rotation=rotation_cylinder,
                           motion_control=agx.RigidBody.KINEMATICS,
                           material=material)
    sim.add(cylinder)

    # Upper part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder", shape=agxCollide.Cylinder(0.004, 0.005),
                           position=agx.Vec3(x,y,0.01),
                           rotation=rotation_cylinder,
                           motion_control=agx.RigidBody.KINEMATICS,
                           material=material)
    sim.add(cylinder)


def add_rendering(sim):
    app = agxOSG.ExampleApplication(sim)

    # Set renderer
    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    # Create scene graph for rendering
    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        node = agxOSG.createVisual(rb, root)
        if rb.getName() == "ground":
            agxOSG.setDiffuseColor(node, agxRender.Color.SlateGray())
        elif rb.getName() == "cylinder":
            agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
        elif rb.getName() == "cylinder_inner":
            agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
        elif rb.getName() == "gripper":
            agxOSG.setDiffuseColor(node, agxRender.Color.DarkBlue())
        elif "dlo" in  rb.getName():  # Cable segments
            agxOSG.setDiffuseColor(node, agxRender.Color(0.8, 0.2, 0.2, 1.0))
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.0)

    # Set rendering options
    scene_decorator = app.getSceneDecorator()
    scene_decorator.setEnableLogo(False)
    scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0,1.0, 1.0))

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

    # Define materials
    material_hard = agx.Material("Aluminum")
    material_hard_bulk = material_hard.getBulkMaterial()
    material_hard_bulk.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    material_hard_bulk.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # Create a ground plane
    ground = create_body(name="ground", shape=agxCollide.Box(GROUND_LENGTH_X,
                                                             GROUND_LENGTH_Y,
                                                             GROUND_HEIGHT),
                         position=agx.Vec3(0, 0, -0.005),
                         motion_control=agx.RigidBody.STATIC,
                         material=material_hard)
    sim.add(ground)

    # Creates poles
    for position in POLE_POSITIONS:
        create_pole(sim=sim, position=position, material=material_hard)

    # Create gripper
    gripper = create_body(name="gripper",
                         shape=agxCollide.Sphere(0.002),
                         position=agx.Vec3(-(DIAMETER),0.0,0.02),
                         motion_control=agx.RigidBody.DYNAMICS,
                         material=material_hard)
    gripper.getRigidBody("gripper").getGeometry("gripper").setEnableCollisions(False)

    sim.add(gripper)

    #Create base for pusher motors
    prismatic_base = create_locked_prismatic_base("gripper", gripper.getRigidBody("gripper"),
                                                  position_ranges=[(-GROUND_LENGTH_X+DIAMETER, GROUND_LENGTH_X+DIAMETER),
                                                                   (-GROUND_LENGTH_Y, GROUND_LENGTH_Y),
                                                                   (-0.1, 0.01)],
                                                  motor_ranges=[(-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE),
                                                                (-MAX_MOTOR_FORCE, MAX_MOTOR_FORCE)])

    sim.add(prismatic_base)

    # Create rope and set name + properties
    rubber_band = agxCable.Cable(RADIUS, RESOLUTION)
    rubber_band.setName("DLO")
    material_rubber_band= rubber_band.getMaterial()
    rubber_band_material = material_rubber_band.getBulkMaterial()
    rubber_band_material.setPoissonsRatio(PEG_POISSON_RATIO)
    properties = rubber_band.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    # Initialize dlo on circle
    steps = DLO_CIRCLE_STEPS
    for a in np.linspace(0.0, 2*np.pi - 2*np.pi/steps, steps):
        x = np.cos(a)*DIAMETER
        y = np.sin(a) *DIAMETER
        rubber_band.add(agxCable.FreeNode(x,y,0.02))

    sim.add(rubber_band)

    segments_cable = list()
    cable = agxCable.Cable.find(sim, "DLO")
    segment_iterator = cable.begin()
    n_segments = cable.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()

            mass_props = seg.getMassProperties()
            mass_props.setMass(mass_props.getMass())

            segments_cable.append(seg)
            segment_iterator.inc()

    # Get segments at ends and middle
    s0 = segments_cable[0]
    s1 = segments_cable[int(n_segments/2)]
    s2 = segments_cable[-1]

    # Add ball joint between gripper and rubber band
    f0 = agx.Frame()
    f1 = agx.Frame()
    ball_joint = agx.BallJoint(gripper.getRigidBody("gripper"), f0, s1,  f1)
    sim.add(ball_joint)

    # Connect ends of rubber band
    f0 = agx.Frame()
    f0.setLocalTranslate(0.0, 0.0, -2*np.pi*DIAMETER/cable.getNumSegments())
    f1 = agx.Frame()
    lock = agx.LockJoint(s0, f0, s2, f1)
    lock.setCompliance(1.0e-4)
    sim.add(lock)

    # Try to initialize dlo
    report = rubber_band.tryInitialize()
    if report.successful():
        print("Successful dlo initialization.")
    else:
        print(report.getActualError())

    # Add rope to simulation
    sim.add(rubber_band)

    # Set rope material
    material_rubber_band = rubber_band.getMaterial()
    material_rubber_band.setName("rope_material")

    contactMaterial = sim.getMaterialManager().getOrCreateContactMaterial(material_hard, material_rubber_band)
    contactMaterial.setYoungsModulus(1e12)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contactMaterial.setFrictionModel(fm)

    # Add keyboard listener
    motor_x = sim.getConstraint1DOF("gripper_joint_base_x").getMotor1D()
    motor_y = sim.getConstraint1DOF("gripper_joint_base_y").getMotor1D()
    motor_z = sim.getConstraint1DOF("gripper_joint_base_z").getMotor1D()
    key_motor_map = {agxSDK.GuiEventListener.KEY_Up: (motor_y, 0.2),
                     agxSDK.GuiEventListener.KEY_Down: (motor_y, -0.2),
                     agxSDK.GuiEventListener.KEY_Right: (motor_x, 0.2),
                     agxSDK.GuiEventListener.KEY_Left: (motor_x, -0.2),
                     65365: (motor_z, 0.2),
                     65366: (motor_z, -0.2)}
    sim.add(KeyboardMotorHandler(key_motor_map))

    rbs = rubber_band.getRigidBodies()
    for i in range(len(rbs)):
        rbs[i].setName('dlo_' + str(i+1))

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

    # Add app
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable, '--window', '600', '600'] + args))
    app.setTimeStep(TIMESTEP)
    app.setCameraHome(EYE, CENTER, UP)
    app.initSimulation(sim, True)

    dlo = agxCable.Cable.find(sim, "DLO")

    def is_goal_reached():
        points_encompassed = 0
        for j in range(0,3):

            is_correct_height = all_segment_below_z(dlo, max_z=GOAL_MAX_Z)
            is_within_polygon = False

            if is_correct_height:
               is_within_polygon = dlo_encompass_point(dlo, POLE_POSITIONS[j])

            if is_within_polygon and is_correct_height:
                points_encompassed += 1

        if points_encompassed >= 3:
            return True
        else:
            return False

    for _ in range(10000):
        sim.stepForward()
        app.executeOneStepWithGraphics()

        if is_goal_reached():
            print("Success!")



if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
