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
from gym_agx.utils.agx_utils import create_body, create_locked_prismatic_base, save_simulation, to_numpy_array
from gym_agx.utils.utils import point_inside_polygon, all_points_below_z
from gym_agx.utils.agx_classes import KeyboardMotorHandler

logger = logging.getLogger('gym_agx.sims')

FILE_NAME = "rubber_band"
# Simulation parameters
TIMESTEP = 1 / 1000
N_SUBSTEPS = 20
GRAVITY = True
# Rubber band parameters
DIAMETER = 0.0225
DLO_CIRCLE_STEPS = 20
RADIUS = 0.001  # meters
RESOLUTION = 750  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 1e6
YOUNG_MODULUS_TWIST = 1e6
YOUNG_MODULUS_STRETCH = 6e5

GROUND_LENGTH_X = 0.02
GROUND_LENGTH_Y = 0.02
GROUND_HEIGHT = 0.0025

# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

POLE_POSITION_OFFSETS = [[0.0, 0.01], [-0.01, -0.01],[0.01, -0.01]]
POLE_RADIUS = 0.003

# Ground Parameters
EYE = agx.Vec3(0, -0.1, 0.15)
CENTER = agx.Vec3(0, 0, 0.01)
UP = agx.Vec3(0., 0., 0.0)

# Control parameters
MAX_MOTOR_FORCE = 1
GOAL_MAX_Z = 0.0125

GRIPPER_HEIGHT = 0.025
MAX_X = 0.01
MAX_Y = 0.01
GRIPPER_MAX_X = 0.0275
GRIPPER_MAX_Y = 0.0275
GRIPPER_MIN_Z = -0.035
GRIPPER_MAX_Z = 0.007

def create_pole(id, sim, position, material):

    x = position[0]
    y = position[1]

    # Lower part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder_low_" + str(id), shape=agxCollide.Cylinder(POLE_RADIUS, 0.005),
                           position=agx.Vec3(x,y,0.0),
                           rotation=rotation_cylinder,
                           motion_control=agx.RigidBody.KINEMATICS,
                           material=material)
    sim.add(cylinder)

    # Middle part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder_inner_" + str(id), shape=agxCollide.Cylinder(POLE_RADIUS-0.0007, 0.005),
                           position=agx.Vec3(x,y,0.005),
                           rotation=rotation_cylinder,
                           motion_control=agx.RigidBody.KINEMATICS,
                           material=material)
    sim.add(cylinder)

    # Upper part
    rotation_cylinder = agx.OrthoMatrix3x3()
    rotation_cylinder.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    cylinder = create_body(name="cylinder_top_" + str(id), shape=agxCollide.Cylinder(POLE_RADIUS, 0.005),
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
        elif rb.getName() == "cylinder_top_0" or rb.getName() == "cylinder_top_1" or rb.getName() == "cylinder_top_2":
            agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
        elif rb.getName() == "cylinder_inner_0" or rb.getName() == "cylinder_inner_1" or rb.getName() == "cylinder_inner_2":
            agxOSG.setDiffuseColor(node, agxRender.Color.LightSteelBlue())
        elif rb.getName() == "cylinder_low_0" or rb.getName() == "cylinder_low_1" or rb.getName() == "cylinder_low_2":
            agxOSG.setDiffuseColor(node, agxRender.Color.DarkGray())
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
                         motion_control=agx.RigidBody.STATIC)
    sim.add(ground)

    # Creates poles
    for i, position in enumerate(POLE_POSITION_OFFSETS):
        create_pole(id=i, sim=sim, position=position, material=material_hard)

    # Create gripper
    gripper = create_body(name="gripper",
                          shape=agxCollide.Sphere(0.002),
                          position=agx.Vec3(0.0,0.0, GRIPPER_HEIGHT+DIAMETER/2.0),
                          motion_control=agx.RigidBody.DYNAMICS,
                          material=material_hard)
    gripper.getRigidBody("gripper").getGeometry("gripper").setEnableCollisions(False)

    sim.add(gripper)

    #Create base for pusher motors
    prismatic_base = create_locked_prismatic_base("gripper", gripper.getRigidBody("gripper"),
                                                  position_ranges=[(-GRIPPER_MAX_X, GRIPPER_MAX_X),
                                                                   (-GRIPPER_MAX_Y, GRIPPER_MAX_Y),
                                                                   (GRIPPER_MIN_Z, GRIPPER_MAX_Z)],
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
    for a in np.linspace(-np.pi/2, (3.0/2.0)*np.pi - 2*np.pi/steps, steps):
        x = np.cos(a)*DIAMETER/2.0
        z = np.sin(a) *DIAMETER/2.0
        rubber_band.add(agxCable.FreeNode(x,0,GRIPPER_HEIGHT+z))

    sim.add(rubber_band)

    segments_cable = list()
    cable = agxCable.Cable.find(sim, "DLO")
    segment_iterator = cable.begin()
    n_segments = cable.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()

            seg.setAngularVelocityDamping(1e4)

            mass_props = seg.getMassProperties()
            mass_props.setMass(1.25*mass_props.getMass())

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
    f0.setLocalTranslate(0.0, 0.0, -1*np.pi*DIAMETER/cable.getNumSegments())
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


def set_center_obstacle(sim, center_pos):

    ground = sim.getRigidBody("ground")
    ground.setPosition(agx.Vec3(center_pos[0], center_pos[1], -0.005))

    for i in range(3):
        sim.getRigidBody("cylinder_low_" + str(i)).setPosition(agx.Vec3(center_pos[0]+ POLE_POSITION_OFFSETS[i][0],
                                                                        center_pos[1]+ POLE_POSITION_OFFSETS[i][1],
                                                                        0.0))

        sim.getRigidBody("cylinder_inner_" + str(i)).setPosition(agx.Vec3(center_pos[0]+ POLE_POSITION_OFFSETS[i][0],
                                                                        center_pos[1]+ POLE_POSITION_OFFSETS[i][1],
                                                                        0.005))

        sim.getRigidBody("cylinder_top_" + str(i)).setPosition(agx.Vec3(center_pos[0]+ POLE_POSITION_OFFSETS[i][0],
                                                                        center_pos[1]+ POLE_POSITION_OFFSETS[i][1],
                                                                        0.01))


def get_poles_enclosed(segments_pos, pole_pos):
    poles_enclosed = np.zeros(3)
    segments_xy = np.array(segments_pos)[:,0:2]
    for i in range(0,3):
        is_within_polygon = point_inside_polygon(segments_xy, pole_pos[i])
        poles_enclosed[i] = int(is_within_polygon)

    return poles_enclosed


def compute_segments_pos(sim):
    segments_pos = []
    dlo = agxCable.Cable.find(sim, "DLO")
    segment_iterator = dlo.begin()
    n_segments = dlo.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            pos = segment_iterator.getGeometry().getPosition()
            segments_pos.append(to_numpy_array(pos))
            segment_iterator.inc()

    return segments_pos


def is_goal_reached(center_pos, segment_pos):
    pole_pos = center_pos + POLE_POSITION_OFFSETS
    n_enclosed = get_poles_enclosed(segment_pos, pole_pos)
    if np.sum(n_enclosed) >= 3 and all_points_below_z(segment_pos, max_z=GOAL_MAX_Z):
        return True
    return False


def compute_dense_reward_and_check_goal(center_pos, segment_pos_0, segment_pos_1):

    pole_pos = center_pos + POLE_POSITION_OFFSETS

    poles_enclosed_0 = get_poles_enclosed(segment_pos_0, pole_pos)
    poles_enclosed_1 = get_poles_enclosed(segment_pos_1, pole_pos)
    poles_enclosed_diff = poles_enclosed_0 - poles_enclosed_1

    # Check if final goal is reached
    is_correct_height = all_points_below_z(segment_pos_0, max_z=GOAL_MAX_Z)
    n_enclosed_0 = np.sum(poles_enclosed_0)
    final_goal_reached = n_enclosed_0 >= 3 and is_correct_height

    return np.sum(poles_enclosed_diff) + 5*float(final_goal_reached), final_goal_reached


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

    segment_pos_old = compute_segments_pos(sim)
    reward_type = "dense"

    # Set random obstacle position
    center_pos = np.random.uniform([-MAX_X, -MAX_Y], [MAX_X, MAX_Y])
    center_pos = np.array([-MAX_X, -MAX_Y])
    set_center_obstacle(sim, center_pos)

    for _ in range(10000):
        sim.stepForward()
        app.executeOneStepWithGraphics()

        # Get segments positions
        segment_pos = compute_segments_pos(sim)

        # Compute reward
        if reward_type == "dense":
            reward, goal_reached = compute_dense_reward_and_check_goal(center_pos, segment_pos, segment_pos_old)
        else:
            goal_reached = is_goal_reached(center_pos, segment_pos)
            reward = float(goal_reached)

        segment_pos_old = segment_pos

        if reward !=0:
            print("reward: ", reward)

        if goal_reached:
            print("Success!")
            break


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
