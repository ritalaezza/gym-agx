"""Simulation of peg in hole

This module creates the simulation files which will be used in PegInHole environments.
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
import agxUtil
import agxRender

# Python modules
import os
import sys
import numpy as np

# Local modules
from gym_agx.utils.agx_utils import create_body, save_simulation, to_numpy_array
from gym_agx.utils.agx_classes import KeyboardMotorHandler


FILE_NAME = "peg_in_hole"
# Simulation parameters
TIMESTEP = 1 / 500
N_SUBSTEPS = 20
GRAVITY = True
# Rope parameters
RADIUS = 0.00975  # meters
RESOLUTION = 100  # segments per meter
PEG_POISSON_RATIO = 0.1  # no unit
YOUNG_MODULUS_BEND = 1e5  # 1e4
YOUNG_MODULUS_TWIST = 1e6  # 1e10
YOUNG_MODULUS_STRETCH = 1e8  # Pascals

# Aluminum Parameters
ALUMINUM_POISSON_RATIO = 0.35  # no unit
ALUMINUM_YOUNG_MODULUS = 69e9  # Pascals
ALUMINUM_YIELD_POINT = 5e7  # Pascals

# Meshes and textures
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.split(FILE_DIR)[0]
TEXTURE_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "envs/assets/textures/texture_gripper.png")
MESH_GRIPPER_FILE = os.path.join(PACKAGE_DIR, "envs/assets/meshes/mesh_gripper.obj")
MESH_HOLLOW_CYLINDER_FILE = os.path.join(PACKAGE_DIR, "envs/assets/meshes/mesh_hollow_cylinder.obj")

# Ground Parameters
EYE = agx.Vec3(0, -1, 0.2)
CENTER = agx.Vec3(0, 0, 0.2)
UP = agx.Vec3(0., 0., 1.0)

# Control parameters
JOINT_RANGES = {"t_x": [-0.1, 0.1],
                "t_y": [-0.05, 0.05],
                "t_z": [-0.15, 0.05],
                "r_y": [(-1 / 4) * np.pi, (1 / 4) * np.pi]}
FORCE_RANGES = {"t_x": [-5, 5], "t_y": [-5, 5], "t_z": [-5, 5], "r_y": [-5, 5]}


def create_gripper_peg_in_hole(sim=None,
                               name="",
                               material=None,
                               position=agx.Vec3(0.0, 0.0, 0.0),
                               geometry_transform=agx.AffineMatrix4x4(),
                               geometry_scaling=agx.Matrix3x3(agx.Vec3(0.045)),
                               joint_ranges=None,
                               force_ranges=None):
    # Create gripper object
    gripper_mesh = agxUtil.createTrimeshFromWavefrontOBJ(MESH_GRIPPER_FILE, agxCollide.Trimesh.NO_WARNINGS,
                                                         geometry_scaling)
    gripper_geom = agxCollide.Geometry(gripper_mesh, agx.AffineMatrix4x4.rotate(agx.Vec3(0, 1, 0), agx.Vec3(0, 0, 1)))
    gripper_body = agx.RigidBody(name + "_body")
    gripper_body.add(gripper_geom)
    gripper_body.setMotionControl(agx.RigidBody.DYNAMICS)
    gripper_body.setPosition(position)
    gripper_body.setCmRotation(agx.EulerAngles(0.0, np.pi, 0.0))
    if material is not None:
        gripper_geom.setMaterial(material)
    sim.add(gripper_body)

    # Add kinematic structure
    rotation_y_to_z = agx.OrthoMatrix3x3()
    rotation_y_to_z.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    rotation_y_to_x = agx.OrthoMatrix3x3()
    rotation_y_to_x.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.X_AXIS())

    base_z = create_body(name=name + "_base_z",
                         shape=agxCollide.Cylinder(0.005, 0.025),
                         position=position,
                         rotation=rotation_y_to_z,
                         motion_control=agx.RigidBody.DYNAMICS,
                         disable_collisions=True)
    sim.add(base_z)

    base_y = create_body(name=name + "_base_y",
                         shape=agxCollide.Cylinder(0.005, 0.025),
                         position=position,
                         motion_control=agx.RigidBody.DYNAMICS,
                         disable_collisions=True)
    sim.add(base_y)

    base_x = create_body(name=name + "_base_x",
                         shape=agxCollide.Cylinder(0.005, 0.025),
                         position=position,
                         rotation=rotation_y_to_x,
                         motion_control=agx.RigidBody.DYNAMICS,
                         disable_collisions=True)
    sim.add(base_x)

    base_x_body = base_x.getRigidBody("gripper_base_x")
    base_y_body = base_y.getRigidBody("gripper_base_y")
    base_z_body = base_z.getRigidBody("gripper_base_z")

    base_x_body.getGeometry("gripper_base_x").setEnableCollisions(False)
    base_y_body.getGeometry("gripper_base_y").setEnableCollisions(False)
    base_z_body.getGeometry("gripper_base_z").setEnableCollisions(False)

    # Add prismatic joints between bases
    joint_base_x = agx.Prismatic(agx.Vec3(1, 0, 0), base_x_body)
    joint_base_x.setEnableComputeForces(True)
    joint_base_x.setEnable(True)
    joint_base_x.setName(name + "_joint_base_x")
    sim.add(joint_base_x)

    joint_base_y = agx.Prismatic(agx.Vec3(0, 1, 0), base_x_body, base_y_body)
    joint_base_y.setEnableComputeForces(True)
    joint_base_y.setEnable(True)
    joint_base_y.setName(name + "_joint_base_y")
    sim.add(joint_base_y)

    joint_base_z = agx.Prismatic(agx.Vec3(0, 0, -1), base_y_body, base_z_body)
    joint_base_z.setEnableComputeForces(True)
    joint_base_z.setEnable(True)
    joint_base_z.setName(name + "_joint_base_z")
    sim.add(joint_base_z)

    # Hinge joint to rotate gripper around y axis
    hf = agx.HingeFrame()
    hf.setCenter(position)
    hf.setAxis(agx.Vec3(0, 1, 0))
    joint_rot_y = agx.Hinge(hf, base_z_body, gripper_body)
    joint_rot_y.setEnableComputeForces(True)
    joint_rot_y.setName(name + "_joint_rot_y")
    sim.add(joint_rot_y)

    # Set joint ranges
    if joint_ranges is not None:
        # x range
        joint_base_x_range_controller = joint_base_x.getRange1D()
        joint_base_x_range_controller.setRange(joint_ranges["t_x"][0], joint_ranges["t_x"][1])
        joint_base_x_range_controller.setEnable(True)

        # y range
        joint_base_y_range_controller = joint_base_y.getRange1D()
        joint_base_y_range_controller.setRange(joint_ranges["t_y"][0], joint_ranges["t_y"][1])
        joint_base_y_range_controller.setEnable(True)

        # z range
        joint_base_z_range_controller = joint_base_z.getRange1D()
        joint_base_z_range_controller.setRange(joint_ranges["t_z"][0], joint_ranges["t_z"][1])
        joint_base_z_range_controller.setEnable(True)

        # rot y
        joint_rot_y_range_controller = joint_rot_y.getRange1D()
        joint_rot_y_range_controller.setRange(joint_ranges["r_y"][0], joint_ranges["r_y"][1])
        joint_rot_y_range_controller.setEnable(True)

    # Enable motors
    joint_base_x_motor = joint_base_x.getMotor1D()
    joint_base_x_motor.setEnable(True)
    joint_base_x_motor.setLockedAtZeroSpeed(False)

    joint_base_y_motor = joint_base_y.getMotor1D()
    joint_base_y_motor.setEnable(True)
    joint_base_y_motor.setLockedAtZeroSpeed(False)

    joint_base_z_motor = joint_base_z.getMotor1D()
    joint_base_z_motor.setEnable(True)
    joint_base_z_motor.setLockedAtZeroSpeed(False)

    joint_rot_y_motor = joint_rot_y.getMotor1D()
    joint_rot_y_motor.setEnable(True)
    joint_rot_y_motor.setLockedAtZeroSpeed(False)

    # Set max forces in motors
    if force_ranges is not None:
        # force x
        joint_base_x_motor.setForceRange(force_ranges['t_x'][0], force_ranges['t_x'][1])

        # force y
        joint_base_y_motor.setForceRange(force_ranges['t_y'][0], force_ranges['t_y'][1])

        # force z
        joint_base_z_motor.setForceRange(force_ranges['t_z'][0], force_ranges['t_z'][1])

        # force rotation around y
        joint_rot_y_motor.setForceRange(force_ranges['r_y'][0], force_ranges['r_y'][1])


def add_rendering(sim):
    app = agxOSG.ExampleApplication(sim)

    # Set renderer
    app.setAutoStepping(True)
    app.setEnableDebugRenderer(False)
    app.setEnableOSGRenderer(True)

    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    gripper_texture = os.path.join(package_directory, TEXTURE_GRIPPER_FILE)

    # Create scene graph for rendering
    root = app.getSceneRoot()
    rbs = sim.getRigidBodies()
    for rb in rbs:
        node = agxOSG.createVisual(rb, root)
        if rb.getName() == "hollow_cylinder":
            agxOSG.setDiffuseColor(node, agxRender.Color_SteelBlue())
            agxOSG.setShininess(node, 15)
        elif rb.getName() == "gripper_body":
            agxOSG.setDiffuseColor(node, agxRender.Color(1.0, 1.0, 1.0, 1.0))
            agxOSG.setTexture(node, gripper_texture, False, agxOSG.DIFFUSE_TEXTURE)
            agxOSG.setShininess(node, 2)
        elif "dlo" in rb.getName():  # Cable segments
            agxOSG.setDiffuseColor(node, agxRender.Color(0.0, 1.0, 0.0, 1.0))
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.0)

    # Set rendering options
    scene_decorator = app.getSceneDecorator()
    scene_decorator.setEnableLogo(False)
    scene_decorator.setBackgroundColor(agxRender.Color(1.0, 1.0, 1.0, 1.0))

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

    # Define materials
    material_hard = agx.Material("Aluminum")
    material_hard_bulk = material_hard.getBulkMaterial()
    material_hard_bulk.setPoissonsRatio(ALUMINUM_POISSON_RATIO)
    material_hard_bulk.setYoungsModulus(ALUMINUM_YOUNG_MODULUS)

    # Create gripper
    create_gripper_peg_in_hole(sim=sim,
                               name="gripper",
                               material=material_hard,
                               position=agx.Vec3(0.0, 0.0, 0.35),
                               geometry_transform=agx.AffineMatrix4x4(),
                               joint_ranges=JOINT_RANGES,
                               force_ranges=FORCE_RANGES
                               )

    # Create hollow cylinder with hole
    scaling_cylinder = agx.Matrix3x3(agx.Vec3(0.0275))
    hullMesh = agxUtil.createTrimeshFromWavefrontOBJ(MESH_HOLLOW_CYLINDER_FILE, 0, scaling_cylinder)
    hullGeom = agxCollide.Geometry(hullMesh, agx.AffineMatrix4x4.rotate(agx.Vec3(0, 1, 0), agx.Vec3(0, 0, 1)))
    hollow_cylinder = agx.RigidBody("hollow_cylinder")
    hollow_cylinder.add(hullGeom)
    hollow_cylinder.setMotionControl(agx.RigidBody.STATIC)
    hullGeom.setMaterial(material_hard)
    sim.add(hollow_cylinder)

    # Create rope and set name + properties
    peg = agxCable.Cable(RADIUS, RESOLUTION)
    peg.setName("DLO")
    material_peg = peg.getMaterial()
    peg_material = material_peg.getBulkMaterial()
    peg_material.setPoissonsRatio(PEG_POISSON_RATIO)
    properties = peg.getCableProperties()
    properties.setYoungsModulus(YOUNG_MODULUS_BEND, agxCable.BEND)
    properties.setYoungsModulus(YOUNG_MODULUS_TWIST, agxCable.TWIST)
    properties.setYoungsModulus(YOUNG_MODULUS_STRETCH, agxCable.STRETCH)

    # Add connection between cable and gripper
    tf_0 = agx.AffineMatrix4x4()
    tf_0.setTranslate(0.0, 0, 0.075)
    peg.add(agxCable.BodyFixedNode(sim.getRigidBody("gripper_body"), tf_0))
    peg.add(agxCable.FreeNode(0.0, 0.0, 0.1))

    sim.add(peg)

    segment_iterator = peg.begin()
    n_segments = peg.getNumSegments()
    for i in range(n_segments):
        if not segment_iterator.isEnd():
            seg = segment_iterator.getRigidBody()
            seg.setAngularVelocityDamping(1e3)
            segment_iterator.inc()

    # Try to initialize rope
    report = peg.tryInitialize()
    if report.successful():
        print("Successful rope initialization.")
    else:
        print(report.getActualError())

    # Add rope to simulation
    sim.add(peg)

    # Set rope material
    material_peg = peg.getMaterial()
    material_peg.setName("rope_material")

    contactMaterial = sim.getMaterialManager().getOrCreateContactMaterial(material_hard, material_peg)
    contactMaterial.setYoungsModulus(1e12)
    fm = agx.IterativeProjectedConeFriction()
    fm.setSolveType(agx.FrictionModel.DIRECT)
    contactMaterial.setFrictionModel(fm)

    # Add keyboard listener
    motor_x = sim.getConstraint1DOF("gripper_joint_base_x").getMotor1D()
    motor_y = sim.getConstraint1DOF("gripper_joint_base_y").getMotor1D()
    motor_z = sim.getConstraint1DOF("gripper_joint_base_z").getMotor1D()
    motor_rot_y = sim.getConstraint1DOF("gripper_joint_rot_y").getMotor1D()
    key_motor_map = {agxSDK.GuiEventListener.KEY_Up: (motor_y, 0.5),
                     agxSDK.GuiEventListener.KEY_Down: (motor_y, -0.5),
                     agxSDK.GuiEventListener.KEY_Right: (motor_x, 0.5),
                     agxSDK.GuiEventListener.KEY_Left: (motor_x, -0.5),
                     65365: (motor_z, 0.5),
                     65366: (motor_z, -0.5),
                     120: (motor_rot_y, 5),
                     121: (motor_rot_y, -5)}
    sim.add(KeyboardMotorHandler(key_motor_map))

    rbs = peg.getRigidBodies()
    for i in range(len(rbs)):
        rbs[i].setName('dlo_' + str(i + 1))

    return sim


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


def is_goal_reached(sim):
    """
    Checks if positions of cable segments on lower end are within goal region. Returns True if cable is partially
    inserted and False otherwise.
    """
    cable = agxCable.Cable.find(sim, "DLO")
    n_segments = cable.getNumSegments()
    segment_iterator = cable.begin()
    cylinder_pos = sim.getRigidBody("hollow_cylinder").getPosition()

    for i in range(0, n_segments):
        if not segment_iterator.isEnd():
            p = segment_iterator.getGeometry().getPosition()
            segment_iterator.inc()

            if i >= n_segments / 2:
                # Return False if segment is outside bounds
                if not (cylinder_pos[0] - 0.015 <= p[0] <= cylinder_pos[0] + 0.015 and
                        cylinder_pos[1] - 0.015 <= p[1] <= cylinder_pos[1] + 0.015 and
                        -0.1 <= p[2] <= 0.07):
                    return False

    return True


def determine_n_segments_inserted(segment_pos, cylinder_pos):
    """
    Determine number of segments that are inserted into the hole.

    :param segment_pos:
    :return:
    """

    n_inserted = 0
    for p in segment_pos:
        # Return False if segment is outside bounds
        if cylinder_pos[0] - 0.015 <= p[0] <= cylinder_pos[0] + 0.015 and \
                cylinder_pos[1] - 0.015 <= p[1] <= cylinder_pos[1] + 0.015 and \
                -0.1 <= p[2] <= 0.07:
            n_inserted += 1
    return n_inserted


def compute_dense_reward_and_check_goal(sim, segment_pos_0, segment_pos_1):
    cylinder_pos = sim.getRigidBody("hollow_cylinder").getPosition()
    n_segs_inserted_0 = determine_n_segments_inserted(segment_pos_0, cylinder_pos)
    n_segs_inserted_1 = determine_n_segments_inserted(segment_pos_1, cylinder_pos)
    n_segs_inserted_diff = n_segs_inserted_0 - n_segs_inserted_1

    cable = agxCable.Cable.find(sim, "DLO")
    n_segments = cable.getNumSegments()

    # Check if final goal is reached
    final_goal_reached = n_segs_inserted_0 >= n_segments / 2

    return np.sum(n_segs_inserted_diff) + 5 * float(final_goal_reached), final_goal_reached


def main(args):
    # Build simulation object
    sim = build_simulation()

    # Save simulation to file
    success = save_simulation(sim, FILE_NAME)
    if success:
        print("Simulation saved!")
    else:
        print("Simulation not saved!")

    # Add app
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable, '--window', '400', '600'] + args))
    app.setTimeStep(TIMESTEP)
    app.setCameraHome(EYE, CENTER, UP)
    app.initSimulation(sim, True)

    cylinder_pos_x = np.random.uniform(-0.1, 0.1)
    cylinder_pos_y = np.random.uniform(0.05, 0.05)

    cylinder = sim.getRigidBody("hollow_cylinder")
    cylinder.setPosition(agx.Vec3(cylinder_pos_x, cylinder_pos_y, 0.0))

    segment_pos_old = compute_segments_pos(sim)
    reward_type = "dense"

    for _ in range(10000):
        sim.stepForward()
        app.executeOneStepWithGraphics()

        # Get segments positions
        segment_pos = compute_segments_pos(sim)

        # Compute reward
        if reward_type == "sparse":
            reward, goal_reached = compute_dense_reward_and_check_goal(sim, segment_pos, segment_pos_old)
        else:
            goal_reached = is_goal_reached(sim)
            reward = float(goal_reached)

        segment_pos_old = segment_pos

        if reward != 0:
            print("reward: ", reward)

        if goal_reached:
            print("Success!")
            break


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
