"""Simulation of bend wire environment

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
import numpy as np
import math
import sys
import os

# Local modules
from gym_agx.utils.agx_utils import create_body, save_simulation, to_numpy_array, create_locked_prismatic_base, \
    add_goal_assembly_from_file
from gym_agx.utils.utils import polynomial_trajectory, sample_sphere

PURELY_ELASTIC = False

suffix = ""
if PURELY_ELASTIC:
    suffix = "_elastic"
FILE_NAME = 'bend_wire' + suffix
# Simulation Parameters
N_SUBSTEPS = 2
TIMESTEP = 1 / 100  # seconds (eq. 100 Hz)
LENGTH = 0.1  # meters
RADIUS = LENGTH / 100  # meters
LENGTH += 2 * RADIUS  # meters
RESOLUTION = 1000  # segments per meter
GRAVITY = True

# Aluminum Parameters
MATERIAL_STRING = "Aluminum"
POISSON_RATIO = 0.35  # no unit
YOUNG_MODULUS = 69e9  # Pascals
YIELD_POINT = 5e7  # Pascals

if PURELY_ELASTIC:
    # Rubber Band Parameters
    MATERIAL_STRING = "Rubber"
    POISSON_RATIO = 0.4999  # no unit
    YOUNG_MODULUS = 0.1e9  # Pascals

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
        name = rb.getName()
        node = agxOSG.createVisual(rb, root)
        if name == "ground":
            agxOSG.setDiffuseColor(node, agxRender.Color.Gray())
        elif "gripper_left" in name and "base" not in name:
            agxOSG.setDiffuseColor(node, agxRender.Color.Red())
        elif "gripper_right" in name and "base" not in name:
            agxOSG.setDiffuseColor(node, agxRender.Color.Blue())
        elif "dlo" in name:
            agxOSG.setDiffuseColor(node, agxRender.Color.Green())
        else:
            agxOSG.setDiffuseColor(node, agxRender.Color.Beige())
            agxOSG.setAlpha(node, 0.5)
        if "goal" in name:
            agxOSG.setAlpha(node, 0.2)

    scene_decorator = app.getSceneDecorator()
    light_source_0 = scene_decorator.getLightSource(agxOSG.SceneDecorator.LIGHT0)
    light_source_0.setPosition(light_pos)
    light_source_0.setDirection(light_dir)
    scene_decorator.setEnableLogo(False)

    return app


def sample_random_goal(sim, app=None, dof_vector=np.ones(3)):
    """Goal Randomization: Sample 2 points, and execute point-to-point trajectory.

    :param sim: AGX Dynamics simulation object
    :param app: AGX Dynamics application object
    :param np.array dof_vector: desired degrees of freedom of the gripper(s), [x, y, z]
    """
    assert np.sum(dof_vector) >= 1, "There must be at least one degree of freedom"
    right_motor_x = sim.getConstraint1DOF("gripper_right_goal_joint_base_x").getMotor1D()
    right_motor_y = sim.getConstraint1DOF("gripper_right_goal_joint_base_y").getMotor1D()
    right_motor_z = sim.getConstraint1DOF("gripper_right_goal_joint_base_z").getMotor1D()
    right_gripper = sim.getRigidBody("gripper_right_goal")

    settling_time = 1
    n_waypoints = 2
    n_seconds = 10 * n_waypoints + settling_time
    center = np.array([2 * (2 * RADIUS + SIZE_GRIPPER), 0, 0])
    waypoints = [to_numpy_array(right_gripper.getPosition())]
    scales = np.zeros(n_waypoints)
    for i in range(0, n_waypoints):
        waypoint, length = sample_sphere(center,
                                         [2 * (2 * RADIUS + SIZE_GRIPPER) + 4 * SIZE_GRIPPER,
                                          LENGTH - 4 * SIZE_GRIPPER],
                                         [0, np.pi],
                                         [0, np.pi],
                                         [-math.pi / 2, 0, 0])
        waypoints.append(waypoint)
        scales[i] = length

    norm_scales = scales / sum(scales)
    time_scales = norm_scales * n_seconds

    t = sim.getTimeStamp()
    start_time = t
    while t < n_seconds:
        if app:
            app.executeOneStepWithGraphics()
        velocity = polynomial_trajectory(t, start_time, waypoints, time_scales, degree=3)
        velocity = velocity * dof_vector
        right_motor_x.setSpeed(velocity[0])
        right_motor_y.setSpeed(velocity[1])
        right_motor_z.setSpeed(velocity[2])

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # reset timestamp, after simulation
    sim.setTimeStamp(0)


def sample_fixed_goal(sim, app=None):
    """Define the trajectory to generate fixed goal.

    :param sim: AGX Dynamics simulation object
    :param app: AGX Dynamics application object
    """
    right_motor_x = sim.getConstraint1DOF("gripper_right_goal_joint_base_x").getMotor1D()
    right_gripper = sim.getRigidBody("gripper_right_goal")

    middle_point = np.array([2 * (2 * RADIUS + SIZE_GRIPPER) + LENGTH / 2, 0, 0])
    quarter_point = np.array([2 * (2 * RADIUS + SIZE_GRIPPER) + LENGTH / 4, 0, 0])
    waypoints = [to_numpy_array(right_gripper.getPosition()), quarter_point, middle_point]
    time_scales = np.array([15, 5])

    n_seconds = np.sum(time_scales)
    n_time_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    t = sim.getTimeStamp()
    start_time = t
    velocities = np.zeros([n_time_steps, 3])
    for i in range(n_time_steps):
        if app:
            app.executeOneStepWithGraphics()
        velocity = polynomial_trajectory(t, start_time, waypoints, time_scales, degree=5)
        velocities[i, :] = velocity
        right_motor_x.setSpeed(velocity[0])

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()

    # reset timestamp, after simulation
    sim.setTimeStamp(0)
    print(right_gripper.getPosition())

    # Trajectory limits:
    max_velocity = np.max(np.max(abs(velocities)))
    accelerations = np.gradient(velocities, (TIMESTEP * N_SUBSTEPS), axis=0)
    max_acceleration = np.max(np.max(abs(accelerations)))
    print('Max velocity: ' + str(max_velocity))
    print('Max acceleration: ' + str(max_acceleration))

    # Save trajectory
    np.save('bend_wire_traj.npy', velocities)


def build_simulation(goal=False):
    """Builds simulations for both start and goal configurations.

    :param bool goal: toggles between simulation definition of start and goal configurations
    :return agxSDK.Simulation: simulation object
    """
    assembly_name = "start_"
    goal_string = ""
    if goal:
        assembly_name = "goal_"
        goal_string = "_goal"

    # Instantiate a simulation
    sim = agxSDK.Simulation()

    # By default, the gravity vector is 0,0,-9.81 with a uniform gravity field. (we CAN change that
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

    # Create a new empty Assembly
    scene = agxSDK.Assembly()
    scene.setName(assembly_name + "assembly")

    # Add start assembly to simulation
    sim.add(scene)

    # Create a ground plane for reference
    if not goal:
        ground = create_body(name="ground", shape=agxCollide.Box(LENGTH, LENGTH, GROUND_WIDTH),
                             position=agx.Vec3(LENGTH / 2, 0, -(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH)),
                             motion_control=agx.RigidBody.STATIC)
        scene.add(ground)

    # Create cable
    cable = agxCable.Cable(RADIUS, RESOLUTION)
    cable.setName("DLO" + goal_string)

    gripper_left = create_body(name="gripper_left" + goal_string,
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=agx.Vec3(0, 0, 0), motion_control=agx.RigidBody.DYNAMICS)
    scene.add(gripper_left)

    gripper_right = create_body(name="gripper_right" + goal_string,
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=agx.Vec3(LENGTH, 0, 0),
                                motion_control=agx.RigidBody.DYNAMICS)
    scene.add(gripper_right)

    # Disable collisions for grippers
    gripper_left_body = scene.getRigidBody("gripper_left" + goal_string)
    gripper_left_body.getGeometry("gripper_left" + goal_string).setEnableCollisions(False)
    gripper_right_body = scene.getRigidBody("gripper_right" + goal_string)
    gripper_right_body.getGeometry("gripper_right" + goal_string).setEnableCollisions(False)

    print("Mass of grippers: {}".format(scene.getRigidBody("gripper_right" + goal_string).calculateMass()))

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
        print("Successful cable initialization.")
    else:
        print(report.getActualError())

    actual_length = report.getLength()
    print("Actual length: " + str(actual_length))

    if not PURELY_ELASTIC:
        # Add cable plasticity
        plasticity = agxCable.CablePlasticity()
        plasticity.setYieldPoint(YIELD_POINT, agxCable.BEND)  # set torque required for permanent deformation
        cable.addComponent(plasticity)  # NOTE: Stretch direction is always elastic

    # Define material
    material = agx.Material(MATERIAL_STRING)
    bulk_material = material.getBulkMaterial()
    bulk_material.setPoissonsRatio(POISSON_RATIO)
    bulk_material.setYoungsModulus(YOUNG_MODULUS)
    cable.setMaterial(material)

    # Add cable to simulation
    scene.add(cable)

    # Add segment names and get first and last segment
    count = 1
    iterator = cable.begin()
    segment_left = iterator.getRigidBody()
    segment_left.setName('dlo_' + str(count) + goal_string)
    while not iterator.isEnd():
        count += 1
        segment_right = iterator.getRigidBody()
        segment_right.setName('dlo_' + str(count) + goal_string)
        iterator.inc()

    # Add hinge constraints
    hinge_joint_left = agx.Hinge(scene.getRigidBody("gripper_left" + goal_string), frame_left, segment_left)
    hinge_joint_left.setName('hinge_joint_left' + goal_string)
    motor_left = hinge_joint_left.getMotor1D()
    motor_left.setEnable(True)
    motor_left_param = motor_left.getRegularizationParameters()
    motor_left_param.setCompliance(1e12)
    motor_left.setLockedAtZeroSpeed(False)
    lock_left = hinge_joint_left.getLock1D()
    lock_left.setEnable(False)
    # Set range of hinge joint
    # range_left = hinge_joint_left.getRange1D()
    # range_left.setEnable(True)
    # range_left.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    scene.add(hinge_joint_left)

    hinge_joint_right = agx.Hinge(scene.getRigidBody("gripper_right" + goal_string), frame_right, segment_right)
    hinge_joint_right.setName('hinge_joint_right' + goal_string)
    motor_right = hinge_joint_right.getMotor1D()
    motor_right.setEnable(True)
    motor_right_param = motor_right.getRegularizationParameters()
    motor_right_param.setCompliance(1e12)
    motor_right.setLockedAtZeroSpeed(False)
    lock_right = hinge_joint_right.getLock1D()
    lock_right.setEnable(False)
    # Set range of hinge joint
    # range_right = hinge_joint_right.getRange1D()
    # range_right.setEnable(True)
    # range_right.setRange(agx.RangeReal(-math.pi / 2, math.pi / 2))
    scene.add(hinge_joint_right)

    # Create base for gripper motors
    prismatic_base_right = create_locked_prismatic_base("gripper_right" + goal_string, gripper_right_body, compliance=0,
                                                        motor_ranges=[(-FORCE_RANGE, FORCE_RANGE),
                                                                      (-FORCE_RANGE, FORCE_RANGE),
                                                                      (-FORCE_RANGE, FORCE_RANGE)],
                                                        position_ranges=[
                                                            (-LENGTH + 2 * (2 * RADIUS + SIZE_GRIPPER),
                                                             0),
                                                            (-LENGTH, LENGTH),
                                                            (-LENGTH, LENGTH)],
                                                        compute_forces=True,
                                                        lock_status=[False, False, False])
    scene.add(prismatic_base_right)
    prismatic_base_left = create_locked_prismatic_base("gripper_left" + goal_string, gripper_left_body, compliance=0,
                                                       lock_status=[True, True, True])
    scene.add(prismatic_base_left)

    return sim


# Build and save scene to file
def main(args):
    # 1) Build start simulation object
    sim = build_simulation()

    # Save start simulation to file
    success = save_simulation(sim, FILE_NAME)
    if not success:
        print("Simulation not saved!")

    # 2) Build goal simulation object
    goal_sim = build_simulation(goal=True)

    # Save simulation to file
    success = save_simulation(goal_sim, FILE_NAME + "_goal_random")
    if not success:
        print("Goal simulation not saved!")

    # Render simulation
    app = add_rendering(goal_sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(goal_sim, True)  # This changes timestep and Gravity!
    goal_sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        print("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        goal_sim.setUniformGravity(g)

    # 3) Sample fixed goal
    sample_fixed_goal(goal_sim, app)

    # Set goal objects to static
    rbs = goal_sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        if "_goal" in name:
            rb.setMotionControl(agx.RigidBody.STATIC)

    # Save fixed goal simulation to file
    success = save_simulation(goal_sim, FILE_NAME + "_goal")
    if not success:
        print("Fixed goal simulation not saved!")

    # 4) Test random goal generation
    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    random_goal_file = os.path.join(package_directory, 'envs/assets', FILE_NAME + "_goal_random.agx")
    add_goal_assembly_from_file(sim, random_goal_file)

    # Render simulation
    app = add_rendering(sim)
    app.init(agxIO.ArgumentParser([sys.executable] + args))
    app.setCameraHome(EYE, CENTER, UP)  # should only be added after app.init
    app.initSimulation(sim, True)  # This changes timestep and Gravity!
    sim.setTimeStep(TIMESTEP)
    if not GRAVITY:
        print("Gravity off.")
        g = agx.Vec3(0, 0, 0)  # remove gravity
        sim.setUniformGravity(g)

    # Test random goal generation
    sample_random_goal(sim, app)


if __name__ == '__main__':
    if agxPython.getContext() is None:
        init = agx.AutoInit()
        main(sys.argv)
