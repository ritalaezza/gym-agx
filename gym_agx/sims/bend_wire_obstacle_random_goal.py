"""
This module adds a randomly generated goal configuration to the BendWireObstacle environment.
"""
import agx
import agxCollide
import agxCable

# Python modules
import math
import numpy as np
from numpy import random

# Local modules
from gym_agx.utils.agx_utils import create_body
from gym_agx.utils.agx_utils import create_locked_prismatic_base, to_numpy_array
from gym_agx.utils.utils import point_to_point_trajectory

# Simulation Parameters
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
GROUND_WIDTH = 0.0001  # meters
POISSON_RATIO = 0.35  # no unit
YOUNG_MODULUS = 69e9  # Pascals (1e9)
YIELD_POINT = 5e7  # Pascals (1e7)
CONTACT_YOUNG_MODULUS = 67e12  # Pascals

# Rendering Parameters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS


def add_goal(sim, logger):

    # Get current delta-t (timestep) that is used in the simulation?
    dt = sim.getTimeStep()
    logger.debug("default dt = {}".format(dt))

    # Change the timestep
    sim.setTimeStep(TIMESTEP)

    # Confirm timestep changed
    dt = sim.getTimeStep()
    logger.debug("new dt = {}".format(dt))

    # Create two grippers one static one kinematic
    gripper_left = create_body(name="gripper_left_goal",
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=agx.Vec3(-LENGTH / 2, 0, 0),
                               motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_left)

    gripper_right = create_body(name="gripper_right_goal",
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=agx.Vec3(LENGTH / 2, 0, 0),
                                motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_right)

    gripper_left_body = gripper_left.getRigidBody("gripper_left_goal")
    gripper_right_body = gripper_right.getRigidBody("gripper_right_goal")

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

    cylinder = create_body(name="obstacle_goal",
                           shape=agxCollide.Cylinder(CYLINDER_RADIUS, CYLINDER_LENGTH),
                           position=agx.Vec3(0, 0, -2 * CYLINDER_RADIUS), motion_control=agx.RigidBody.STATIC,
                           material=material_cylinder)
    sim.add(cylinder)

    # Set cable name and properties
    cable.setName("DLO_goal")
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
        logger.debug("Successful cable initialization.")
    else:
        logger.error(report.getActualError())

    # Add cable to simulation
    sim.add(cable)

    # Add segment names and get first and last segment
    segment_count = 0
    iterator = cable.begin()
    segment_left = iterator.getRigidBody()
    segment_left.setName('dlo_' + str(segment_count+1) + '_goal')
    segment_right = None

    while not iterator.isEnd():
        segment_count += 1
        segment_right = iterator.getRigidBody()
        segment_right.setName('dlo_' + str(segment_count+1) + '_goal')
        iterator.inc()

    # Add hinge constraints
    hinge_joint_left = agx.Hinge(sim.getRigidBody("gripper_left_goal"), frame_left, segment_left)
    hinge_joint_left.setName('hinge_joint_left_goal')
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

    hinge_joint_right = agx.Hinge(sim.getRigidBody("gripper_right_goal"), frame_right, segment_right)
    hinge_joint_right.setName('hinge_joint_right_goal')
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
    prismatic_base_left = create_locked_prismatic_base("gripper_left_goal", gripper_left_body, compliance=0,
                                                       position_ranges=[(-LENGTH / 2 + CYLINDER_RADIUS,
                                                                         LENGTH / 2 - CYLINDER_RADIUS),
                                                                        (-CYLINDER_LENGTH / 3, CYLINDER_LENGTH / 3),
                                                                        (-(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH),
                                                                         0)],
                                                       lock_status=[False, False, False])
    sim.add(prismatic_base_left)
    prismatic_base_right = create_locked_prismatic_base("gripper_right_goal", gripper_right_body, compliance=0,
                                                        position_ranges=[(-LENGTH / 2 + CYLINDER_RADIUS,
                                                                          LENGTH / 2 - CYLINDER_RADIUS),
                                                                         (-CYLINDER_LENGTH / 3, CYLINDER_LENGTH / 3),
                                                                         (-(GROUND_WIDTH + SIZE_GRIPPER / 2 + LENGTH),
                                                                          0)],
                                                        lock_status=[False, False, False])
    sim.add(prismatic_base_right)

    right_motor_x = sim.getConstraint1DOF("gripper_right_goal_joint_base_x").getMotor1D()
    right_motor_y = sim.getConstraint1DOF("gripper_right_goal_joint_base_y").getMotor1D()
    right_motor_z = sim.getConstraint1DOF("gripper_right_goal_joint_base_z").getMotor1D()
    left_motor_x = sim.getConstraint1DOF("gripper_left_goal_joint_base_x").getMotor1D()
    left_motor_y = sim.getConstraint1DOF("gripper_left_goal_joint_base_y").getMotor1D()
    left_motor_z = sim.getConstraint1DOF("gripper_left_goal_joint_base_z").getMotor1D()
    right_gripper = sim.getRigidBody("gripper_right_goal")
    left_gripper = sim.getRigidBody("gripper_left_goal")

    x_right = LENGTH / 2
    y_right = 0
    z_right = 0

    x_left = -LENGTH / 2
    y_left = 0
    z_left = 0

    coords_allowed = False
    counter = 1

    while not coords_allowed:

        # Randomize the goal end position in a safe space
        x_right = np.random.uniform(CYLINDER_RADIUS, LENGTH / 2)
        y_right = np.random.uniform(-CYLINDER_LENGTH, CYLINDER_LENGTH)
        z_right = np.random.uniform(-3 * CYLINDER_RADIUS, -10 * CYLINDER_RADIUS)

        x_left = np.random.uniform(-CYLINDER_RADIUS, -LENGTH / 2)
        y_left = np.random.uniform(-CYLINDER_LENGTH, CYLINDER_LENGTH)
        z_left = np.random.uniform(-3 * CYLINDER_RADIUS, -10 * CYLINDER_RADIUS)

        # print(f"LEFT:\t[{x_left}, {y_left}, {z_left}]")
        # print(f"RIGHT:\t[{x_right}, {y_right}, {z_right}]")

        # To comply with the length constraint, we approximate the minimum wire length by assuming a cuboid obstacle
        # Start by calculating the relevant "per-coordinate" distances
        x_dist = x_right - x_left
        y_dist = y_right - y_left

        # which ratio of the entire x distance lies to the left of / on / to the right of the obstacle?
        x_ratio_left = (-CYLINDER_RADIUS - x_left) / x_dist
        x_ratio_obstacle = (2 * CYLINDER_RADIUS) / x_dist
        # x_ratio_right = (x_right - CYLINDER_RADIUS) / x_dist  # only used for verification

        # use these ratios to determine the respective y distance crossed (assuming a straight x-y-connection)
        y_dist_left = x_ratio_left * y_dist
        y_dist_obstacle = x_ratio_obstacle * y_dist
        # y_dist_right = x_ratio_right * y_dist  # only used for verification

        # this gives us the coordinates of the cable on both top edges of the imaginary cuboid
        # Note: x and z are predetermined (+/- CYLINDER_RADIUS and -CYLINDER_RADIUS respectively)
        y_cuboid_left = y_left + y_dist_left
        y_cuboid_right = y_cuboid_left + y_dist_obstacle

        left_gripper_coords = np.array([x_left, y_left, z_left])
        left_cuboid_coords = np.array([-CYLINDER_RADIUS, y_cuboid_left, -CYLINDER_RADIUS])

        right_cuboid_coords = np.array([CYLINDER_RADIUS, y_cuboid_right, -CYLINDER_RADIUS])
        right_gripper_coords = np.array([x_right, y_right, z_right])

        # Minimal length approximation:
        length_left = np.linalg.norm(left_gripper_coords - left_cuboid_coords)
        length_obst = np.linalg.norm(right_cuboid_coords - left_cuboid_coords)
        length_right = np.linalg.norm(right_gripper_coords - right_cuboid_coords)

        length_approx = length_left + length_obst + length_right

        if length_approx < LENGTH:
            logger.info(f"Found valid coordinate combination! Number of tries: {counter}\n"
                        f"Approximated minimal length is ~{round(length_approx/LENGTH*100)}% of the wire length.")
            coords_allowed = True
        else:
            counter += 1

    goal_end_position_right = np.array([x_right, y_right, z_right])
    goal_end_position_left = np.array([x_left, y_left, z_left])

    time_steps = []
    start_time = 0
    n_seconds = 20
    n_steps = int(n_seconds / (TIMESTEP * N_SUBSTEPS))
    positions_right = []
    positions_left = []
    for k in range(n_steps):
        current_time = k * (TIMESTEP * N_SUBSTEPS)
        velocity_right = point_to_point_trajectory(current_time, start_time, n_seconds, np.array([LENGTH / 2, 0, 0]),
                                                   goal_end_position_right, degree=5)
        velocity_left = point_to_point_trajectory(current_time, start_time, n_seconds, np.array([-LENGTH / 2, 0, 0]),
                                                  goal_end_position_left, degree=5)
        time_steps.append(current_time)

        right_motor_x.setSpeed(velocity_right[0])
        left_motor_x.setSpeed(velocity_left[0])
        right_motor_y.setSpeed(velocity_right[1])
        left_motor_y.setSpeed(velocity_left[1])
        right_motor_z.setSpeed(velocity_right[2])
        left_motor_z.setSpeed(velocity_left[2])

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()
        current_position_right = to_numpy_array(right_gripper.getPosition())
        current_position_left = to_numpy_array(left_gripper.getPosition())
        positions_right = np.append(positions_right, current_position_right, axis=0)
        positions_left = np.append(positions_left, current_position_left, axis=0)

    # disable collisions and set to static for all goal-related bodies
    rbs = sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        if "_goal" in name:
            rb.setMotionControl(agx.RigidBody.STATIC)
            rb_geometries = rb.getGeometries()
            rb_geometries[0].setEnableCollisions(False)

    return cable.getCurrentLength(), segment_count
