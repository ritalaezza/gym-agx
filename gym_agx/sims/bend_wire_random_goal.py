"""
This module adds a randomly generated goal configuration to the BendWire environment.
"""
import agx
import agxCollide
import agxCable

# Python modules
import numpy as np
from numpy import random

# Local modules
from gym_agx.utils.agx_utils import create_body
from gym_agx.utils.agx_utils import create_locked_prismatic_base, to_numpy_array
from gym_agx.utils.utils import point_to_point_trajectory

# Simulation Parameters
N_SUBSTEPS = 2
TIMESTEP = 1 / 100  # seconds (eq. 100 Hz)
LENGTH = 0.1  # meters
RADIUS = LENGTH / 100  # meters
LENGTH += 2 * RADIUS  # meters
RESOLUTION = 1000  # segments per meter
POISSON_RATIO = 0.35  # no unit
YOUNG_MODULUS = 69e9  # Pascals
YIELD_POINT = 5e7  # Pascals
# Rendering Parameters
CABLE_GRIPPER_RATIO = 2
SIZE_GRIPPER = CABLE_GRIPPER_RATIO * RADIUS
# Control parameters
FORCE_RANGE = 5  # N


def add_goal(sim, logger):

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
    cable.setName("DLO_goal")

    gripper_left = create_body(name="gripper_left_goal",
                               shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                               position=agx.Vec3(0, 0, 0), motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_left)

    gripper_right = create_body(name="gripper_right_goal",
                                shape=agxCollide.Box(SIZE_GRIPPER, SIZE_GRIPPER, SIZE_GRIPPER),
                                position=agx.Vec3(LENGTH, 0, 0),
                                motion_control=agx.RigidBody.DYNAMICS)
    sim.add(gripper_right)

    # Disable collisions for grippers
    gripper_left_body = sim.getRigidBody("gripper_left_goal")
    gripper_left_body.getGeometry("gripper_left_goal").setEnableCollisions(False)
    gripper_right_body = sim.getRigidBody("gripper_right_goal")
    gripper_right_body.getGeometry("gripper_right_goal").setEnableCollisions(False)

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
    motor_left.setEnable(True)
    motor_left_param = motor_left.getRegularizationParameters()
    motor_left_param.setCompliance(1e12)
    motor_left.setLockedAtZeroSpeed(False)
    lock_left = hinge_joint_left.getLock1D()
    lock_left.setEnable(False)
    sim.add(hinge_joint_left)

    hinge_joint_right = agx.Hinge(sim.getRigidBody("gripper_right_goal"), frame_right, segment_right)
    hinge_joint_right.setName('hinge_joint_right_goal')
    motor_right = hinge_joint_right.getMotor1D()
    motor_right.setEnable(True)
    motor_right_param = motor_right.getRegularizationParameters()
    motor_right_param.setCompliance(1e12)
    motor_right.setLockedAtZeroSpeed(False)
    lock_right = hinge_joint_right.getLock1D()
    lock_right.setEnable(False)
    sim.add(hinge_joint_right)

    # Add prismatic constraint
    prismatic_frame_left = agx.PrismaticFrame(agx.Vec3(LENGTH, 0, 0), agx.Vec3.X_AXIS())
    prismatic_joint_left = agx.Prismatic(prismatic_frame_left, sim.getRigidBody("gripper_left_goal"))
    prismatic_joint_left.setName('prismatic_joint_left_goal')
    lock = prismatic_joint_left.getLock1D()
    lock.setEnable(True)
    sim.add(prismatic_joint_left)

    # Create base for gripper motors
    prismatic_base_right = create_locked_prismatic_base("gripper_right_goal", gripper_right_body, compliance=0,
                                                        motor_ranges=[(-FORCE_RANGE, FORCE_RANGE),
                                                                      (-FORCE_RANGE, FORCE_RANGE),
                                                                      (-FORCE_RANGE, FORCE_RANGE)],
                                                        position_ranges=[
                                                            (-LENGTH + 2 * (2 * RADIUS + SIZE_GRIPPER) + SIZE_GRIPPER,
                                                             0),
                                                            (-LENGTH, LENGTH),
                                                            (-LENGTH, LENGTH)],
                                                        compute_forces=True,
                                                        lock_status=[False, False, False])

    sim.add(prismatic_base_right)

    base_x = sim.getRigidBody("gripper_right_goal_base_x")
    constraint_x = sim.getConstraint1DOF("gripper_right_goal_joint_base_x")
    base_y = sim.getRigidBody("gripper_right_goal_base_y")
    constraint_y = sim.getConstraint1DOF("gripper_right_goal_joint_base_y")
    base_z = sim.getRigidBody("gripper_right_goal_base_z")
    constraint_z = sim.getConstraint1DOF("gripper_right_goal_joint_base_z")
    right_motor_x = sim.getConstraint1DOF("gripper_right_goal_joint_base_x").getMotor1D()
    right_motor_y = sim.getConstraint1DOF("gripper_right_goal_joint_base_y").getMotor1D()
    right_motor_z = sim.getConstraint1DOF("gripper_right_goal_joint_base_z").getMotor1D()
    right_gripper = sim.getRigidBody("gripper_right_goal")
    constraint_base = sim.getConstraint("gripper_right_goal_joint_rb")
    n_seconds = 10
    t = 0
    start_time = 0
    time_limit = 6
    start_position = to_numpy_array(right_gripper.getPosition())
    left_gripper_distance = 0.002  # in meters
    start_distance = np.linalg.norm(start_position - left_gripper_distance)

    # Uniform sampling in a cuboid then checking the LENGTH-constraint

    intermediate_found = False
    x_1 = 0
    y_1 = 0
    z_1 = 0
    r_1 = 0
    while not intermediate_found:
        x_1 = random.uniform(LENGTH/20, LENGTH)
        y_1 = random.uniform(-LENGTH, LENGTH)
        z_1 = random.uniform(-LENGTH, LENGTH)

        r_1 = np.linalg.norm([x_1, y_1, z_1])

        if r_1 <= LENGTH:
            intermediate_found = True

    logger.info(f"Intermediate position has normalized coordinates "
                f"[{round(x_1/LENGTH*100)/100}, {round(y_1/LENGTH*100)/100}, {round(z_1/LENGTH*100)/100}] - "
                f"normalized distance to left gripper is: {round(r_1/LENGTH*100)/100}")

    final_found = False
    x_2 = 0
    y_2 = 0
    z_2 = 0
    r_2 = 0
    while not final_found:
        x_2 = random.uniform(LENGTH / 20, LENGTH)
        y_2 = random.uniform(-LENGTH, LENGTH)
        z_2 = random.uniform(-LENGTH, LENGTH)

        r_2 = np.linalg.norm([x_2, y_2, z_2])

        if r_2 <= LENGTH:
            final_found = True

    logger.info(f"Final position has normalized coordinates "
                f"[{round(x_2/LENGTH*100)/100}, {round(y_2/LENGTH*100)/100}, {round(z_2/LENGTH*100)/100}] - "
                f"normalized distance to left gripper is: {round(r_2/LENGTH*100)/100}")

    end_position = [x_1, y_1, z_1]

    velocity = np.zeros_like(start_position)
    velocities = [velocity * 100]
    positions = [start_position * 100]
    distances = [start_distance * 100]
    forces = [velocity * 100]
    forces_x = [velocity * 100]
    forces_y = [velocity * 100]
    forces_z = [velocity * 100]
    torques = [velocity * 100]
    time_steps = [t]

    while t <= n_seconds:
        right_motor_x.setSpeed(velocity[0])
        right_motor_y.setSpeed(velocity[1])
        right_motor_z.setSpeed(velocity[2])

        t = sim.getTimeStamp()
        t_0 = t
        while t < t_0 + TIMESTEP * N_SUBSTEPS:
            sim.stepForward()
            t = sim.getTimeStamp()
        position = to_numpy_array(right_gripper.getPosition())
        distance = np.linalg.norm(position-left_gripper_distance)
        if t > time_limit and start_time == 0:
            start_time = t
            start_position = position
            end_position = [x_2, y_2, z_2]
            time_limit = 4
        velocity = point_to_point_trajectory(t, start_time, time_limit, start_position, end_position, degree=5)
        time_steps.append(t)
        velocities.append(velocity * 100)
        positions.append(position * 100)
        distances.append(distance * 100)
        force = agx.Vec3()
        torque = agx.Vec3()
        constraint_base.getLastForce(right_gripper, force, torque)
        force_x = agx.Vec3()
        torque_x = agx.Vec3()
        constraint_x.getLastForce(base_x, force_x, torque_x)
        force_y = agx.Vec3()
        torque_y = agx.Vec3()
        constraint_y.getLastForce(base_y, force_y, torque_y)
        force_z = agx.Vec3()
        torque_z = agx.Vec3()
        constraint_z.getLastForce(base_z, force_z, torque_z)

        force = to_numpy_array(force)
        forces.append(force)
        force_x = to_numpy_array(force_x)
        forces_x.append(force_x)
        force_y = to_numpy_array(force_y)
        forces_y.append(force_y)
        force_z = to_numpy_array(force_z)
        forces_z.append(force_z)
        torque = to_numpy_array(torque)
        torques.append(torque * 100)

    # disable collisions and set to static for all goal-related bodies
    rbs = sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        if "_goal" in name:
            rb.setMotionControl(agx.RigidBody.STATIC)
            rb_geometries = rb.getGeometries()
            rb_geometries[0].setEnableCollisions(False)

    return cable.getCurrentLength(), segment_count
