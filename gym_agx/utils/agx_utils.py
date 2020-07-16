import agx
import agxIO
import agxSDK
import agxCollide

import os
import math
import logging
import numpy as np

logger = logging.getLogger('gym_agx.utils')


def save_simulation(sim, file_name):
    """Save AGX simulation object to file.
    :param sim: AGX simulation object
    :param file_name: name of the file
    :return: Boolean for success/failure
    """
    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    markup_file = os.path.join(package_directory, 'envs/assets', file_name + ".aagx")
    if not agxIO.writeFile(markup_file, sim):
        print("Unable to save simulation to markup file!")
        return False
    binary_file = os.path.join(package_directory, 'envs/assets', file_name + ".agx")
    if not agxIO.writeFile(binary_file, sim):
        print("Unable to save simulation to binary file!")
        return False
    return True


def save_goal_simulation(sim, file_name, remove_assemblies=[]):
    """Save AGX simulation object to file.
    :param sim: AGX simulation object
    :param file_name: name of the file
    :param remove_assemblies: string list of assemblies to remove
    :return: Boolean for success/failure
    """
    # Remove assemblies
    for assembly in remove_assemblies:
        ass = sim.getAssembly(assembly)
        sim.remove(ass)
    # Make all rigid bodies left static, collision free and add goal to their name
    rbs = sim.getRigidBodies()
    for rb in rbs:
        name = rb.getName()
        rb.setName(name + '_goal')
        rb.setMotionControl(agx.RigidBody.STATIC)
        rb_geometries = rb.getGeometries()
        rb_geometries[0].setEnableCollisions(False)
    file_directory = os.path.dirname(os.path.abspath(__file__))
    package_directory = os.path.split(file_directory)[0]
    markup_file = os.path.join(package_directory, 'envs/assets', file_name + "_goal.aagx")
    if not agxIO.writeFile(markup_file, sim):
        print("Unable to save simulation to markup file!")
        return False
    binary_file = os.path.join(package_directory, 'envs/assets', file_name + "_goal.agx")
    if not agxIO.writeFile(binary_file, sim):
        print("Unable to save simulation to binary file!")
        return False
    return True


def to_numpy_array(agx_list):
    """Convert from AGX data structure to NumPy array.
    :param agx_list: AGX data structure
    :return: NumPy array
    """
    agx_type = type(agx_list)
    if agx_type == agx.Vec3:
        np_array = np.zeros(shape=(3,), dtype=np.float64)
        for i in range(3):
            np_array[i] = agx_list[i]
    elif agx_type == agx.Quat:
        np_array = np.zeros(shape=(4,), dtype=np.float64)
        for i in range(4):
            np_array[i] = agx_list[i]
    elif agx_type == agx.OrthoMatrix3x3:
        np_array = np.zeros(shape=(3, 3), dtype=np.float64)
        for i in range(3):
            row = agx_list.getRow(i)
            for j in range(3):
                np_array[i, j] = row[j]
    else:
        logger.warning('Conversion for {} type is not supported.'.format(agx_type))

    return np_array


def to_agx_list(np_array, agx_type):
    """Convert from Numpy array to AGX data structure.
    :param np_array:  NumPy array
    :param agx_type: Target AGX data structure
    :return: AGX data object
    """
    agx_list = None
    if agx_type == agx.Vec3:
        agx_list = agx.Vec3(np_array[0].item(), np_array[1].item(), np_array[2].item())
    elif agx_type == agx.Quat:
        agx_list = agx.Quat(np_array[0].item(), np_array[1].item(), np_array[2].item(), np_array[3].item())
    elif agx_type == agx.OrthoMatrix3x3:
        agx_list = agx.OrthoMatrix3x3(np_array[0, 0].item(), np_array[0, 1].item(), np_array[0, 2].item(),
                                      np_array[1, 0].item(), np_array[1, 1].item(), np_array[1, 2].item(),
                                      np_array[2, 0].item(), np_array[2, 1].item(), np_array[2, 2].item())
    else:
        logger.warning('Conversion for {} type is not supported.'.format(agx_type))

    return agx_list


def get_cable_segment_edges(cable):
    """Get AGX Cable segments' begin and end positions.
    :param cable: AGX Cable object
    :return: NumPy array with segments' edge positions
    """
    num_segments = cable.getNumSegments()
    cable_state = np.zeros(shape=(3, num_segments + 1))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            position_begin = segment_iterator.getBeginPosition()
            cable_state[:3, i] = to_numpy_array(position_begin)
            if i == num_segments - 1:
                position_end = segment_iterator.getEndPosition()
                cable_state[:3, -1] = to_numpy_array(position_end)

            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_state


def create_body(shape, name="", position=agx.Vec3(0, 0, 0), rotation=agx.OrthoMatrix3x3(),
                geometry_transform=agx.AffineMatrix4x4(), motion_control=agx.RigidBody.DYNAMICS, material=None):
    """Helper function that creates a RigidBody according to the given definition.
    Returns the body itself, it's geometry and the OSG node that was created for it.
    :param agxCollide.Shape shape: shape of object.
    :param string name: Optional. Defaults to "". The name of the new body.
    :param agx.Vec3 position: Position of the object in world coordinates.
    :param agx.OrthoMatrix3x3 rotation: Rotation of the object in world coordinate frames
    :param agx.AffineMatrix4x4 geometry_transform: Optional. Defaults to identity transformation. The local
    transformation of the shape relative to the body.
    :param agx.RigidBody.MotionControl motion_control: Optional. Defaults to DYNAMICS.
    :param agx.Material material: Optional. Ignored if not given. Material assigned to the geometry created for the
    body.
    :return: assembly
    """
    assembly = agxSDK.Assembly()
    assembly.setName(name)
    try:
        body = agx.RigidBody(name)
        geometry = agxCollide.Geometry(shape)
        geometry.setName(name)
        if material:
            geometry.setMaterial(material)
        body.add(geometry, geometry_transform)
        body.setMotionControl(motion_control)
        assembly.add(body)
        assembly.setPosition(position)
        assembly.setRotation(rotation)

    except Exception as exception:
        logger.error(exception)
    finally:
        return assembly


def create_ring(name, radius, element_shape, num_elements, constraint_type=agx.LockJoint, rotation_shift=0,
                translation_shift=0, compliance=None, material=None, center=agx.Vec3(), normal=agx.Vec3.Z_AXIS()):
    """Creates a Ring object.
    :param string name: name of ring object as a string
    :param float radius: radius of the ring circumference, centered on the composing elements
    :param agxCollide.Shape element_shape: AGX shape type to be used as building block of ring
    :param int num_elements: number of elements of element_type which will be used to construct ring
    :param agx.Constraint constraint_type: type of constraint that should connect each element
    :param float rotation_shift: positive rotation around z axis of seed element (radians). Useful for shapes which are
    initialized along axis other than x.
    :param float translation_shift: translation of constraints, off the center of mass, along y axis of the object
    :param list compliance: compliance of constraints along 6DOF linking each element of the ring
    :param agx.Vec3 center: position in world coordinates of the center of the ring
    :param agx.Material material: material the ring elements are made of
    :param agx.Vec3 normal: unit vector placed at the center position to define plane where the ring should lie
    :return assembly
    """
    if compliance is None:
        compliance = [0, 0, 0, 0, 0, 0]
    elements = []
    constraints = []
    assembly = agxSDK.Assembly()
    try:
        assembly_rotation = agx.OrthoMatrix3x3()
        assembly_rotation.setRotate(agx.Vec3.Z_AXIS(), normal)

        prev_element = None
        step_angle = 360 / num_elements

        # Frames for constraints
        frame1 = agx.Frame()
        frame1.setLocalTranslate(0, -translation_shift, 0)
        frame1.setLocalRotate(agx.EulerAngles(0, 0, -math.radians(step_angle)))
        frame2 = agx.Frame()
        frame2.setLocalTranslate(0, translation_shift, 0)
        for i in range(1, num_elements + 1):
            x = math.sin(math.radians(i * step_angle)) * radius
            y = math.cos(math.radians(i * step_angle)) * radius
            z = 0.0
            position = agx.Vec3(x, y, z)

            element = create_ring_element(name + "_" + str(i), element_shape, material)
            element.setPosition(position)
            rotation = agx.OrthoMatrix3x3(agx.EulerAngles(0, 0, - math.radians(i * step_angle) + rotation_shift))
            element.setRotation(rotation)
            assembly.add(element)

            elements.append(element)

            if prev_element:
                element_constraint = constraint_type(prev_element, frame1, element, frame2)
                element_constraint.setName("ring_constraint_" + str(i))
                constraints.append(element_constraint)
                assembly.add(element_constraint)

                geometry = prev_element.getGeometry(name + "_" + str(i - 1))
                geometry.setEnableCollisions(element.getGeometry(name + "_" + str(i)), False)

            prev_element = element

        element_constraint = constraint_type(elements[-1], frame1, elements[0], frame2)
        element_constraint.setName("ring_constraint_" + str(num_elements + 1))
        constraints.append(element_constraint)
        assembly.add(element_constraint)

        geometry = elements[-1].getGeometry(name + "_" + str(num_elements))
        geometry.setEnableCollisions(elements[0].getGeometry(name + "_" + str(1)), False)

        # Set compliance of ring:
        for c in constraints:
            c.setCompliance(compliance[0], agx.LockJoint.TRANSLATIONAL_1)
            c.setCompliance(compliance[1], agx.LockJoint.TRANSLATIONAL_2)
            c.setCompliance(compliance[2], agx.LockJoint.TRANSLATIONAL_3)
            c.setCompliance(compliance[3], agx.LockJoint.ROTATIONAL_1)
            c.setCompliance(compliance[4], agx.LockJoint.ROTATIONAL_2)
            c.setCompliance(compliance[5], agx.LockJoint.ROTATIONAL_3)

        assembly.setPosition(center)
        assembly.setRotation(assembly_rotation)

    except Exception as exception:
        logger.error(exception)
    finally:
        return assembly


def create_ring_element(name, element_shape, material):
    """Creates single ring element based on shape and material.
    :param string name: name of ring element
    :param agxCollide.Shape element_shape: enum value of ring geometric shape
    :param agx.Material material: material of rigid body
    :return ring_element"""
    if element_shape.getType() in [agxCollide.Shape.SPHERE, agxCollide.Shape.CAPSULE,
                                   agxCollide.Shape.BOX, agxCollide.Shape.CYLINDER]:
        element = agx.RigidBody(name)
        geometry = agxCollide.Geometry(element_shape.clone())
        geometry.setName(name)
        if material:
            geometry.setMaterial(material)
        element.add(geometry, agx.AffineMatrix4x4())
        return element
    else:
        return None


def create_locked_prismatic_base(name, rigid_body, compliance=0, damping=1 / 3, position_ranges=None, motor_ranges=None,
                                 locked_at_zero_speed=None, lock_status=None, compute_forces=False,
                                 radius=0.005, length=0.050):
    """Creates a prismatic, collision free, base object and attaches a rigid body to it, via LockJoint.
    :param string name: name of prismatic base object as a string
    :param agx.RigidBody rigid_body: AGX rigid body object which should be attached to prismatic base
    :param float compliance: compliance of the LockJoint which attaches the rigid body to the base
    :param float damping: damping of the LockJoint which attaches the rigid body to the base
    :param list position_ranges: a list containing three tuples with the position range for each constraint (x,y,z)
    :param list motor_ranges: a list containing three tuples with the position range for each constraint (x,y,z)
    :param list locked_at_zero_speed: a list containing three Booleans indicating zero speed behaviour each constraint
    (x,y,z)
    :param list lock_status: a list containing boolean values for whether to activate the constraint locks (x,y,z)
    :param boolean compute_forces: set whether forces are computed for this base.
    :param float radius: radius of the cylinders making up the base. For visualization purposes only.
    :param float length: radius of the cylinders making up the base. For visualization purposes only.
    :return assembly
    """
    if position_ranges is None:
        position_ranges = [(-1, 1), (-1, 1), (-1, 1)]
    if motor_ranges is None:
        motor_ranges = [(-1, 1), (-1, 1), (-1, 1)]
    if locked_at_zero_speed is None:
        locked_at_zero_speed = [False, False, False]
    if lock_status is None:
        lock_status = [False, False, False]

    position_ranges = {'x': position_ranges[0], 'y': position_ranges[1], 'z': position_ranges[2]}
    motor_ranges = {'x': motor_ranges[0], 'y': motor_ranges[1], 'z': motor_ranges[2]}
    locked_at_zero_speed = {'x': locked_at_zero_speed[0],
                            'y': locked_at_zero_speed[1],
                            'z': locked_at_zero_speed[2]}

    assembly = create_prismatic_base(name, radius, length, compute_forces,
                                     position_ranges, motor_ranges, locked_at_zero_speed, lock_status)

    # Add constraint between base and rigid body
    f1 = agx.Frame()
    f2 = agx.Frame()
    base_z_body = assembly.getRigidBody(name + "_base_z")
    result = agx.Constraint.calculateFramesFromWorld(rigid_body.getPosition(), agx.Vec3(0, 1, 0), rigid_body, f1,
                                                     base_z_body, f2)
    if not result:
        logger.error("There was an error when calculating frames from world.")
    joint_rb = agx.LockJoint(rigid_body, f1, base_z_body, f2)
    joint_rb.setName(name + "_joint_rb")
    joint_rb.setEnableComputeForces(True)
    joint_rb.setCompliance(compliance)
    joint_rb.setDamping(damping)
    assembly.add(joint_rb)

    return assembly


def create_hinge_prismatic_base(name, rigid_body, compliance=0, damping=1 / 3, position_ranges=None,
                                motor_ranges=None, locked_at_zero_speed=None, lock_status=None, axis=agx.Vec3(1, 0, 0),
                                compute_forces=False, radius=0.005, length=0.050):
    """Creates a prismatic, collision free, base object and attaches a rigid body to it, via Hinge.
    :param string name: name of prismatic base object as a string
    :param agx.RigidBody rigid_body: AGX rigid body object which should be attached to prismatic base
    :param float compliance: compliance of the Hinge which attaches the rigid body to the base
    :param float damping: damping of the Hinge which attaches the rigid body to the base
    :param list position_ranges: a list containing three tuples with the position range for each constraint (x,y,z,rb)
    :param list motor_ranges: a list containing three tuples with the position range for each constraint (x,y,z,rb)
    :param list locked_at_zero_speed: a list containing three Booleans indicating zero speed behaviour each constraint
    (x,y,z,rb)
    :param list lock_status: a list containing boolean values for whether to activate the constraint locks (x,y,z)
    :param agx.Vec3 axis: vector determining axis of rotation of rigid body
    :param boolean compute_forces: set whether forces are computed for this base.
    :param float radius: radius of the cylinders making up the base. For visualization purposes only.
    :param float length: radius of the cylinders making up the base. For visualization purposes only.
    :return assembly
    """
    if locked_at_zero_speed is None:
        locked_at_zero_speed = [False, False, False, False]
    if motor_ranges is None:
        motor_ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    if position_ranges is None:
        position_ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    if lock_status is None:
        lock_status = [False, False, False]

    position_ranges = {'x': position_ranges[0], 'y': position_ranges[1], 'z': position_ranges[2],
                       'rb': position_ranges[3]}
    motor_ranges = {'x': motor_ranges[0], 'y': motor_ranges[1], 'z': motor_ranges[2], 'rb': motor_ranges[3]}
    locked_at_zero_speed = {'x': locked_at_zero_speed[0],
                            'y': locked_at_zero_speed[1],
                            'z': locked_at_zero_speed[2],
                            'rb': locked_at_zero_speed[3]}

    assembly = create_prismatic_base(name, radius, length, compute_forces,
                                     position_ranges, motor_ranges, locked_at_zero_speed, lock_status)

    # Add constraint between base and rigid body
    base_z_body = assembly.getRigidBody(name + "_base_z")
    f1 = agx.Frame()
    f2 = agx.Frame()
    result = agx.Constraint.calculateFramesFromWorld(rigid_body.getPosition(), axis, rigid_body, f1,
                                                     base_z_body, f2)
    if not result:
        logger.error("There was an error when calculating frames from world.")

    joint_rb = agx.Hinge(rigid_body, f1, base_z_body, f2)
    joint_rb.setName(name + "_joint_rb")
    joint_rb.setEnableComputeForces(True)
    joint_rb.setCompliance(compliance)
    joint_rb.setDamping(damping)
    assembly.add(joint_rb)

    # Enable ranges and motors
    joint_rb_range = joint_rb.getRange1D()
    joint_rb_range.setEnable(True)
    joint_rb_range.setRange(position_ranges['rb'][0], position_ranges['rb'][1])
    joint_rb_motor = joint_rb.getMotor1D()
    joint_rb_motor.setEnable(True)
    joint_rb_motor.setForceRange(motor_ranges['rb'][0], motor_ranges['rb'][1])
    joint_rb_motor.setLockedAtZeroSpeed(locked_at_zero_speed['rb'])

    return assembly


def create_universal_prismatic_base(name, rigid_body, compliance=0, damping=1 / 3, position_ranges=None,
                                    motor_ranges=None, locked_at_zero_speed=None, lock_status=None,
                                    compute_forces=False, radius=0.005, length=0.050):
    """Creates a prismatic, collision free, base object and attaches a rigid body to it, via UniversalJoint. Note that
    at this time, the UniversalJoint constraint has known issues. I should be avoided if possible.
    :param string name: name of prismatic base object as a string
    :param agx.RigidBody rigid_body: AGX rigid body object which should be attached to prismatic base
    :param float compliance: compliance of the UniversalJoint which attaches the rigid body to the base
    :param float damping: damping of the UniversalJoint which attaches the rigid body to the base
    :param list position_ranges: a list containing three tuples with the position range for each constraint (x,y,z,2rb)
    :param list motor_ranges: a list containing three tuples with the position range for each constraint (x,y,z,2rb)
    :param list locked_at_zero_speed: a list containing three Booleans indicating zero speed behaviour each constraint
    (x,y,z,2rb)
    :param list lock_status: a list containing boolean values for whether to activate the constraint locks (x,y,z)
    :param boolean compute_forces: set whether forces are computed for this base.
    :param float radius: radius of the cylinders making up the base. For visualization purposes only.
    :param float length: radius of the cylinders making up the base. For visualization purposes only.
    :return assembly
    """
    if locked_at_zero_speed is None:
        locked_at_zero_speed = [False, False, False, False, False]
    if motor_ranges is None:
        motor_ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    if position_ranges is None:
        position_ranges = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
    if lock_status is None:
        lock_status = [False, False, False]

    position_ranges = {'x': position_ranges[0], 'y': position_ranges[1], 'z': position_ranges[2],
                       'rb_1': position_ranges[3], 'rb_2': position_ranges[4]}
    motor_ranges = {'x': motor_ranges[0], 'y': motor_ranges[1], 'z': motor_ranges[2], 'rb_1': motor_ranges[3],
                    'rb_2': motor_ranges[4]}
    locked_at_zero_speed = {'x': locked_at_zero_speed[0],
                            'y': locked_at_zero_speed[1],
                            'z': locked_at_zero_speed[2],
                            'rb_1': locked_at_zero_speed[3],
                            'rb_2': locked_at_zero_speed[4]}

    assembly = create_prismatic_base(name, radius, length, compute_forces,
                                     position_ranges, motor_ranges, locked_at_zero_speed, lock_status)

    # Add constraint between base and rigid body
    f1 = agx.Frame()
    f2 = agx.Frame()
    base_z_body = assembly.getRigidBody(name + "_base_z")
    result = agx.Constraint.calculateFramesFromWorld(rigid_body.getPosition(), agx.Vec3(0, 0, 1), rigid_body, f1,
                                                     base_z_body, f2)
    if not result:
        logger.error("There was an error when calculating frames from world.")
    joint_rb = agx.UniversalJoint(rigid_body, f1, base_z_body, f2)
    joint_rb.setName(name + "_joint_rb")
    joint_rb.setEnableComputeForces(True)
    joint_rb.setCompliance(compliance)
    joint_rb.setDamping(damping)
    assembly.add(joint_rb)

    # Enable ranges and motors
    joint_rb_range_1 = joint_rb.getRange1D(agx.UniversalJoint.ROTATIONAL_CONTROLLER_1)
    joint_rb_range_1.setEnable(True)
    joint_rb_range_1.setRange(position_ranges['rb_1'][0], position_ranges['rb_1'][1])
    joint_rb_motor_1 = joint_rb.getMotor1D(agx.UniversalJoint.ROTATIONAL_CONTROLLER_1)
    joint_rb_motor_1.setName(name + "_joint_rb_motor_1")
    joint_rb_motor_1.setEnable(True)
    joint_rb_motor_1.setForceRange(motor_ranges['rb_1'][0], motor_ranges['rb_1'][1])
    joint_rb_motor_1.setLockedAtZeroSpeed(locked_at_zero_speed['rb_1'])
    joint_rb_range_2 = joint_rb.getRange1D(agx.UniversalJoint.ROTATIONAL_CONTROLLER_2)
    joint_rb_range_2.setEnable(True)
    joint_rb_range_2.setRange(position_ranges['rb_2'][0], position_ranges['rb_2'][1])
    joint_rb_motor_2 = joint_rb.getMotor1D(agx.UniversalJoint.ROTATIONAL_CONTROLLER_2)
    joint_rb_motor_2.setName(name + "_joint_rb_motor_2")
    joint_rb_motor_2.setEnable(True)
    joint_rb_motor_2.setForceRange(motor_ranges['rb_2'][0], motor_ranges['rb_2'][1])
    joint_rb_motor_2.setLockedAtZeroSpeed(locked_at_zero_speed['rb_2'])

    return assembly


def create_prismatic_base(name, radius, length, compute_forces,
                          position_ranges, motor_ranges, locked_at_zero_speed, lock_status):
    assembly = agxSDK.Assembly()
    assembly.setName(name + "_prismatic_base")

    rotation_y_to_z = agx.OrthoMatrix3x3()
    rotation_y_to_z.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.Z_AXIS())
    rotation_y_to_x = agx.OrthoMatrix3x3()
    rotation_y_to_x.setRotate(agx.Vec3.Y_AXIS(), agx.Vec3.X_AXIS())

    base_z = create_body(name=name + "_base_z",
                         shape=agxCollide.Cylinder(radius, length),
                         position=agx.Vec3(0, 0, 0),
                         rotation=rotation_y_to_z,
                         motion_control=agx.RigidBody.DYNAMICS)
    assembly.add(base_z)

    base_y = create_body(name=name + "_base_y",
                         shape=agxCollide.Cylinder(radius, length),
                         position=agx.Vec3(0, 0, 0),
                         motion_control=agx.RigidBody.DYNAMICS)
    assembly.add(base_y)

    base_x = create_body(name=name + "_base_x",
                         shape=agxCollide.Cylinder(radius, length),
                         position=agx.Vec3(0, 0, 0),
                         rotation=rotation_y_to_x,
                         motion_control=agx.RigidBody.DYNAMICS)
    assembly.add(base_x)

    base_x_body = base_x.getRigidBody(name + "_base_x")
    base_y_body = base_y.getRigidBody(name + "_base_y")
    base_z_body = base_z.getRigidBody(name + "_base_z")
    base_x_body.getGeometry(name + "_base_x").setEnableCollisions(False)
    base_y_body.getGeometry(name + "_base_y").setEnableCollisions(False)
    base_z_body.getGeometry(name + "_base_z").setEnableCollisions(False)

    # Add prismatic joints between bases
    joint_base_x = agx.Prismatic(agx.Vec3(1, 0, 0), base_x_body)
    joint_base_x.setEnableComputeForces(compute_forces)
    joint_base_x.setName(name + "_joint_base_x")
    assembly.add(joint_base_x)

    joint_base_y = agx.Prismatic(agx.Vec3(0, -1, 0), base_x_body, base_y_body)
    joint_base_y.setEnableComputeForces(compute_forces)
    joint_base_y.setName(name + "_joint_base_y")
    assembly.add(joint_base_y)

    joint_base_z = agx.Prismatic(agx.Vec3(0, 0, -1), base_y_body, base_z_body)
    joint_base_z.setEnableComputeForces(compute_forces)
    joint_base_z.setName(name + "_joint_base_z")
    assembly.add(joint_base_z)

    # Set and enable position ranges
    joint_base_x_range = joint_base_x.getRange1D()
    joint_base_x_range.setEnable(True)
    joint_base_x_range.setRange(position_ranges['x'][0], position_ranges['x'][1])

    joint_base_y_range = joint_base_y.getRange1D()
    joint_base_y_range.setEnable(True)
    joint_base_y_range.setRange(position_ranges['y'][0], position_ranges['y'][1])

    joint_base_z_range = joint_base_z.getRange1D()
    joint_base_z_range.setEnable(True)
    joint_base_z_range.setRange(position_ranges['z'][0], position_ranges['z'][1])

    # Set and enable motor ranges
    joint_base_x_motor = joint_base_x.getMotor1D()
    joint_base_x_motor.setEnable(True)
    joint_base_x_motor.setForceRange(motor_ranges['x'][0], motor_ranges['x'][1])
    joint_base_x_motor.setLockedAtZeroSpeed(locked_at_zero_speed['x'])

    joint_base_y_motor = joint_base_y.getMotor1D()
    joint_base_y_motor.setEnable(True)
    joint_base_y_motor.setForceRange(motor_ranges['y'][0], motor_ranges['y'][1])
    joint_base_y_motor.setLockedAtZeroSpeed(locked_at_zero_speed['y'])

    joint_base_z_motor = joint_base_z.getMotor1D()
    joint_base_z_motor.setEnable(True)
    joint_base_z_motor.setForceRange(motor_ranges['z'][0], motor_ranges['z'][1])
    joint_base_z_motor.setLockedAtZeroSpeed(locked_at_zero_speed['z'])

    # Set and enable locks
    if lock_status[0]:
        joint_base_x_lock = joint_base_x.getLock1D()
        joint_base_x_lock.setEnable(True)

    if lock_status[1]:
        joint_base_y_lock = joint_base_y.getLock1D()
        joint_base_y_lock.setEnable(True)

    if lock_status[2]:
        joint_base_z_lock = joint_base_z.getLock1D()
        joint_base_z_lock.setEnable(True)

    return assembly
