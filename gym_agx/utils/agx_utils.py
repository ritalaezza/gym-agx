import agx
import agxIO
import agxSDK
import agxCollide
from agxPythonModules.utils.numpy_utils import create_numpy_array

import os
import math
import logging
import tempfile
import numpy as np
from enum import Enum

try:
    import matplotlib.pyplot as plt
except:
    print("Could not find matplotlib. Continuing without displaying depth buffer image.")
    plt = None

logger = logging.getLogger('gym_agx.utils')


class EndEffectorConstraint:
    class Dof(Enum):
        X_TRANSLATIONAL = 0,
        Y_TRANSLATIONAL = 1,
        Z_TRANSLATIONAL = 2,
        X_ROTATIONAL = 3,
        Y_ROTATIONAL = 4,
        Z_ROTATIONAL = 5

    def __init__(self, end_effector_dof, compute_forces_enabled, velocity_control, compliance_control, velocity_index,
                 compliance_index):
        """End effector constraint object, defining important parameters.
        :param end_effector_dof: (GDof) degree of freedom of end effector that this constraint controls
        :param compute_forces_enabled: (Boolean) force and torque can be measured
        :param velocity_control: (Boolean) is velocity controlled
        :param compliance_control: (Boolean) is compliance controlled
        :param velocity_index: (int) index of action vector which controls velocity of this constraint's motor
        :param compliance_index: (int) index of action vector which controls compliance of this constraint's motor"""
        self.end_effector_dof = end_effector_dof
        self.velocity_control = velocity_control
        self.compute_forces_enabled = compute_forces_enabled
        self.compliance_control = compliance_control
        self.velocity_index = velocity_index
        self.compliance_index = compliance_index

    @property
    def is_active(self):
        return True if self.velocity_control or self.compliance_control else False


class EndEffector:
    last_action_index = -1

    def __init__(self, name, controllable, observable, max_velocity=1, max_acceleration=1, min_compliance=0,
                 max_compliance=1e6):
        """EndEffector class which keeps track of end effector constraints and action indices.
        :param name
        :param controllable
        :param observable
        :param max_velocity
        :param max_acceleration
        :param max_compliance
        """
        self.name = name
        self.controllable = controllable
        self.observable = observable
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_compliance = min_compliance
        self.max_compliance = max_compliance
        self.constraints = {}

    def add_constraint(self, name, end_effector_dof, compute_forces_enabled=False, velocity_control=False,
                       compliance_control=False):
        velocity_index = None
        compliance_index = None
        if velocity_control:
            self.last_action_index += 1
            velocity_index = self.last_action_index
        if compliance_control:
            self.last_action_index += 1
            compliance_index = self.last_action_index
        end_effector_constraint = EndEffectorConstraint(end_effector_dof, compute_forces_enabled, velocity_control,
                                                        compliance_control, velocity_index, compliance_index)
        self.constraints.update({name: end_effector_constraint})

    def apply_control(self, sim, action, dt):
        if self.controllable:
            for key, constraint in self.constraints.items():
                joint = sim.getConstraint1DOF(key)
                motor = joint.getMotor1D()
                if constraint.velocity_control:
                    current_velocity = self.get_velocity(sim, constraint.end_effector_dof)
                    velocity = self.rescale_velocity(action[constraint.velocity_index], current_velocity, dt)
                    logger.debug(f'{key} velocity: {velocity}')
                    motor.setSpeed(np.float64(velocity))
                if constraint.compliance_control:
                    motor_param = motor.getRegularizationParameters()
                    compliance = self.rescale_compliance(action[constraint.compliance_index])
                    motor_param.setCompliance(np.float64(compliance))
        else:
            logger.debug("Received apply_control command for uncontrollable end effector.")

    def get_velocity(self, sim, constraint_dof):
        end_effector = sim.getRigidBody(self.name)
        if constraint_dof == EndEffectorConstraint.Dof.X_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[0]
        elif constraint_dof == EndEffectorConstraint.Dof.Y_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[1]
        elif constraint_dof == EndEffectorConstraint.Dof.Z_TRANSLATIONAL:
            velocity = end_effector.getVelocity()[2]
        elif constraint_dof == EndEffectorConstraint.Dof.X_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[0]
        elif constraint_dof == EndEffectorConstraint.Dof.Y_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[1]
        elif constraint_dof == EndEffectorConstraint.Dof.Z_ROTATIONAL:
            velocity = end_effector.getAngularVelocity()[2]
        else:
            logger.error("Unexpected EndEffectorConstraint.Dof.")

        return velocity

    def get_state(self, sim):
        if self.observable:
            state = []
            for key, constraint in self.constraints.items():
                if constraint.compute_forces_enabled:
                    constraint_state = get_end_effector_state(sim, key).ravel()
                    logger.debug(f"{key} state: {constraint_state}")
                    state.append(constraint_state)
            return np.asarray(state)
        else:
            logger.error("Received get_state command for unobservable end effector.")

    def rescale_velocity(self, velocity, current_velocity, dt):
        logger.debug(f'Current velocity: {current_velocity}')
        logger.debug(f'Initial target velocity: {velocity}')
        if abs(velocity - current_velocity) > self.max_acceleration:
            velocity = current_velocity + np.sign(velocity - current_velocity) * (self.max_acceleration * dt)
        if abs(velocity) > self.max_velocity:
            velocity = self.max_velocity * np.sign(velocity)
        logger.debug(f'Rescaled target velocity: {velocity}')
        return velocity

    def rescale_compliance(self, compliance):
        # Assumes an action range between -1 and 1
        return (compliance + 1) / 2 * (self.max_compliance - self.min_compliance) + self.min_compliance


class CameraSpecs:
    def __init__(self, eye, center, up, light_position, light_direction):
        self.camera_pose = {'eye': eye,
                            'center': center,
                            'up': up}
        self.light_pose = {'light_position': light_position,
                           'light_direction': light_direction}


class ShowImages(agxSDK.StepEventListener):
    def __init__(self, rti_depth, rti_color, size_depth, size_color):
        super().__init__()

        self.rti_depth = rti_depth
        self.rti_color = rti_color
        self.size_color = size_color
        self.size_depth = size_depth

        if plt is not None:
            self.fig, self.ax = plt.subplots(2, figsize=(10, 10))
            self.obj_color = self.ax[0].imshow(np.ones(self.size_color, dtype=np.uint8))
            self.obj_depth = self.ax[1].imshow(np.ones((self.size_depth[0], size_depth[1]), dtype=np.float32), vmin=0,
                                               vmax=1, cmap='gray')

            plt.ion()
            plt.show()

    def post(self, t):
        # Get pointer to the image
        ptr_color = self.rti_color.getImageData()
        image_color = create_numpy_array(ptr_color, self.size_color, np.uint8)

        ptr_depth = self.rti_depth.getImageData()
        image_depth = create_numpy_array(ptr_depth, self.size_depth, np.float32)

        # check that numpy arrays are created correctly
        if image_color is None or image_depth is None:
            return

        if plt is not None:
            self.obj_color.set_data(np.flip(image_color, 0))
            self.obj_depth.set_data(np.flip(np.squeeze(image_depth), 0))
            plt.draw()
            plt.pause(1e-5)
        else:
            print("Max depth buffer value at time {}: {}".format(t, np.max(image_depth)))
            print("Min depth buffer value at time {}: {}".format(t, np.min(image_depth)))

        # save images to disk at second timestep
        if self.getSimulation().getTimeStep() < t < self.getSimulation().getTimeStep() * 3:
            temp_dir = tempfile.mkdtemp(prefix="agxRenderToImage_")
            filename_color = os.path.join(temp_dir, "color.png")
            filename_depth = os.path.join(temp_dir, "depth.png")
            # Try to save color image to disk. This will work.
            if self.rti_color.saveImage(filename_color):
                print("Saving color image as {} succeeded ".format(filename_color))
            else:
                print("Saving color image as {} failed".format(filename_color))
            # Try to save depth image to disk. This will fail.
            if self.rti_depth.saveImage(filename_depth):
                print("Saving depth image as {} succeeded".format(filename_depth))
            else:
                print("Saving depth image as {} failed".format(filename_depth))


class KeyboardMotorHandler(agxSDK.GuiEventListener):
    """General class to control simulations using keyboard.
    """

    def __init__(self, key_motor_maps):
        """Each instance of this class takes a dictionary
        :param dict key_motor_maps: This dictionary of tuples will assign a motor per key and set the desired speed when
        pressed, taking into account desired direction {agxSDK.GuiEventListener.KEY: (motor, speed)}
        :return Boolean handled: indicates success"""
        super().__init__()
        self.key_motor_maps = key_motor_maps

    def keyboard(self, pressed_key, x, y, alt, down):
        handled = False
        for key, motor_map in self.key_motor_maps.items():
            if pressed_key == key:
                if down:
                    motor_map[0].setSpeed(motor_map[1])
                else:
                    motor_map[0].setSpeed(0)
            handled = True

        return handled


class InfoPrinter(agxSDK.StepEventListener):
    def __init__(self, app, text_table, text_color):
        """Write help text. textTable is a table with strings that will be drawn above the default text.
        :param app: OSG Example Application object
        :param text_table: table with text to be printed on screen
        :return: AGX simulation object
        """
        super().__init__(agxSDK.StepEventListener.POST_STEP)
        self.text_table = text_table
        self.text_color = text_color
        self.app = app
        self.row = 31

    def post(self, t):
        if self.textTable:
            color = agx.Vec4(0.3, 0.6, 0.7, 1)
            if self.text_color:
                color = self.text_color
            for i, v in enumerate(self.text_table):
                self.app.getSceneDecorator().setText(i, str(v[0]) + " " + v[1](), color)


class HelpListener(agxSDK.StepEventListener):
    def __init__(self, app, text_table):
        """Write information to screen from lambda functions during the simulation.
        :param app: OSG Example Application object
        :param text_table: table with text to be printed on screen
        :return: AGX simulation object
        """
        super().__init__(agxSDK.StepEventListener.PRE_STEP)
        self.text_table = text_table
        self.app = app
        self.row = 31

    def pre(self, t):
        if t > 3.0:
            self.app.getSceneDecorator().setText(self.row, "", agx.Vec4f(1, 1, 1, 1))
            if self.text_table:
                start_row = self.row - len(self.text_table)
                for i, v in enumerate(self.text_table):
                    self.app.getSceneDecorator().setText(start_row + i - 1, "", agx.Vec4f(0.3, 0.6, 0.7, 1))

            self.getSimulation().remove(self)

    def addNotification(self):
        if self.text_table:
            start_row = self.row - len(self.text_table)
            for i, v in enumerate(self.text_table):
                self.app.getSceneDecorator().setText(start_row + i - 1, v, agx.Vec4f(0.3, 0.6, 0.7, 1))

        self.app.getSceneDecorator().setText(self.row, "Press e to start simulation", agx.Vec4f(0.3, 0.6, 0.7, 1))


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
        logger.warning('Conversion for type {} type is not supported.'.format(agx_type))

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
        logger.warning('Conversion for type {} type is not supported.'.format(agx_type))

    return agx_list


def get_cable_pose(cable, gain=1):
    """Get AGX Cable segments' positions and rotations.
    :param cable: AGX Cable object
    :param gain: gives possibility to rescale position values
    :return: NumPy array with segments' position and rotations
    """
    num_segments = cable.getNumSegments()
    cable_pose = np.zeros(shape=(7, num_segments))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            position = segment_iterator.getGeometry().getPosition() * gain
            cable_pose[:3, i] = to_numpy_array(position)

            rotation = segment_iterator.getGeometry().getRotation()
            cable_pose[3:, i] = to_numpy_array(rotation)
            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_pose


def get_cable_state(cable, gain=1):
    """Get AGX Cable segments' begin and end positions.
    :param cable: AGX Cable object
    :param gain: gives possibility to rescale position values
    :return: NumPy array with segments' position
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

    return cable_state * gain


def get_end_effector_state(sim, key, include_position=False, gain=1):
    """Get AGX 'end_effector' positions, force and torque.
    :param sim: AGX Dynamics simulation object
    :param key: name of end effector
    :param include_position: Boolean to determine if end effector position is part of state
    :param gain: gives possibility to rescale position values
    :return: NumPy array with end effector position, rotations, force and torque
    """
    end_effector = sim.getRigidBody(key)
    if include_position:
        state = np.zeros(shape=(3, 3))
        state[:, 0] = to_numpy_array(end_effector.getPosition()) * gain
        state[:, 1], state[:, 2] = get_force_torque(sim, end_effector, key)
    else:
        state = np.zeros(shape=(3, 2))
        state[:, 0], state[:, 1] = get_force_torque(sim, end_effector, key)
    logger.debug("End effector {} state (force, torque): {}".format(key, state))

    return state


def get_force_torque(sim, rigid_body, constraint_name):
    """Gets force an torque on rigid object, computed by a constraint defined by 'constraint_name'.
    :param sim: AGX Simulation object
    :param rigid_body: RigidBody object on which to compute force and torque
    :param constraint_name: Name indicating which constraint contains force torque information for this object
    :return: force an torque
    """
    force = agx.Vec3()
    torque = agx.Vec3()

    constraint = sim.getConstraint(constraint_name)
    constraint.getLastForce(rigid_body, force, torque)

    force = to_numpy_array(force)
    torque = to_numpy_array(torque)

    return force, torque


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


def create_ring(name, radius, element_shape, num_elements, constraint, rotation_shift=0,
                compliance=0, material=None, center=agx.Vec3(), normal=agx.Vec3.Z_AXIS()):
    """Creates a Ring object.
    :param string name: name of ring object as a string
    :param float radius: radius of the ring circumference, centered on the composing elements
    :param agxCollide.Shape element_shape: AGX shape type to be used as building block of ring
    :param int num_elements: number of elements of element_type which will be used to construct ring
    :param agx.Constraint constraint: type of constraint that should connect each element
    :param float rotation_shift: positive rotation around z axis of seed element (radians). Useful for shapes which are
    initialized along axis other than x.
    :param float compliance: compliance of constraints linking each element of the ring
    :param agx.Vec3 center: position in world coordinates of the center of the ring
    :param agx.Material material: material the ring elements are made of
    :param agx.Vec3 normal: unit vector placed at the center position to define plane where the ring should lie
    :return assembly
    """
    elements = []
    constraints = []
    assembly = agxSDK.Assembly()
    try:
        assembly_rotation = agx.OrthoMatrix3x3()
        assembly_rotation.setRotate(agx.Vec3.Z_AXIS(), normal)

        prev_element = None
        step_angle = 360 / num_elements
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
                f1 = agx.Frame()
                f2 = agx.Frame()
                result = agx.Constraint.calculateFramesFromWorld(element.getPosition(), agx.Vec3(0, 1, 0),
                                                                 prev_element, f1, element, f2)
                if not result:
                    logger.debug("Problem calculating frames when creating ring.")

                element_constraint = constraint(prev_element, f1, element, f2)
                constraints.append(element_constraint)
                assembly.add(element_constraint)

                geometry = prev_element.getGeometry(name + "_" + str(i-1))
                geometry.setEnableCollisions(element.getGeometry(name + "_" + str(i)), False)

            prev_element = element

        # Constraint first and last element
        f1 = agx.Frame()
        f2 = agx.Frame()
        result = agx.Constraint.calculateFramesFromWorld(elements[0].getPosition(), agx.Vec3(0, 1, 0),
                                                         elements[-1], f1, elements[0], f2)
        if not result:
            logger.debug("Problem calculating last frame when creating ring.")

        element_constraint = constraint(elements[-1], f1, elements[0], f2)
        constraints.append(element_constraint)
        assembly.add(element_constraint)

        geometry = elements[-1].getGeometry(name + "_" + str(num_elements))
        geometry.setEnableCollisions(elements[0].getGeometry(name + "_" + str(1)), False)

        # Set compliance of ring:
        for c in constraints:
            c.setCompliance(compliance)

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


def create_prismatic_base(name, rigid_body, compliance=0, damping=1 / 3, position_ranges=None,
                          motor_ranges=None, locked_at_zero_speed=None,
                          radius=0.005, length=0.050):
    """Creates a prismatic, collision free, base object and attaches a rigid body to it.
    :param string name: name of prismatic base object as a string
    :param agx.RigidBody rigid_body: AGX rigid body object which should be attached to prismatic base
    :param float compliance: compliance of the LockJoint which attaches the rigid body to the base
    :param float damping: damping of the LockJoint which attaches the rigid body to the base
    :param list position_ranges: a list containing three tuples with the position range for each coordinate (x,y,z)
    :param list motor_ranges: a list containing three tuples with the position range for each coordinate (x,y,z)
    :param list locked_at_zero_speed: a list containing three Booleans indicating zero speed behaviour each coordinate
    (x,y,z)
    :param float radius: radius of the cylinders making up the base. For visualization purposes only.
    :param float length: radius of the cylinders making up the base. For visualization purposes only.
    :return assembly
    """
    if locked_at_zero_speed is None:
        locked_at_zero_speed = [False, False, False]
    if motor_ranges is None:
        motor_ranges = [(-1, 1), (-1, 1), (-1, 1)]
    if position_ranges is None:
        position_ranges = [(-1, 1), (-1, 1), (-1, 1)]
    assembly = agxSDK.Assembly()

    position_ranges = {'x': position_ranges[0], 'y': position_ranges[1], 'z': position_ranges[2]}
    motor_ranges = {'x': motor_ranges[0], 'y': motor_ranges[1], 'z': motor_ranges[2]}
    locked_at_zero_speed = {'x': locked_at_zero_speed[0],
                            'y': locked_at_zero_speed[1],
                            'z': locked_at_zero_speed[2]}

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
    joint_base_x.setEnableComputeForces(True)
    joint_base_x.setName(name + "_joint_base_x")
    assembly.add(joint_base_x)

    joint_base_y = agx.Prismatic(agx.Vec3(0, -1, 0), base_x_body, base_y_body)
    joint_base_y.setEnableComputeForces(True)
    joint_base_y.setName(name + "_joint_base_y")
    assembly.add(joint_base_y)

    joint_base_z = agx.Prismatic(agx.Vec3(0, 0, -1), base_y_body, base_z_body)
    joint_base_z.setEnableComputeForces(True)
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

    # Add lock joints between base and rigid body
    lock_joint = agx.LockJoint(rigid_body, base_z_body)
    lock_joint.setEnableComputeForces(True)
    lock_joint.setCompliance(compliance)
    lock_joint.setDamping(damping)
    assembly.add(lock_joint)

    return assembly
