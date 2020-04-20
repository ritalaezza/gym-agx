import agx
import agxIO
import agxSDK
import agxCollide

import os
import math
import logging
import numpy as np
from enum import Enum

logger = logging.getLogger('gym_agx.utils')


class GripperConstraint:
    class Dof(Enum):
        X_TRANSLATIONAL = 0,
        Y_TRANSLATIONAL = 1,
        Z_TRANSLATIONAL = 2,
        X_ROTATIONAL = 3,
        Y_ROTATIONAL = 4,
        Z_ROTATIONAL = 5

    def __init__(self, gripper_dof, compute_forces_enabled, velocity_control, compliance_control, velocity_index,
                 compliance_index):
        """Gripper constraint object, defining important parameters.
        :param gripper_dof: (GripperDof) degree of freedom of gripper that this constraint controls
        :param compute_forces_enabled: (Boolean) force and torque can be measured
        :param velocity_control: (Boolean) is velocity controlled
        :param compliance_control: (Boolean) is compliance controlled
        :param velocity_index: (int) index of action vector which controls velocity of this constraint's motor
        :param compliance_index: (int) index of action vector which controls compliance of this constraint's motor"""
        self.gripper_dof = gripper_dof
        self.velocity_control = velocity_control
        self.compute_forces_enabled = compute_forces_enabled
        self.compliance_control = compliance_control
        self.velocity_index = velocity_index
        self.compliance_index = compliance_index

    @property
    def is_active(self):
        return True if self.velocity_control or self.compliance_control else False


class Gripper:
    last_action_index = -1

    def __init__(self, name, controllable, observable, max_velocity=1, max_acceleration=1, min_compliance=0,
                 max_compliance=1e6):
        """Gripper class which keeps track of gripper constraints and action indices.
        :param name
        :param controllable
        :param observable
        :param max_velocity
        :param max_acceleration
        :param max_compliance
        :return Gripper object"""
        self.name = name
        self.controllable = controllable
        self.observable = observable
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.min_compliance = min_compliance
        self.max_compliance = max_compliance
        self.constraints = {}

    def add_constraint(self, name, gripper_dof, compute_forces_enabled=False, velocity_control=False,
                       compliance_control=False):
        velocity_index = None
        compliance_index = None
        if velocity_control:
            self.last_action_index += 1
            velocity_index = self.last_action_index
        if compliance_control:
            self.last_action_index += 1
            compliance_index = self.last_action_index
        gripper_constraint = GripperConstraint(gripper_dof, compute_forces_enabled, velocity_control,
                                               compliance_control, velocity_index, compliance_index)
        self.constraints.update({name: gripper_constraint})

    def apply_control(self, sim, action, dt):
        if self.controllable:
            for key, constraint in self.constraints.items():
                joint = sim.getConstraint1DOF(key)
                motor = joint.getMotor1D()
                if constraint.velocity_control:
                    gripper_velocity = self.get_gripper_velocity(sim, constraint.gripper_dof)
                    velocity = self.rescale_velocity(action[constraint.velocity_index], gripper_velocity, dt)
                    logger.debug(f'{key} velocity: {velocity}')
                    motor.setSpeed(np.float64(velocity))
                if constraint.compliance_control:
                    motor_param = motor.getRegularizationParameters()
                    compliance = self.rescale_compliance(action[constraint.compliance_index])
                    motor_param.setCompliance(np.float64(compliance))
        else:
            logger.debug("Received apply_control command for uncontrollable gripper.")

    def get_gripper_velocity(self, sim, constraint_dof):
        gripper = sim.getRigidBody(self.name)
        if constraint_dof == GripperConstraint.Dof.X_TRANSLATIONAL:
            gripper_velocity = gripper.getVelocity()[0]
        elif constraint_dof == GripperConstraint.Dof.Y_TRANSLATIONAL:
            gripper_velocity = gripper.getVelocity()[1]
        elif constraint_dof == GripperConstraint.Dof.Z_TRANSLATIONAL:
            gripper_velocity = gripper.getVelocity()[2]
        elif constraint_dof == GripperConstraint.Dof.X_ROTATIONAL:
            gripper_velocity = gripper.getAngularVelocity()[0]
        elif constraint_dof == GripperConstraint.Dof.Y_ROTATIONAL:
            gripper_velocity = gripper.getAngularVelocity()[1]
        elif constraint_dof == GripperConstraint.Dof.Z_ROTATIONAL:
            gripper_velocity = gripper.getAngularVelocity()[2]
        else:
            logger.error("Unexpected GripperDof.")

        assert not math.isnan(gripper_velocity), "NaN found in gripper velocity: %r" % gripper_velocity
        return gripper_velocity

    def get_state(self, sim):
        if self.observable:
            state = []
            for key, constraint in self.constraints.items():
                if constraint.compute_forces_enabled:
                    constraint_state = get_gripper_state(sim, key).ravel()
                    logger.debug(f"{key} state: {constraint_state}")
                    state.append(constraint_state)
            return np.asarray(state)
        else:
            logger.error("Received get_state command for unobservable gripper.")

    def rescale_velocity(self, velocity, gripper_velocity, dt):
        logger.debug(f'Gripper velocity: {gripper_velocity}')
        logger.debug(f'Initial target velocity: {velocity}')
        if abs(velocity - gripper_velocity) > self.max_acceleration:
            velocity = gripper_velocity + np.sign(velocity - gripper_velocity) * (self.max_acceleration * dt)
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


class InfoPrinter(agxSDK.StepEventListener):
    def __init__(self, app, text_table, text_color):
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


def create_info_printer(sim, app, text_table=None, text_color=None):
    """Write information to screen from lambda functions during the simulation.
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table: table with text to be printed on screen
    :param text_color: Color of text
    :return: AGX simulation object
    """
    return sim.add(InfoPrinter(sim, app, text_table, text_color))


def create_help_text(sim, app, text_table=None):
    """Write help text. textTable is a table with strings that will be drawn above the default text.
    :param sim: AGX Simulation object
    :param app: OSG Example Application object
    :param text_table: table with text to be printed on screen
    :return: AGX simulation object
    """
    return sim.add(HelpListener(sim, app, text_table))


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


def create_body(sim, shape, **args):
    """Helper function that creates a RigidBody according to the given definition.
    Returns the body itself, it's geometry and the OSG node that was created for it.
    :param sim: AGX Simulation object
    :param shape: shape of object - agxCollide.Shape.
    :param args: The definition contains the following parts:
    name - string. Optional. Defaults to "". The name of the new body.
    geometryTransform - agx.AffineMatrix4x4. Optional. Defaults to identity transformation. The local transformation of
    the shape relative to the body.
    motionControl - agx.RigidBody.MotionControl. Optional. Defaults to DYNAMICS.
    material - agx.Material. Optional. Ignored if not given. Material assigned to the geometry created for the body.
    :return: body, geometry
    """
    geometry = agxCollide.Geometry(shape)

    if "geometryTransform" not in args.keys():
        geometry_transform = agx.AffineMatrix4x4()
    else:
        geometry_transform = args["geometryTransform"]

    if "name" in args.keys():
        body = agx.RigidBody(args["name"])
    else:
        body = agx.RigidBody("")

    body.add(geometry, geometry_transform)

    if "position" in args.keys():
        body.setPosition(args["position"])

    if "motionControl" in args.keys():
        body.setMotionControl(args["motionControl"])

    if "material" in args.keys():
        geometry.setMaterial(args["material"])

    sim.add(body)

    return body, geometry


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
    :return: NumPy array with segments' position and rotations
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


def get_gripper_state(sim, key, include_position=False, gain=1):
    """Get AGX 'gripper' positions, force and torque.
    :param sim: AGX Dynamics simulation object
    :param key: name of gripper
    :param include_position: Boolean to determine if gripper position is part of state
    :param gain: gives possibility to rescale position values
    :return: NumPy array with gripper position, rotations, force and torque
    """
    gripper = sim.getRigidBody(key)
    if include_position:
        gripper_state = np.zeros(shape=(3, 3))
        gripper_state[:, 0] = to_numpy_array(gripper.getPosition()) * gain
        gripper_state[:, 1], gripper_state[:, 2] = get_force_torque(sim, gripper, key)
    else:
        gripper_state = np.zeros(shape=(3, 2))
        gripper_state[:, 0], gripper_state[:, 1] = get_force_torque(sim, gripper, key)

    return gripper_state


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
