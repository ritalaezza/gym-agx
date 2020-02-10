import agx
import agxIO
import agxSDK
import agxCollide

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


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


def get_gripper_state(sim, grippers, gain=1):
    """Get AGX 'gripper' positions, rotations, force and torque.
    :param sim: AGX Dynamics simulation object
    :param grippers: List AGX kinematic object(s)
    :param gain: gives possibility to rescale position values
    :return: NumPy array with gripper position, rotations, force and torque
    """
    gripper_state = np.zeros(shape=(14, len(grippers)))
    for i, key in enumerate(grippers):
        gripper = sim.getRigidBody(key)
        gripper_state[:3, i] = to_numpy_array(gripper.getPosition()) * gain
        gripper_state[3:7, i] = to_numpy_array(gripper.getRotation())
        gripper_state[7:10, i], gripper_state[10:13, i] = get_force_torque(sim, gripper, key + '_constraint')
        gripper_state[13, i] = 0  # For now just a filler, but could contain other information

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
