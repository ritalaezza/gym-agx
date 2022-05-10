import agx
import agxCable
import logging
import numpy as np
from enum import Enum

from agxPythonModules.utils.numpy_utils import create_numpy_array
from gym_agx.utils.agx_utils import to_numpy_array, get_cable_segment_edges
from gym_agx.utils.utils import get_cable_torsion, get_cable_curvature, get_cable_angles

logger = logging.getLogger('gym_agx.rl')


class ObservationType(Enum):
    DLO_POSITIONS = "dlo_positions"
    DLO_ROTATIONS = "dlo_rotations"
    DLO_ANGLES = "dlo_angles"
    DLO_CURVATURE = "dlo_curvature"
    DLO_TORSION = "dlo_torsion"
    IMG_RGB = "img_rgb"
    IMG_DEPTH = "img_depth"
    EE_FORCE_TORQUE = "ee_force_torque"
    EE_POSITION = "ee_position"
    EE_ROTATION = "ee_rotation"
    EE_VELOCITY = "ee_velocity"
    EE_ANGULAR_VELOCITY = "ee_angular_velocity"
    

class ObservationConfig:

    def __init__(self, goals, observations=None):
        """Initialize observation configuration object with list of observations and goal type
        :param list goals: list of ObservationType which will be used to compute reward, based on achieved
        goals and desired goals
        :param list observations: list of ObservationType values that will be used as input to agent. Can be given as
        input, or constructed by class methods
        """
        self.goals = set(dict.fromkeys(goals))
        if observations is None:
            self.observations = set()
        else:
            self.observations = set(dict.fromkeys(observations))

        self.rgb_in_obs = False
        self.depth_in_obs = False
        self.image_size = (256, 256)  # (default) all image data will have same first two dimensions.
        if ObservationType.IMG_RGB in (self.observations | self.goals):
            self.rgb_in_obs = True
        if ObservationType.IMG_DEPTH in (self.observations | self.goals):
            self.depth_in_obs = True

    def get_observations(self, sim, rti, end_effectors, cable=None, goal_only=False):
        """Main function which gets observations, based on configuration. To avoid repeated calls to same observation,
        goals can be obtained at the same time, by taking the union of the two sets
        :param agx.Simulation sim: AGX Dynamics simulation object
        :param list rti: agxOSG.RenderToImage buffers to render image observations
        :param EndEffector end_effectors: List of EndEffector objects which are required to obtain observations of the
        end-effectors in the simulation
        :param agx.Cable cable: If the simulation contains an AGX Cable structure, there are special functions to obtain
        its state
        :param goal_only: If set to True, only goals will be retrieved.
        :return: Dictionaries with observations and achieved goals, or just desired goals.
        """
        observation_set = self.goals
        if not goal_only:
            goal_string = ""
            observation_set = observation_set.union(self.observations)
        else:
            goal_string = "_goal"

        if cable is not None:
            cable_object = agxCable.Cable.find(sim, cable + goal_string)
            cable_segment_edges = get_cable_segment_edges(cable_object)
        else:
            raise NotImplementedError("Observations of non Cable objects are not available yet.")

        rgb_buffer = None
        depth_buffer = None
        for buffer in rti:
            name = buffer.getName()
            if name == 'rgb_buffer':
                rgb_buffer = buffer
            elif name == 'depth_buffer':
                depth_buffer = buffer

        goal_obs = dict()
        for gobs in observation_set:
            if gobs == ObservationType.DLO_POSITIONS:
                goal_obs[gobs.value] = get_cable_segment_positions(cable_object)
            elif gobs == ObservationType.DLO_ROTATIONS:
                goal_obs[gobs.value] = get_cable_segment_rotations(cable_object)
            elif gobs == ObservationType.DLO_ANGLES:
                goal_obs[gobs.value] = get_cable_angles(cable_segment_edges)
            elif gobs == ObservationType.DLO_CURVATURE:
                goal_obs[gobs.value] = get_cable_curvature(cable_segment_edges)
            elif gobs == ObservationType.DLO_TORSION:
                goal_obs[gobs.value] = get_cable_torsion(cable_segment_edges)
            elif gobs == ObservationType.IMG_RGB:
                if rgb_buffer:
                    image_ptr = rgb_buffer.getImageData()
                    image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1], 3), np.uint8)
                    goal_obs[gobs.value] = np.flipud(image_data)
                else:
                    goal_obs[gobs.value] = np.zeros(shape=(self.image_size[0], self.image_size[1], 3))
            elif gobs == ObservationType.IMG_DEPTH:
                if depth_buffer:
                    image_ptr = depth_buffer.getImageData()
                    image_data = create_numpy_array(image_ptr, (self.image_size[0], self.image_size[1]), np.float32)
                    goal_obs[gobs.value] = np.flipud(image_data)
                else:
                    goal_obs[gobs.value] = np.zeros(shape=self.image_size)
            elif gobs == ObservationType.EE_FORCE_TORQUE:
                ee_force_torque = dict()
                for ee in end_effectors:
                    if ee.observable:
                        force_torque = dict()
                        for key, constraint in ee.constraints.items():
                            if constraint.compute_forces_enabled:
                                force_torque[key] = get_constraint_force_torque(sim, ee.name + goal_string,
                                                                                key + goal_string)
                        assert force_torque, "At least one constraint must have compute_forces_enabled set to True."
                        ee_force_torque[ee.name] = force_torque
                assert ee_force_torque, "At least one of the end-effectors must be observable to obtain force-torque."
                goal_obs[gobs.value] = ee_force_torque
            elif gobs == ObservationType.EE_VELOCITY:
                ee_velocity = dict()
                for ee in end_effectors:
                    if ee.observable:
                        ee_velocity[ee.name] = get_rigid_body_velocity(sim, ee.name + goal_string)
                assert ee_velocity, "At least one of the end-effectors must be observable to obtain velocity."
                goal_obs[gobs.value] = ee_velocity
            elif gobs == ObservationType.EE_ANGULAR_VELOCITY:
                ee_angular_velocity = dict()
                for ee in end_effectors:
                    if ee.observable:
                        ee_angular_velocity[ee.name] = get_rigid_body_angular_velocity(sim, ee.name + goal_string)
                assert ee_angular_velocity, "At least one of the end-effectors must be observable to obtain angular " \
                                            "velocity."
                goal_obs[gobs.value] = ee_angular_velocity
            elif gobs == ObservationType.EE_POSITION:
                ee_position = dict()
                for ee in end_effectors:
                    if ee.observable:
                        ee_position[ee.name] = get_rigid_body_position(sim, ee.name + goal_string)
                assert ee_position, "At least one of the end-effectors must be observable to obtain position."
                goal_obs[gobs.value] = ee_position
            elif gobs == ObservationType.EE_ROTATION:
                ee_rotation = dict()
                for ee in end_effectors:
                    if ee.observable:
                        ee_rotation[ee.name] = get_rigid_body_rotation(sim, ee.name + goal_string)
                assert ee_rotation, "At least one of the end-effectors must be observable to obtain rotation."
                goal_obs[gobs.value] = ee_rotation

        achieved_goal = dict()
        for goal in self.goals:
            achieved_goal[goal.value] = goal_obs[goal.value]

        if goal_only:
            return achieved_goal

        observation = dict()
        for obs in self.observations:
            observation[obs.value] = goal_obs[obs.value]

        return observation, achieved_goal

    def set_dlo_positions(self):
        """3D coordinates of DLO segments"""
        self.observations.add(ObservationType.DLO_POSITIONS)

    def set_dlo_rotations(self):
        """Quaternions of DLO segments"""
        self.observations.add(ObservationType.DLO_ROTATIONS)

    def set_dlo_poses(self):
        """3D coordinates and quaternions of DLO segments"""
        self.observations.add(ObservationType.DLO_POSITIONS)
        self.observations.add(ObservationType.DLO_ROTATIONS)

    def set_dlo_angles(self):
        """Inner angles of DLO segments"""
        self.observations.add(ObservationType.DLO_ANGLES)

    def set_img_rgb(self, image_size=None):
        """RGB image of scene containing DLO and end-effector(s)
        :param tuple image_size: tuple with dimensions of image
        """
        self.observations.add(ObservationType.IMG_RGB)
        self.rgb_in_obs = True
        if image_size:
            self.image_size = image_size

    def set_img_depth(self, image_size=None):
        """Depth image of scene containing DLO and end-effector(s)
        :param tuple image_size: tuple with dimensions of image
        """
        self.observations.add(ObservationType.IMG_DEPTH)
        self.depth_in_obs = True
        if image_size:
            self.image_size = image_size

    def set_dlo_frenet_curvature(self):
        """Discrete Frenet curvature of DLO"""
        self.observations.add(ObservationType.DLO_CURVATURE)

    def set_dlo_frenet_torsion(self):
        """Discrete Frenet torsion of DLO"""
        self.observations.add(ObservationType.DLO_TORSION)

    def set_dlo_frenet_values(self):
        """Discrete Frenet curvature and torsion of DLO"""
        self.observations.add(ObservationType.DLO_CURVATURE)
        self.observations.add(ObservationType.DLO_TORSION)

    def set_ee_position(self):
        """3D coordinates of edd-effector(s)"""
        self.observations.add(ObservationType.EE_POSITION)

    def set_ee_rotation(self):
        """Quaternions of edd-effector(s)"""
        self.observations.add(ObservationType.EE_ROTATION)

    def set_ee_velocity(self):
        """Linear velocity of edd-effector(s)"""
        self.observations.add(ObservationType.EE_VELOCITY)

    def set_ee_angular_velocity(self):
        """Angular velocity of edd-effector(s)"""
        self.observations.add(ObservationType.EE_ANGULAR_VELOCITY)

    def set_ee_pose(self):
        """3D coordinates and quaternions of edd-effector(s)"""
        self.observations.add(ObservationType.EE_POSITION)
        self.observations.add(ObservationType.EE_ROTATION)

    def set_ee_force_torque(self):
        """Forces and torques sensed by edd-effector(s)"""
        self.observations.add(ObservationType.EE_FORCE_TORQUE)

    def set_all_ee(self):
        """Pose, velocities and force-torques sensed by edd-effector(s)"""
        self.observations.add(ObservationType.EE_POSITION)
        self.observations.add(ObservationType.EE_ROTATION)
        self.observations.add(ObservationType.EE_VELOCITY)
        self.observations.add(ObservationType.EE_ANGULAR_VELOCITY)
        self.observations.add(ObservationType.EE_FORCE_TORQUE)


def get_cable_segment_rotations(cable):
    """Get AGX Cable segments' center of mass rotations
    :param cable: AGX Cable object
    :return: NumPy array with segments' rotations
    """
    num_segments = cable.getNumSegments()
    cable_segments_rotations = np.zeros(shape=(4, num_segments))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            rotation = segment_iterator.getGeometry().getRotation()
            cable_segments_rotations[:, i] = to_numpy_array(rotation)
            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_segments_rotations


def get_cable_segment_positions(cable):
    """Get AGX Cable segments' center of mass positions
    :param cable: AGX Cable object
    :return: NumPy array with segments' positions
    """
    num_segments = cable.getNumSegments()
    cable_positions = np.zeros(shape=(3, num_segments))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            position = segment_iterator.getRigidBody().getPosition()
            cable_positions[:, i] = to_numpy_array(position)
            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_positions


def get_cable_segment_positions_and_velocities(cable):
    """Get AGX Cable segments' center of mass positions
    :param cable: AGX Cable object
    :return: NumPy array with segments' positions
    """
    num_segments = cable.getNumSegments()
    cable_positions = np.zeros(shape=(3, num_segments))
    cable_velocities = np.zeros(shape=(3, num_segments))
    segment_iterator = cable.begin()
    for i in range(num_segments):
        if not segment_iterator.isEnd():
            position = segment_iterator.getRigidBody().getPosition()
            velocity = segment_iterator.getRigidBody().getVelocity()
            cable_positions[:, i] = to_numpy_array(position)
            cable_velocities[:, i] = to_numpy_array(velocity)
            segment_iterator.inc()
        else:
            logger.error('AGX segment iteration finished early. Number or cable segments may be wrong.')

    return cable_positions, cable_velocities


def get_ring_segment_positions(sim, ring_name, num_segments=None):
    """Get ring segments positions
    :param sim: AGX Dynamics simulation object
    :param ring_name: name of ring object
    :param num_segments: number of segments making up the ring (possibly saves search time)
    :return: NumPy array with segments' positions
    """
    if not num_segments:
        rbs = sim.getRigidBodies()
        ring_segments_positions = np.zeros(shape=(3,))
        for rb in rbs:
            if ring_name in rb.getName():
                position = to_numpy_array(rb.getPosition())
                ring_segments_positions = np.stack([ring_segments_positions, position], axis=0)
    else:
        ring_segments_positions = np.zeros(shape=(3, num_segments + 1))
        for i in range(1, num_segments + 1):
            rb = sim.getRigidBody(ring_name + "_" + str(i))
            ring_segments_positions[:, i - 1] = to_numpy_array(rb.getPosition())

    return ring_segments_positions


def get_rigid_body_position(sim, key):
    """Get position of AGX rigid body
    :param sim: AGX Dynamics simulation object
    :param key: name of rigid body
    :return: NumPy array with rigid body position
    """
    rigid_body = sim.getRigidBody(key)
    position = to_numpy_array(rigid_body.getPosition())
    logger.debug("Rigid body {}, position: {}".format(key, position))

    return position


def get_rigid_body_rotation(sim, name):
    """Get rotation of AGX rigid body
    :param sim: AGX Dynamics simulation object
    :param name: name of rigid body
    :return: NumPy array with rigid body rotation
    """
    rigid_body = sim.getRigidBody(name)
    rotation = to_numpy_array(rigid_body.getRotation())
    logger.debug("Rigid body {}, rotation: {}".format(name, rotation))

    return rotation


def get_rigid_body_velocity(sim, name):
    """Get velocity of AGX rigid body
    :param sim: AGX Dynamics simulation object
    :param name: name of rigid body
    :return: NumPy array with rigid body rotation
    """
    rigid_body = sim.getRigidBody(name)
    velocity = to_numpy_array(rigid_body.getVelocity())
    logger.debug("Rigid body {}, velocity: {}".format(name, velocity))

    return velocity


def get_rigid_body_angular_velocity(sim, name):
    """Get rotation of AGX rigid body
    :param sim: AGX Dynamics simulation object
    :param name: name of rigid body
    :return: NumPy array with rigid body rotation
    """
    rigid_body = sim.getRigidBody(name)
    angular_velocity = to_numpy_array(rigid_body.getAngularVelocity())
    logger.debug("Rigid body {}, angular_velocity: {}".format(name, angular_velocity))

    return angular_velocity


def get_constraint_force_torque(sim, name, constraint_name):
    """Gets force a torque on rigid object, computed by a constraint defined by 'constraint_name'
    :param sim: AGX Simulation object
    :param name: name of rigid body
    :param constraint_name: Name indicating which constraint contains force torque information for this object
    :return: force a torque
    """
    rigid_body = sim.getRigidBody(name)
    constraint = sim.getConstraint(constraint_name)

    force = agx.Vec3()
    torque = agx.Vec3()
    constraint.getLastForce(rigid_body, force, torque)

    force = to_numpy_array(force)
    torque = to_numpy_array(torque)

    logger.debug("Constraint {}, force: {}, torque: {}".format(constraint_name, force, torque))

    return np.concatenate((force, torque))
