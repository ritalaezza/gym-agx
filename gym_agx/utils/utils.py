import math
import logging
import numpy as np
from gym import spaces

logger = logging.getLogger('gym_agx.utils')


def construct_space(observation, inc=0):
    """General purpose function to construct OpenAI Gym spaces from a sampled observation.
    :param observation: sampled observation, made up of nested dictionaries which have NumPy arrays at leafs.
    :param inc: helps determine number of recursive calls (should not be manually set).
    :return: OpenAI Gym space
    """
    space_dict = dict()
    for key, value in observation.items():
        if type(value) is np.ndarray:
            space_dict[key] = spaces.Box(low=-np.inf, high=np.inf, shape=value.shape, dtype='float32')
        elif type(value) is dict:
            space_dict[key] = construct_space(value, inc + 1)
        else:
            raise Exception("Unexpected type: {}, in recursion level {}.".format(type(value), inc))

    assert space_dict, "Observation dictionaries must have at least one item."
    return spaces.Dict(space_dict)


def goal_distance(achieved_goal, desired_goal, norm="l2"):
    """Computes distance between achieved goal and desired goal.
    :param achieved_goal: vector of achieved goal
    :param desired_goal: vector of desired goal
    :param norm: type of norm to be computed
    :return:
    """
    assert achieved_goal.shape == desired_goal.shape
    if norm == "l1":
        return np.sum(abs(achieved_goal - desired_goal))
    elif norm == "l2":
        return np.linalg.norm(achieved_goal - desired_goal)
    elif norm == 'inf':
        return np.linalg.norm(achieved_goal - desired_goal, np.inf)
    elif norm == '-inf':
        return np.linalg.norm(achieved_goal - desired_goal, -np.inf)
    else:
        logger.error("Unexpected norm. Choose between: l1, l2, inf and -inf")


def goal_area(achieved_goal, goal):
    """Computes area between desired goal and achieved goal.
    :param achieved_goal: vector of achieved goal
    :param goal: vector of desired goal
    :return:
    """
    assert achieved_goal.shape == goal.shape
    return abs(np.trapz(achieved_goal) - np.trapz(goal))


def sinusoidal_trajectory(A, w, t, phase=0):
    """Assuming a position trajectory of the type: x(t) = A cos(w*t) , the velocity trajectory becomes:
    x'(t) = - A*w sin(w*t)
    :param A: Amplitude in meters
    :param w: frequency in radians per second
    :param t: current timestamp in seconds
    :param phase: phase shift of sinusoid
    :return: instant velocity, x'
    """
    return -A * w * math.sin(w * t + phase)


def find_reference_angle(angle):
    """Finds reference angle in first quadrant.
    :param angle: angle in radians
    :return: reference angle and sign"""
    while angle > 2 * math.pi:
        angle -= 2 * math.pi

    # Determine quadrant:
    if angle < math.pi / 2:
        reference_angle = angle
        sign = 1
    elif angle < math.pi:
        reference_angle = math.pi - angle
        sign = -1
    elif angle < 3 * math.pi / 2:
        reference_angle = angle - math.pi
        sign = 1
    else:
        reference_angle = 2 * math.pi - angle
        sign = -1

    return reference_angle, sign


def compute_linear_distance(v0, v1):
    """Computes linear distance between two points.
    :param v0: NumPy array
    :param v1: NumPy array
    :return: Euclidean distance between v0 and v1.
    """
    return math.sqrt(((v0 - v1) ** 2).sum())


def compute_curvature(v0, v1, segment_length=1):
    """Computes curvature between two segments (through circumscribed osculating circle).
    :param v0: NumPy array
    :param v1: NumPy array
    :param segment_length: length of AGX Cable segment (default 1)
    :return: a positive scalar corresponding to the curvature: K = 2*tan(tangent_angle/2) / segment_length
    """
    length_v0 = np.linalg.norm(v0)
    length_v1 = np.linalg.norm(v1)
    cos_angle = np.dot(v0 / length_v0, v1 / length_v1)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = math.acos(cos_angle)
    ref_angle, sign = find_reference_angle(angle)
    return 2 * sign * np.tan(ref_angle / 2) / segment_length


def get_cable_curvature(cable_segment_edges, segment_length=1):
    """Iterates through cable state to compute curvature between three adjacent points.
    :param cable_segment_edges: Numpy array with coordinates of cable segments
    :param segment_length: length of AGX Cable segment (default 1)
    """
    cable_vectors = np.diff(cable_segment_edges)
    curvature = np.zeros(shape=cable_segment_edges.shape[1] - 2)
    for i in range(cable_segment_edges.shape[1] - 2):
        curvature[i] = compute_curvature(cable_vectors[:, i], cable_vectors[:, i + 1], segment_length)

    return curvature


def compute_torsion(v0, v1, v2, segment_length=1):
    """Computes torsion between two segments (through circumscribed osculating circle).
    :param v0: NumPy array
    :param v1: NumPy array
    :param v2: NumPy array
    :param segment_length: length of AGX Cable segment (default 1)
    :return: a positive scalar corresponding to the curvature: T = 2*tan(binormal_angle/2) / segment_length
    """
    # Binormal vectors
    length_v0 = np.linalg.norm(v0)
    length_v1 = np.linalg.norm(v1)
    length_v2 = np.linalg.norm(v2)
    b01 = np.cross(v0 / length_v0, v1 / length_v1)
    b12 = np.cross(v1 / length_v1, v2 / length_v2)
    length_b01 = np.linalg.norm(b01)
    length_b12 = np.linalg.norm(b12)
    # Torsion angle
    angle = np.arccos(np.dot(b01 / length_b01, b12 / length_b12))
    ref_angle, sign = find_reference_angle(angle)
    return 2 * sign * np.tan(ref_angle / 2) / segment_length


def get_cable_torsion(cable_state, segment_length=1):
    """Iterates through cable state to compute torsion between four adjacent points.
    :param cable_state: Numpy array with coordinates of cable segments
    :param segment_length: length of AGX Cable segment (default 1)
    """
    cable_vectors = np.diff(cable_state)
    torsion = np.zeros(shape=cable_state.shape[1] - 3)
    for i in range(cable_state.shape[1] - 3):
        torsion[i] = compute_torsion(cable_vectors[:, i], cable_vectors[:, i + 1], cable_vectors[:, i + 2],
                                     segment_length)

    return torsion
