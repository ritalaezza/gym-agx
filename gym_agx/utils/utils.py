import math
import logging
import numpy as np
from gym import spaces

logger = logging.getLogger('gym_agx.utils')


def construct_space(observation, inc=0):
    """General purpose function to construct OpenAI Gym spaces from a sampled observation.
    :param observation: sampled observation, made up of nested dictionaries which have NumPy arrays at leaves.
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


def harmonic_trajectory(A, w, t, phase=0):
    """Assuming a position trajectory of the type: x(t) = A cos(w*t) , the velocity trajectory becomes:
    x'(t) = - A*w sin(w*t)
    :param A: Amplitude in meters
    :param w: frequency in radians per second
    :param t: current timestamp in seconds
    :param phase: phase shift of sinusoid
    :return: instant velocity, x'
    """
    return -A * w * math.sin(w * t + phase)


def point_to_point_trajectory(current_time, start_time, time_limit, start_position, end_position, degree=3):
    """Assuming a third order polynomial trajectory: x(s) = x_start + s(x_end - x_start) , s in [0, 1]
    with time scaling: s(t) = a_0 + a_1 t + a_2 t^2 + a_3 t^3 , t in [0, T].
    :param current_time: time t, for instant velocity
    :param start_time: time instant to consider as start of trajectory
    :param time_limit: time limit, independent from start_time. duration of trajectory
    :param start_position: position at start_time
    :param end_position: position at start_time + time_limit
    :param degree: (optional) degree of polynomial (3 or 5)
    :return: instant velocity
    """
    if degree == 3:
        a_2 = 3 / (time_limit ** 2)
        a_3 = - 2 / (time_limit ** 3)
        a_4 = 0
        a_5 = 0
    elif degree == 5:
        a_2 = 0
        a_3 = 10 / (time_limit ** 3)
        a_4 = - 15 / (time_limit ** 4)
        a_5 = 6 / (time_limit ** 5)
    else:
        raise Exception("degree must be 3 or 5")
    t = current_time - start_time
    if t <= time_limit:
        s_dot = 2 * a_2 * t + 3 * a_3 * t ** 2 + 4 * a_4 * t ** 3 + 5 * a_5 * t ** 4
        x_dot = (end_position - start_position) * s_dot
    else:
        x_dot = np.zeros_like(start_position)
    return x_dot


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


def compute_angle(v0, v1):
    """Computes angle between two segments (through circumscribed osculating circle).
    :param v0: NumPy array
    :param v1: NumPy array
    :return: angle in radians
    """
    length_v0 = np.linalg.norm(v0)
    length_v1 = np.linalg.norm(v1)
    cos_angle = np.dot(v0 / length_v0, v1 / length_v1)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = math.acos(cos_angle)
    return angle


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


def get_cable_angles(cable_segment_edges):
    """Iterates through cable state to compute angle between three adjacent points.
    :param cable_segment_edges: Numpy array with coordinates of cable segments
    """
    cable_vectors = np.diff(cable_segment_edges)
    angles = np.zeros(shape=cable_segment_edges.shape[1] - 2)
    for i in range(cable_segment_edges.shape[1] - 2):
        angles[i] = compute_angle(cable_vectors[:, i], cable_vectors[:, i + 1])

    return angles


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


def point_inside_polygon(polygon, point):
    """
    Point in polygon algorithm (Jordan theorem).
    :param polygon:
    :param point:
    :return:
    """
    x = point[0]
    y = point[1]
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def all_points_below_z(points, max_z):
    """
    Test if all segments are below a certain height.
    :param points:
    :param max_z:
    :return:
    """
    for p in points:
        if p[2] > max_z:
            return False
    return True
