import math
import numpy as np


def sinusoidal_trajectory(A, w, t):
    """Assuming a position trajectory of the type: x(t) = A cos(w*t) , the velocity trajectory becomes:
    x'(t) = - A*w sin(w*t)
    :param A: Amplitude in meters
    :param w: frequency in radians per second
    :param t: current timestamp in seconds
    :return: instant velocity, x'
    """
    return -A * w * math.sin(w * t)


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
    :param v0: Numpy array
    :param v1: Numpy array
    :return: Euclidean distance between v0 and v1.
    """
    return math.sqrt(((v0 - v1) ** 2).sum())


def compute_curvature(v0, v1, segment_length=1):
    """Computes curvature between two segments (through circumscribed osculating circle).
    :param v0: Numpy array
    :param v1: Numpy array
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


def compute_torsion(v0, v1, v2, segment_length=1):
    """Computes torsion between two segments (through circumscribed osculating circle).
    :param v0: Numpy array
    :param v1: Numpy array
    :param v2: Numpy array
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


def get_cable_curvature(cable_state, segment_length=1):
    """Iterates through cable state to compute curvature between three adjacent points.
    :param cable_state: Numpy array with coordinates of cable segments
    :param segment_length: length of AGX Cable segment (default 1)
    """
    cable_vectors = np.diff(cable_state)
    curvature = np.zeros(shape=cable_state.shape[1] - 2)
    for i in range(cable_state.shape[1] - 2):
        curvature[i] = compute_curvature(cable_vectors[:, i], cable_vectors[:, i + 1], segment_length)

    return curvature


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
