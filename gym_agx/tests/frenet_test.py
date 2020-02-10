import math
import numpy as np
from gym_agx.utils.utils import get_cable_curvature, get_cable_torsion


def parameterized_helix(a=1, b=1, T=2 * math.pi):
    """Circular helix with radius a and slope b/a or pitch 2*pi*b
    :param a: radius
    :param b: slope factor b/a, or pitch factor 2*pi*b
    :param T: final value of t
    :returns state: points along helix
    """
    t_space = np.linspace(0, 2 * math.pi, num=int(T / (math.pi / 180)))
    state = np.zeros((3, len(t_space)))
    for i, t in enumerate(t_space):
        state[0, i] = a * math.cos(t)
        state[1, i] = a * math.sin(t)
        state[2, i] = b * t
    return state


def test_curvature(a_s, b_s, margin):
    results = []
    for a in a_s:
        for b in b_s:
            state = parameterized_helix(a, b)
            segment_length = np.linalg.norm(state[:, 1] - state[:, 0])
            curvature = get_cable_curvature(state, segment_length)
            expected_curvature = abs(a) / (a ** 2 + b ** 2)

            if all(expected_curvature - margin <= c <= expected_curvature + margin for c in curvature):
                results.append(True)
            else:
                results.append(False)

    return results


def test_torsion(a_s, b_s, margin):
    results = []
    for a in a_s:
        for b in b_s:
            state = parameterized_helix(a, b)
            segment_length = np.linalg.norm(state[:, 1] - state[:, 0])
            torsion = get_cable_torsion(state, segment_length)
            expected_torsion = b / (a ** 2 + b ** 2)

            if all(expected_torsion - margin <= t <= expected_torsion + margin for t in torsion):
                results.append(True)
            else:
                results.append(False)

    return results


if __name__ == "__main__":
    margin = 0.01
    a_s = [1, 2, 3]
    b_s = [1, 0.5, 0.25]
    curvature_results = test_curvature(a_s, b_s, margin)
    if all(curvature_results):
        print("Curvature passed.")
    torsion_results = test_torsion(a_s, b_s, margin)
    if all(torsion_results):
        print("Torsion passed.")
