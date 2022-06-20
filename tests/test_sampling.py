import math
import pytest
import numpy as np
from gym_agx.utils.utils import sample_sphere, rotate_rpy

# BendWire values
bend_wire_values = (np.zeros(3),
                    [1, 10],
                    [0, np.pi],
                    [0, np.pi],
                    [-math.pi / 2, 0, 0])
# BendWireObstacle values
bend_wire_obstacle_values_right = (np.zeros(3),
                                   [1, 10],
                                   [np.pi / 2, np.pi],
                                   [math.pi - 0.3374964624625465, math.pi + 0.3374964624625465],
                                   [-math.pi, 0, 0])
bend_wire_obstacle_values_left = (np.zeros(3),
                                  [1, 10],
                                  [np.pi / 2, np.pi],
                                  [math.pi - 0.3374964624625465, math.pi + 0.3374964624625465],
                                  None)

sample_parameters = [bend_wire_values, bend_wire_obstacle_values_right, bend_wire_obstacle_values_left]


@pytest.mark.parametrize("center, radius_range, polar_range, azimuthal_range, rpy_angles", sample_parameters)
def test_sphere_sampling(center, radius_range, polar_range, azimuthal_range, rpy_angles, visualize=False):
    n_samples = 10000
    points = np.zeros([n_samples, 3])  # stored for possible visualization
    for i in range(n_samples):
        point, radius_to_center = sample_sphere(center, radius_range, polar_range, azimuthal_range, rpy_angles)
        points[i, :] = point  # rotated and translated point is used for visualization

        # Reverse rotation and translation
        if rpy_angles:
            point = rotate_rpy(point - center, *rpy_angles, transpose=True)

        # Reconstruct spherical coordinates:
        radius = np.linalg.norm(point)
        polar_angle = math.acos(point[2] / radius)
        azimuth_angle = math.atan(point[1] / point[0])
        if point[0] < 0:  # Left half
            azimuth_angle += math.pi
        else:  # Right half
            if point[1] < 0:  # Fourth quadrant
                azimuth_angle += 2 * math.pi

        # Assertions:
        assert abs(radius - radius_to_center) < 0.0001, "radius should be the same as norm"
        assert radius_range[1] >= radius >= radius_range[0], "radius_range violated"
        assert polar_range[1] >= polar_angle >= polar_range[0], "polar_range violated"
        assert azimuthal_range[1] >= azimuth_angle >= azimuthal_range[0], "azimuthal_range violated"

    if visualize:
        import open3d as o3d
        import matplotlib.pyplot as plt

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True)
        ax1.hist2d(points[:, 0], points[:, 1])
        ax1.set_xlabel('x axis')
        ax1.set_ylabel('y axis')
        ax2.hist2d(points[:, 1], points[:, 2])
        ax2.set_xlabel('y axis')
        ax2.set_ylabel('z axis')
        ax3.hist2d(points[:, 0], points[:, 2])
        ax3.set_xlabel('x axis')
        ax3.set_ylabel('z axis')
        plt.show()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        viewer.add_geometry(pcd)

        ctrl = viewer.get_view_control()
        ctrl.set_front(np.array([0.0, -1.0, 0.0]))
        ctrl.set_lookat(np.array([0.0, 0.0, -1]))
        ctrl.set_up(np.array([0.0, 0.0, 1.0]))
        ctrl.set_zoom(1)

        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.point_color_option = o3d.visualization.PointColorOption.YCoordinate

        viewer.run()
        viewer.destroy_window()

    return


if __name__ == '__main__':
    # This script can also be executed to visualize the sampling
    c = np.zeros(3)
    r_range = [0, 5]
    p_range = [0, math.pi]
    a_range = [0, 2 * math.pi]
    test_sphere_sampling(c, r_range, p_range, a_range, rpy_angles=None, visualize=True)


