""" Generate dataset for collision model training for Planar Robot
- X dataset (nx2) : (theta1, theta2)
- Y dataset (nx2) : gradient 1 to 0
"""

import os
import sys

wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
import matplotlib.pyplot as plt
import numpy as np
from robot.planar_rr import PlanarRR
from collision_check_geometry import collision_class


def collision_dataset(robot, obs_list):
    # start sample
    sample_size = 360
    theta_candidate = np.linspace(-np.pi, np.pi, sample_size)

    sample_theta = []  # X
    sample_collision = []  # y

    for th1 in range(sample_size):
        for th2 in range(sample_size):
            theta = np.array([[theta_candidate[th1]], [theta_candidate[th2]]])
            link_pose = robot.forward_kinematic(theta, return_link_pos=True)
            linearm1 = collision_class.ObjLine2D(link_pose[0][0], link_pose[0][1], link_pose[1][0], link_pose[1][1])
            linearm2 = collision_class.ObjLine2D(link_pose[1][0], link_pose[1][1], link_pose[2][0], link_pose[2][1])

            col = []
            for i in obs_list:
                col1 = collision_class.intersect_line_v_rectangle(linearm1, i)
                col2 = collision_class.intersect_line_v_rectangle(linearm2, i)
                col.extend((col1, col2))

            if True in col:
                sample_collision.append(1.0)
            else:
                sample_collision.append(0.0)

            sample_theta_row = [theta_candidate[th1], theta_candidate[th2]]
            sample_theta.append(sample_theta_row)

    return np.array(sample_theta), np.array(sample_collision)


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_1
    from robot.planar_rr import PlanarRR

    robot = PlanarRR()
    obs_list = task_rectangle_obs_1()

    X, y = collision_dataset(robot, obs_list)

    print("==>> sample_theta.shape: \n", X.shape)
    print("==>> sample_endeffector_pose.shape: \n", y.shape)

    plt.imshow(y.reshape(360, 360))
    plt.show()