import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
from collision_check_learnbased.planar_rr_collision_dataset import collision_dataset



class CollsionVectorDatabase:
    def __init__(self):
        self.config = []
        self.collisionState = []

    def add(self):
        pass

    def query(self):
        pass


if __name__ == "__main__":
    from map.taskmap_geo_format import task_rectangle_obs_1
    from robot.planar_rr import PlanarRR
    import matplotlib.pyplot as plt

    robot = PlanarRR()
    obs_list = task_rectangle_obs_1()

    X, y = collision_dataset(robot, obs_list)

    print("==>> sample_theta.shape: \n", X.shape)
    print("==>> sample_endeffector_pose.shape: \n", y.shape)


    vector_db = CollsionVectorDatabase()
