import pyximport
pyximport.install()

import numpy as np
# from fastronWrapper.fastronWrapper import PyFastron

from math import pi
import matplotlib.pyplot as plt
from faastron_test import Fastron

# GENERATING DATA -------------------------------------------------------------------------------------------------------------
class NLinkArm(object):
 
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = joint_angles
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + self.link_lengths[i - 1] * np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + self.link_lengths[i - 1] * np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


def detect_collision(line_seg, circle):
    a_vec = np.array([line_seg[0][0], line_seg[0][1]])
    b_vec = np.array([line_seg[1][0], line_seg[1][1]])
    c_vec = np.array([circle[0], circle[1]])
    radius = circle[2]
    line_vec = b_vec - a_vec
    line_mag = np.linalg.norm(line_vec)
    circle_vec = c_vec - a_vec
    proj = circle_vec.dot(line_vec / line_mag)
    if proj <= 0:
        closest_point = a_vec
    elif proj >= line_mag:
        closest_point = b_vec
    else:
        closest_point = a_vec + line_vec * proj / line_mag
    if np.linalg.norm(closest_point - c_vec) > radius:
        return False

    return True


def get_occupancy_grid(arm, obstacles):

    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * pi / M for i in range(-M // 2, M // 2 + 1)]
    # print("Grid:", grid)
    # print(theta_list)
    dataset = []

    for i in range(M):
        for j in range(M):
            arm.update_joints([theta_list[i], theta_list[j]])
            points = arm.points
            collision_detected = False
            for k in range(len(points) - 1):
                for obstacle in obstacles:
                    line_seg = [points[k], points[k + 1]]
                    collision_detected = detect_collision(line_seg, obstacle)
                    if collision_detected:
                        break
                if collision_detected:
                    break
            grid[i][j] = int(collision_detected)

            if int(collision_detected) == 1:
                collision_stat = 1
            elif int(collision_detected) == 0:
                collision_stat = -1
            dataset.append([theta_list[i],theta_list[j],collision_stat])


    return np.array(grid), dataset


# Simulation parameters
M = 150 # number of sample to divide into and number of grid cell
obstacles = [[1.75, 0.75, 0.6], [0.55, 1.5, 0.5], [0, -1, 0.25], [-1.5, -1.5, 0.25]] # x y radius
# obstacles = [[0, -1, 0.25]] # x y radius

arm = NLinkArm([1, 1], [0, 0])
grid,dataset = get_occupancy_grid(arm, obstacles)

dataset = np.array(dataset)

# data = np.delete(dataset, -1, axis=1)
# print(data.shape)

# y = np.delete(dataset, 0, axis=1)
# y = np.delete(y, 0, axis=1)
# print(y.shape)

def data_generation(dataset, alpha = None, gram_computed = None, G = None, F = None):
    # dataset = Collision.collision_check(theta_data)
    dataarray = np.array(dataset)
    data = dataarray[:, 0:2]
    y = dataarray[:, [2]]
    N = data.shape[0]  # number of datapoint = number of row the dataset has
    d = data.shape[1]  # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
    if alpha is None:
        alpha = np.zeros((N, 1))  # weight, init at zero
    if gram_computed is None:
        gram_computed = np.zeros((N, 1))
    if G is None:
        G = np.zeros((N, N))  # kernel gram matrix guassian kernel of dataset
    if F is None:
        F = np.zeros((N, 1))  # hypothesis
    g = 10  # kernel width
    max_updates = 35000  # max update iteration
    max_support_points = 1500  # max support points
    beta = 100
    allowance = 800
    kNS = 4
    sigma = 2
    
    return data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma

data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)
fastron = Fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
data_graph, y_graph = fastron.update_model()
data, alpha, gram_computed, G, F = fastron.active_learning()
# FASTRON -------------------------------------------------------------------------------------------------------------
# Initialize PyFastron
# fastron = PyFastron(data) # where data.shape = (N, d)

# fastron.y = y # where y.shape = (N,)

# fastron.g = 10
# fastron.maxUpdates = 35000
# fastron.maxSupportPoints = 1500
# fastron.beta = 100


# Active Learning
# fastron.activeLearning()

# Update label
# fastron.updateLabels()

# Train model
# fastron.updateModel()

# Predict values for a test set (ask for collision)
# data_test = np.array([[-3.14,0.2]])
# pred = fastron.eval(data_test) # where data_test.shape = (N_test, d)
# print(pred.shape)


def get_occupancy_grid_fastron():
    size = 300
    grid = [[0 for _ in range(size)] for _ in range(size)]
    theta_list = [2 * i * pi / size for i in range(-size // 2, size // 2 + 1)]

    for i in range(size):
        for j in range(size):
            theta = np.array([[theta_list[i],theta_list[j]]])
            collision_detected = fastron.eval(theta)
            # print(collision_detected.shape)
            if collision_detected[0] == 1:
                collision_detected = 1
            elif collision_detected[0] == -1:
                collision_detected = 0
            # if collision_detected[0][0] == 1:
            #     collision_detected = 1
            # elif collision_detected[0][0] == -1:
            #     collision_detected = 0

            grid[i][j] = collision_detected

    return np.array(grid)


# Plot joint config ------------------------------------------------------------------------------------------------------------------------------

grid,_ = get_occupancy_grid(arm, obstacles)
gridft = get_occupancy_grid_fastron()

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# fig.suptitle('Config Space and Workspace Mapping using simple collision detection of <line and circle> only')
# ax1.imshow(grid)

# for obstacle in obstacles:
#         circle = plt.Circle((obstacle[0], obstacle[1]), radius=0.5 * obstacle[2], fc='k')
#         ax2.add_patch(circle)
#         ax2.set_aspect('equal')

#         plt.xlim([-2, 2])
#         plt.ylim([-2, 2])

# ax3.imshow(gridft)

# plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('collision detection of <line and circle>, Fastron')
ax1.imshow(grid)
ax2.imshow(gridft)

plt.show()


# save config space from fastron
