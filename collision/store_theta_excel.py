from collisioncheck2 import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from fastronWrapper.fastronWrapper import PyFastron
import pandas as pd

def data_generation(dataset, alpha = None, gram_computed = None, G = None, F = None):
    # dataset = Collision.collision_check(theta_data)
    dataarray = np.array(dataset, dtype=float)
    data = dataarray[:, 0:3]
    y = dataarray[:, [3]]
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
    max_updates = 10000  # max update iteration
    max_support_points = 800  # max support points
    beta = 100
    allowance = 800
    kNS = 4
    sigma = 2
    
    return data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma

L1 = 3
L2 = 6
L3 = 6
L_r = 0.1


Collision = ColCheck(L1,L2,L3,L_r)
print('thetalist')
theta_data = Collision.theta_list()
base_x = 0
print('colcheck')
dataset, indi, coll  = Collision.collision_check(theta_data,0,2,6,3,base_x)
# data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)
print('excel')
data_excel = pd.DataFrame(np.array(coll))
data_excel.to_excel("x0_coll_dataset.xlsx")

# fastron__ = PyFastron(data)
# fastron__.y = y
# fastron__.g = 10
# fastron__.maxUpdates = 50000
# fastron__.maxSupportPoints = 15000
# fastron__.beta = 200


# fastron__.activeLearning()
# fastron__.updateModel()

# X = np.array(data)


# y_pred = fastron__.eval(data)
# y_collision = 
# print(f"==>> y_pred.shape: {y_pred.shape}")
# # y_pred 
# z_values = X[:, 2]
# range1 = np.where(z_values <= 30)
# range2 = np.where((30 < z_values) & (z_values <= 60))
# range3 = np.where(60 < z_values)
# # range1_ = X[range1]
# # range2_ = X[range2]
# # range3_ = X[range3]
# range1_X = X[:,:2][range1]
# range2_X = X[:,:2][range2]
# range3_X = X[:,:2][range3]
# range1_y = y_pred[:,0][range1]
# range2_y = y_pred[:,0][range2]
# range3_y = y_pred[:,0][range3]
# print(range1_y)
# print(range1_X.shape)
