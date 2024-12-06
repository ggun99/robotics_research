from faastron_test import Fastron
from collisioncheck2 import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# import pandas as pd
from fastronWrapper.fastronWrapper import PyFastron

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



def fastron(data, alpha, gram_computed, G, F):
    data_graph, y_graph = fastron_.update_model()
    data, alpha, gram_computed, G, F = fastron_.active_learning()
    return data_graph, y_graph, data, alpha, gram_computed, G, F

def homogen(alpha, a, d, theta):
    mat = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
           [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
           [0, np.sin(alpha), np.cos(alpha), d],
           [0,0,0,1]])
    return mat
# data = None
# alpha, gram_computed, G, F = None
L1 = 3
L2 = 6
L3 = 6
L_r = 1
# data = []
Collision = ColCheck(L1,L2,L3,L_r)
theta_data = Collision.theta_list()

# data_ = theta_data[:,0:3]
# y__ = theta_data[:,[3]]
dataset, indi  = Collision.collision_check(theta_data,0,0,6,3,0)
data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)
fastron_ = Fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)

# fastron__ = PyFastron(data)
# fastron__.y = y
# fastron__.g = 10
# fastron__.maxUpdates = 2000000
# fastron__.maxSupportPoints = 15000
# fastron__.beta = 200


# fastron__.activeLearning()
# fastron__.updateModel()
# X_ = np.array(theta_data)
# y_ = np.array(y)
# print(y_.shape, X_.shape)
# print(np.unique(y_))
data_graph, y_graph, data__, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F)
# X_ = np.array(data_graph)
# y_ = np.array(y_graph)
X = np.array(data)
y = np.array(y)

# # SVM 모델을 RBF 커널로 훈련시킵니다
# # clf = svm.SVC(kernel='rbf')
# # clf.fit(X, y)
# # print(y_.shape, X_.shape)
# print('1',data.shape)
# print(np.unique(y_))
# fig1 = plt.figure()
# ax1 = fig1.add_subplot(111, projection='3d')
# for i in range(X_.shape[0]):
    
#     if y_[i] == -1:
#         color = 'b'
#     else:
#         color = 'r'
#     ax1.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)

# # 축 레이블을 설정합니다
# ax1.set_xlabel('q1')
# ax1.set_ylabel('q2')
# ax1.set_zlabel('q3')
# ax1.axis('auto')
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# 3D 플롯을 생성합니다
print('start eval')
for i in range(data.shape[0]):
    y_pred = fastron_.eval(data)
    if y_pred[i] == -1:
        color = 'b'
    else:
        color = 'r'
    ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
print('finish eval')
# for i in range(data.shape[0]):

    
# 데이터 포인트를 플롯합니다


# 축 레이블을 설정합니다
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('q3')
ax.axis('auto')
plt.show()

# for i in range(4):
#     print('1',data.shape)
#     Collision1 = ColCheck(L1,L2,L3,L_r)
#     dataset = Collision1.collision_check(data,6-i*0.5+0.5,6,6,3)
#     data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset, alpha, gram_computed, G, F)
#     data_graph, y_graph, data, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
#     # data_excel = pd.DataFrame(data)
#     # data_excel.to_excel("ooo.xlsx")
#     X = np.array(data_graph)
#     y = np.array(y_graph)

#     # SVM 모델을 RBF 커널로 훈련시킵니다
#     clf = svm.SVC(kernel='rbf')
#     clf.fit(X, y)
#     print('2',data.shape)
#     # 3D 플롯을 생성합니다
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # 데이터 포인트를 플롯합니다
#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)

#     # 축 레이블을 설정합니다
#     ax.set_xlabel('q1')
#     ax.set_ylabel('q2')
#     ax.set_zlabel('q3')

#     plt.show()