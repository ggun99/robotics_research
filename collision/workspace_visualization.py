# from faastron_test import Fastron
from collisioncheck2 import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
# from sklearn import svm
import pandas as pd
from fastronWrapper.fastronWrapper import PyFastron


def data_generation(dataset, alpha = None, gram_computed = None, G = None, F = None):
    # dataset = Collision.collision_check(theta_data)
    dataarray = np.array(dataset)
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
    max_updates = 6000  # max update iteration
    max_support_points = 1000  # max support points
    beta = 100
    allowance = 800
    kNS = 4
    sigma = 2
    
    return data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma



# def fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma):
#     fastron = Fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
#     data_graph, y_graph = fastron.update_model()
#     data, alpha, gram_computed, G, F = fastron.active_learning()
#     return data_graph, y_graph, data, alpha, gram_computed, G, F

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

Collision = ColCheck(L1,L2,L3,L_r)
theta_data = Collision.theta_list()
# data_ = theta_data[:,0:3]
# y__ = theta_data[:,[3]]
dataset, indivisual_collision = Collision.collision_check(theta_data,0,2,6,3,0)
data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)
X_ = np.array(theta_data)
y_ = np.array(y)
y_ind = np.array(indivisual_collision)
# print(y_.shape, X_.shape)
# print(np.unique(y_))
# data_graph, y_graph, data, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
# X_ = np.array(data_graph)
# y_ = np.array(y_graph)
# print(y_)
# SVM 모델을 RBF 커널로 훈련시킵니다
# clf = svm.SVC(kernel='rbf')
# clf.fit(X_, y_)

# 3D 플롯을 생성합니다
fig = plt.figure()
# ax = fig.add_subplot(111, projection='2d')

# # 데이터 포인트를 플롯합니다
# for i in range(len(data_)):
    
#     if y_[i] == 1:
#         color = 'r'
#         # ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
#     else:
#         continue
#         # color = 'b'
#         # ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
#     # homgen_0_1 = homogen(0,0,L1,X_[i][0])
#     # homgen_1_2 = homogen(90,0,0,X_[i,1])
#     # homgen_2_3 = homogen(0,L2,0,X_[i,2])
#     # homgen_3_e = homogen(0,L3,0,0)
#     # homgen_1 = homgen_0_1
#     # homgen_2 = homgen_0_1@homgen_1_2
#     # homgen_3 = homgen_0_1@homgen_1_2@homgen_2_3
#     # homgen_e = homgen_0_1@homgen_1_2@homgen_2_3@homgen_3_e
   
#     homgen_0_1 = homogen(np.pi/2, 0, L1, X_[i,0]*np.pi/180)
#     homgen_1_2 = homogen(0, L2, 0, X_[i,1]*np.pi/180)
#     homgen_2_3 = homogen(0, L3, 0, X_[i,2]*np.pi/180)
#     # homgen_3_e = homogen(0,L3,0,0)
#     homgen_1 = homgen_0_1
#     homgen_2 = homgen_0_1@homgen_1_2
#     homgen_3 = homgen_0_1@homgen_1_2@homgen_2_3
    
#     # pe = np.array([0 + 5*np.sin(X[i, 1])+3*np.sin(X[i, 1]+X[i, 2]), 0, 10+5*np.cos(X[i, 1])+3*np.cos(X[i, 1]+X[i, 2])])
#     ax.scatter(homgen_1[0][3],homgen_1[1][3],homgen_1[2][3], s=10, c='g')
#     ax.scatter(homgen_2[0][3],homgen_2[1][3],homgen_2[2][3], s=10, c='b')
#     ax.scatter(homgen_3[0][3],homgen_3[1][3],homgen_3[2][3], s=10, c='color')
#     # ax.scatter(homgen_e[0][3],homgen_e[1][3],homgen_e[2][3], s=10, c=color)
# # 축 레이블을 설정합니다
# ax.scatter(6,6,6,s=20,c='b')

# ax.set_xlabel('q1')
# ax.set_ylabel('q2')
# ax.set_zlabel('q3')
# ax.axis('equal')
# plt.show()
reference_data = []
oo = 0
# 데이터 포인트를 플롯합니다
for i in range(X_.shape[0]):
    
    boolcollisions = y[i][0]
    this_y_ind = y_ind[i][:]
    if y[i] == 1:
        oo += 1
        color = 'r'

        # ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
    else:
        # continue
        color = 'b'
        # ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
    
    homgen_0_1 = homogen(np.pi/2, 0, L1, X_[i,0]*np.pi/180)
    homgen_1_2 = homogen(0, L2, 0, X_[i,1]*np.pi/180)
    homgen_2_3 = homogen(0, L3, 0, X_[i,2]*np.pi/180)
    homgen_1 = homgen_0_1
    homgen_2 = homgen_0_1@homgen_1_2
    homgen_3 = homgen_0_1@homgen_1_2@homgen_2_3
    
    reference_data.append([homgen_1[0][3],homgen_1[1][3],homgen_1[2][3], homgen_2[0][3],homgen_2[1][3],homgen_2[2][3], homgen_3[0][3],homgen_3[1][3],homgen_3[2][3], this_y_ind[0], this_y_ind[1], this_y_ind[2]])
    # pe = np.array([0 + 5*np.sin(X[i, 1])+3*np.sin(X[i, 1]+X[i, 2]), 0, 10+5*np.cos(X[i, 1])+3*np.cos(X[i, 1]+X[i, 2])])
    # ax.plot3D(homgen_1[0,3],homgen_1[1,3],homgen_1[2,3], 'rx')
    # ax.plot3D(homgen_2[0,3],homgen_2[1,3],homgen_2[2,3], 'bx')
    # ax.plot3D(homgen_3[0,3],homgen_3[1,3],homgen_3[2,3], 'gx')
    # ax.scatter(homgen_3[0][3],homgen_3[1][3],homgen_3[2][3], s=10, c=color)

dataset = np.array(dataset)
# np.save('refer.npy', np.array(reference_data))
plotdata = np.array(reference_data)
print("here:", plotdata.shape)
condition1 = np.where(plotdata[:,9]==1)
condition2 = np.where(plotdata[:,10]==1)
condition3 = np.where(plotdata[:,11]==1)
condition1_no = np.where(plotdata[:,9]==-1)
condition2_no = np.where(plotdata[:,10]==-1)
condition3_no = np.where(plotdata[:,11]==-1)

condition_mani = np.where(dataset[:,3]==1)
condition_mani_no = np.where(dataset[:,3]==-1)

print("i:", condition1, condition2, condition3)
# condition2 = np.logical_and(plotdata[:,3]==-1)

# data_collision = plotdata[condition1]
# data_nocollision = plotdata[condition2]
# print("here:", data_collision.shape, data_nocollision.shape)

# plt.plot(plotdata[:,:1])
# plt.subplot(141)
# plt.plot(plotdata[condition1_no,0], plotdata[condition1_no,1],'ob')
# plt.plot(plotdata[condition1,0], plotdata[condition1,1],'xr')
# # plt.legend(['no Collsion','Collision'])
# plt.axis('equal')
# plt.xlim([-12,12])
# plt.ylim([-12,12])
# plt.title('Link1')
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.subplot(142)
# plt.plot(plotdata[condition2_no,3], plotdata[condition2_no,4],'ob')
# plt.plot(plotdata[condition2,3], plotdata[condition2,4],'xr')
# # plt.legend(['no Collsion','Collision'])
# plt.title('Link2')
# plt.axis('equal')
# plt.xlim([-12,12])
# plt.ylim([-12,12])
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.subplot(143)
# plt.plot(plotdata[condition3_no,6], plotdata[condition3_no,7],'ob')
# plt.plot(plotdata[condition3,6], plotdata[condition3,7],'xr')
# # plt.legend(['no Collsion','Collision'])
# plt.title('Link3')
# plt.axis('equal')
# plt.xlim([-12,12])
# plt.ylim([-12,12])
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.subplot(144)
plt.plot(plotdata[condition_mani_no,6], plotdata[condition_mani_no,7],'ob')
plt.plot(plotdata[condition_mani,6], plotdata[condition_mani,7],'xr')
# plt.legend(['no Collsion','Collision'])
plt.title('Manipulator')
plt.axis('equal')
plt.xlim([-12,12])
plt.ylim([-12,12])
plt.xlabel('X')
plt.ylabel('Y')

# data_excel = pd.DataFrame(np.array(reference_data))
# data_excel.to_excel("oooo.xlsx")

print('number of collision:',oo)
# 축 레이블을 설정합니다
# ax.scatter(6,6,6,s=30,c='g')

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.axis('auto')
# plt.xlabel('X')
# plt.ylabel('Y')

plt.show()

# for i in range(4):
    
#     Collision1 = ColCheck(3,6,6,1)
#     dataset = Collision1.collision_check(data)
#     data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset, alpha, gram_computed, G, F)
#     data_graph, y_graph, data, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
#     # data_excel = pd.DataFrame(data)
#     # data_excel.to_excel("ooo.xlsx")
#     X = np.array(data_graph)
#     y = np.array(y_graph)
#     print(np.unique(y))
#     # SVM 모델을 RBF 커널로 훈련시킵니다
#     clf = svm.SVC(kernel='rbf')
#     # print
#     clf.fit(X, y)
#     print(np.unique(clf.predict(X)))
#     # 3D 플롯을 생성합니다
#     fig = plt.figure()
#     # ax = fig.add_subplot(111, projection='3d')
#     ax1 = fig.add_subplot(131)
#     ax2 = fig.add_subplot(132)
#     ax3 = fig.add_subplot(133)

#     # 데이터 포인트를 플롯합니다
#     q3_min = np.min(X[:, 0])
#     q3_max = np.max(X[:, 0])
#     q3_range = q3_max-q3_min
#     for i in range(X.shape[0]):
#         if y[i] == -1:
#             color = 'r'
#         else:
#             color = 'b'

#         if X[i, 0] < q3_min + q3_range / 3:
#             ax1.scatter(X[i, 1], X[i, 2], s=10, c=color)
#         elif q3_min + q3_range / 3 <= X[i, 0] < q3_min + 2 * q3_range / 3:
#             ax2.scatter(X[i, 1], X[i, 2], s=10, c=color)
#         else:
#             ax3.scatter(X[i, 1], X[i, 2], s=10, c=color)

#     # 축 레이블을 설정합니다
#     ax.set_xlabel('q1')
#     ax.set_ylabel('q2')
#     ax.set_zlabel('q3')

#     plt.show()