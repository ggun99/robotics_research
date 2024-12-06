from collisioncheck2 import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from fastronWrapper.fastronWrapper import PyFastron

def data_generation(dataset, alpha = None, gram_computed = None, G = None, F = None):
    # dataset = Collision.collision_check(theta_data
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
    allowance = 1800
    kNS = 4
    sigma = 2
    
    return data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma

L1 = 3
L2 = 6
L3 = 6
L_r = 0.1

Collision = ColCheck(L1,L2,L3,L_r)
theta_data = Collision.theta_list()
base_x = 0
dataset, indi ,col = Collision.collision_check(theta_data,0,2,6,3,base_x)
data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)

fastron__ = PyFastron(data)
fastron__.y = y
fastron__.g = 100
fastron__.maxUpdates = 105000
fastron__.maxSupportPoints = 15000
fastron__.beta = 100

print('here',fastron__.data.shape)

fastron__.activeLearning()
fastron__.updateModel()

X = np.array(data)
# y = np.array(y)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 3D 플롯을 생성합니다
# print('start eval')
# for i in range(data.shape[0]):
#     y_pred = fastron__.eval(data)
#     if y_pred[i] == -1:
#         color = 'ob'
#     else:
#         color = 'xr'
    
#     # ax.scatter(X[i, 0], X[i, 1], X[i, 2], s=10, c=color)
# print('finish eval')



y_pred = fastron__.eval(data)
print(f"==>> y_pred.shape: {y_pred.shape}")
# y_pred 
z_values = X[:, 2]
range1 = np.where(z_values <= 30)
range2 = np.where((30 < z_values) & (z_values <= 60))
range3 = np.where(60 < z_values)
# range1_ = X[range1]
# range2_ = X[range2]
# range3_ = X[range3]
range1_X = X[:,:2][range1]
range2_X = X[:,:2][range2]
range3_X = X[:,:2][range3]
# range1_y = y[:,0][range1]
# range2_y = y[:,0][range2]
# range3_y = y[:,0][range3]
range1_y = y_pred[:,0][range1]
range2_y = y_pred[:,0][range2]
range3_y = y_pred[:,0][range3]
# print(range1_y)
# print(range1_X.shape)
# condition = np.where(y_pred == 1)
# condition_no = np.where(y_pred == -1)
# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# svc = svm.SVC(kernel='rbf')
# svc.fit(range1_X,range1_y)
# # 축 레이블을 설정합니다
# ax.set_xlabel('q1')
# ax.set_ylabel('q2')
# ax.set_zlabel('q3')
# ax.axis('auto')
# plt.show()


C = 1.0  # SVM regularization parameter
clf = svm.SVC(kernel='rbf', gamma=5.0, C=C)

range3_y = range3_y.astype(int)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
range3_X_scaled = scaler.fit_transform(range3_X)
clf.fit(range3_X_scaled, range3_y)

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X=range3_X_scaled, y=range3_y, clf=clf)

plt.scatter(range3_X[:, 0], range3_X[:, 1], c=range3_y, edgecolors='k', marker='o', alpha=0.6)
plt.xlabel("scaled joint 2 angle(q2)")
plt.ylabel("scaled joint 3 angle(q3)")
plt.show()

def plot(num):
    if num == 1:
        range_X = range1_X
        range_y = range1_y
    elif num == 2:
        range_X = range2_X
        range_y = range2_y
    else:
        range_X = range3_X
        range_y = range3_y
    C = 1.0  # SVM regularization parameter
    clf = svm.SVC(kernel='rbf', gamma=1.0, C=C)

    range_y = range_y.astype(int)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    range_X_scaled = scaler.fit_transform(range_X)
    clf.fit(range_X_scaled, range_y)

    from mlxtend.plotting import plot_decision_regions
    plot_decision_regions(X=range_X_scaled, y=range_y, clf=clf)

    plt.scatter(range3_X[:, 0], range3_X[:, 1], c=range_y, edgecolors='k', marker='o', alpha=0.6)
    plt.xlabel("scaled joint 1 angle(q1)")
    plt.ylabel("scaled joint 2 angle(q2)")
    plt.show()
# w = clf.coef_[0]
# a = -w[0]/w[1]
# xx = np.linspace(1,9)
# yy = a*xx-(clf.intercept_[0]/w[1])

# plt.scatter(range1_X[:,0],range1_X[:,1],c=range1_y)
# plt.plot(xx,yy)
# plt.show()

# # Set-up 2x2 grid for plotting.
# fig, sub = plt.subplots(1, 1)
# plt.subplots_adjust(wspace=0.4, hspace=0.4)

# X0, X1 = X[:, 0], X[:, 1]
# xx, yy = make_meshgrid(X0, X1)
# print('3')

# fig = plt.figure()
# ax = fig.add_subplot(111)

# plot_contours(ax, clf, xx, yy,
#                 cmap=plt.cm.coolwarm, alpha=0.8)
# print('4')
# ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
# print('5')
# ax.set_xlim(xx.min(), xx.max())
# ax.set_ylim(yy.min(), yy.max())
# ax.set_xlabel('Sepal length')
# ax.set_ylabel('Sepal width')
# ax.set_xticks(())
# ax.set_yticks(())
# ax.set_title('title')

# plt.show()


