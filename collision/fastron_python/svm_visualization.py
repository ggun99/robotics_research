from faastron_test import Fastron
from collisioncheck import ThreeDOF_CollisionCheck as ColCheck
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
# import pandas as pd


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



def fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma):
    fastron = Fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
    data_graph, y_graph = fastron.update_model()
    data, alpha, gram_computed, G, F = fastron.active_learning()
    return data_graph, y_graph, data, alpha, gram_computed, G, F


# data = None
# alpha, gram_computed, G, F = None

Collision = ColCheck(10,5,3,1,0,0)
theta_data = Collision.theta_list()
dataset = Collision.collision_check(theta_data)
data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset)
data_graph, y_graph, data, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
X = np.array(data_graph)
y = np.array(y_graph)
print(y)
# SVM 모델을 RBF 커널로 훈련시킵니다
clf = svm.SVC(kernel='rbf')
clf.fit(X, y)

# 3D 플롯을 생성합니다
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 데이터 포인트를 플롯합니다
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)

# 축 레이블을 설정합니다
ax.set_xlabel('q1')
ax.set_ylabel('q2')
ax.set_zlabel('q3')

plt.show()

for i in range(4):
    
    Collision1 = ColCheck(3,6,6,1, i*0.3+0.3,0)
    dataset = Collision1.collision_check(data)
    data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma = data_generation(dataset, alpha, gram_computed, G, F)
    data_graph, y_graph, data, alpha, gram_computed, G, F = fastron(data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma)
    # data_excel = pd.DataFrame(data)
    # data_excel.to_excel("ooo.xlsx")
    X = np.array(data_graph)
    y = np.array(y_graph)

    # SVM 모델을 RBF 커널로 훈련시킵니다
    clf = svm.SVC(kernel='rbf')
    print
    clf.fit(X, y)

    # 3D 플롯을 생성합니다
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 데이터 포인트를 플롯합니다
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=10)

    # 축 레이블을 설정합니다
    ax.set_xlabel('q1')
    ax.set_ylabel('q2')
    ax.set_zlabel('q3')

    plt.show()