import gjk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
from time import time

# color
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
BLUE  = (  0,   0, 255)
GREEN = (  0, 255,   0)
RED   = (255,   0,   0)

# information of test environment
robot_link1 = 150
robot_link2 = 100
robot_thickness = 20
obstacle = ((50, 100), 30)
# obstacle = ((np.random.randint(10, 200),np.random.randint(10, 200)), np.random.randint(10, 50))

collision_true = []
collision_false = []
result = []
X_s = []

# calculate collision with circle obstacle
def run_random_angle():
    t1 =time()
    for i in range(1000):
        q1_rad = (np.random.rand(1)[0] * 360)
        q2_rad = (np.random.rand(1)[0] * 360)
        
        q = [q1_rad] + [q2_rad]


        q1 = math.radians(q1_rad)
        q2 = math.radians(q2_rad)

        link1_1=np.array(([0, robot_thickness/2], [0,-robot_thickness/2], [robot_link1, -robot_thickness/2], [robot_link1, robot_thickness/2]))
        r1 = np.array(([math.cos(q1), -math.sin(q1)], [math.sin(q1), math.cos(q1)]))
        link1_ro = np.matmul(r1, link1_1.T)
        link_1 = (link1_ro.T)

        link2_1 = np.array(([0, robot_thickness/2], [0, -robot_thickness/2], [robot_link2, -robot_thickness/2], [robot_link2, robot_thickness/2]))
        r2 = np.array(([math.cos(q2), -math.sin(q2)], [math.sin(q2), math.cos(q2)]))
        link2_ro = np.matmul(r2, link2_1.T)
        link2_ro += np.array(([robot_link1], [0]))
        link2_ro2 = np.matmul(r1, link2_ro)
        link_2 = (link2_ro2.T)

        # collision check with circle obstacle
        collide_1 = gjk.collidePolyCircle(link_1, obstacle)
        circle(obstacle)
        collide_2 = gjk.collidePolyCircle(link_2, obstacle)
        circle(obstacle)

        if collide_1 or collide_2:
            collision = 1
            collision_true.append(q)
            result.append(collision)
        else:
            collision = 0
            collision_false.append(q)
            result.append(collision)
        
        X_s.append(q1_rad)
        X_s.append(q2_rad)
    t2 =time()
    print('t_C_space',t2-t1)
        
    # show robot link

    # fig, ax = plt.subplots()
    # body = Polygon(link_1)
    # ax.add_patch(body)
    # body = Polygon(link_2)
    # ax.add_patch(body)
    # ax.set_xlim(-300, 300)
    # ax.set_ylim(-300, 300)
    # plt.show()


def make_C_space():

    plt.scatter([], [], color='BLUE', marker='s', label='collision_true')
    plt.scatter([], [], color='RED', marker='o', label='collision_false')

    for coordinates in collision_true:
        x, y = coordinates
        plt.scatter(x, y, color='BLUE', s=10, alpha=0.5, marker='s')
    for coordinates in collision_false:
        x, y = coordinates
        plt.scatter(x, y, color='RED', s=10, alpha=0.5, marker='o')

    plt.xlabel("joint 1 angle(q1, degrees)")
    plt.ylabel("joint 2 angle(q2, degrees)")
    plt.title("C-space")
    plt.legend()
    plt.show()


def svm_map():

    svm = SVC(kernel='rbf', random_state=1, C=10, gamma=0.001)

    X = np.array(X_s)
    X = np.reshape(X, (-1, 2))
    y = np.array(result)

    svm.fit(X, y)

    # Plotting decision regions
    plot_decision_regions(X=X, y=y, clf=svm)

    plt.show()


    # import itertools
    # import matplotlib.gridspec as gridspec
    # gs = gridspec.GridSpec(2, 2)

    # c10_g0001 = SVC(kernel='rbf', random_state=1, C=10, gamma=0.001)
    # c10_g01 = SVC(kernel='rbf', random_state=1, C=10, gamma=0.1)
    # c01_g0001 = SVC(kernel='rbf', random_state=1, C=0.1, gamma=0.001)
    # c01_g01 = SVC(kernel='rbf', random_state=1, C=0.1, gamma=0.1)

    # labels = ['c10_g0001', 'c10_g01', 'c01_g0001', 'c01_g01']

    # for clf, lab, grd in zip([c10_g0001, c10_g01, c01_g0001, c01_g01],
    #                      labels,
    #                      itertools.product([0, 1], repeat=2)):
    #     clf.fit(X, y)
    #     ax = plt.subplot(gs[grd[0], grd[1]])
    #     fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    #     plt.title(lab)

    # plt.show()


def pairs(points):
    for i, j in enumerate(range(-1, len(points) - 1)):
        yield (points[i], points[j])
def circles(cs, color=BLACK, camera=(0, 0)):
    for c in cs:
        circle(c, color, camera)
def circle(c, color=BLACK, camera=(0, 0)):
    ()
def polygon(points, color=BLACK, camera=(0, 0)):
    for a, b in pairs(points):
        line(a, b, color, camera)
def line(start, end, color=BLACK, camera=(0, 0)):
    ()
def add(p1, p2):
    return p1[0] + p2[0], p1[1] + p2[1]


# run code 
if __name__ == '__main__':

    # collision check for N time
    run_random_angle()
    # make C-space graph
    make_C_space()
    # make svm_C-space graph
    svm_map()



# mlxtend.plotting 라이브러리의 plot_decision_regions 이용
# 경계면을 그리는 속도 단축
