import gjk
import math
import numpy as np
from mpl_toolkits.mplot3d import axes3d
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
robot_link1 = 150.0
robot_link2 = 100.0
robot_link3 = 50.0
robot_thickness = 20.0
obstacle = ((50, 100), 30)
# obstacle = ((np.random.randint(10, 200),np.random.randint(10, 200)), np.random.randint(10, 50))

collision_true = []
collision_false = []
result1 = []
result2 = []
result3 = []
X_s_1 = []
X_s_2 = []
X_s_3 = []

# calculate collision with circle obstacle
def run_random_angle():
    t1 =time()
    for i in range(1000):
        q1_rad = (np.random.rand(1)[0] * 360)
        q2_rad = (np.random.rand(1)[0] * 360)
        q3_rad = (np.random.rand(1)[0] * 360)
        q = [q1_rad] + [q2_rad] + [q3_rad]


        q1 = math.radians(q1_rad)
        q2 = math.radians(q2_rad)
        q3 = math.radians(q3_rad)

        link1_1=np.array(([0, robot_thickness/2], [0,-robot_thickness/2], [robot_link1, -robot_thickness/2], [robot_link1, robot_thickness/2]))
        r1 = np.array(([math.cos(q1), -math.sin(q1)], [math.sin(q1), math.cos(q1)]))
        link1_ro = np.matmul(r1, link1_1.T)
        link_1 = link1_ro.T

        link2_1 = np.array(([0, robot_thickness/2], [0, -robot_thickness/2], [robot_link2, -robot_thickness/2], [robot_link2, robot_thickness/2]))
        r2 = np.array(([math.cos(q2), -math.sin(q2)], [math.sin(q2), math.cos(q2)]))
        link2_ro = np.matmul(r2, link2_1.T)
        # link2_ro += np.array(([math.cos(q1)*robot_link1], [math.sin(q1)*robot_link1]))
        link2_ro += np.array(([robot_link1], [0]))
        link2_ro2 = np.matmul(r1, link2_ro)
        link_2 = link2_ro2.T
        # link_2 = link2_ro.T

        link3_1 = np.array(([0, robot_thickness/2], [0, -robot_thickness/2], [robot_link3, -robot_thickness/2], [robot_link3, robot_thickness/2]))
        r3 = np.array(([math.cos(q3), -math.sin(q3)], [math.sin(q3), math.cos(q3)]))
        link3_ro = np.matmul(r3, link3_1.T)
        # link2_ro += np.array(([math.cos(q1)*robot_link1], [math.sin(q1)*robot_link1]))
        link3_ro += np.array(([robot_link2], [0]))
        link3_ro2 = np.matmul(r2, link3_ro)
        link3_ro2 += np.array(([robot_link1], [0]))
        link3_ro3 = np.matmul(r1, link3_ro2)
        link_3 = link3_ro3.T

        # collision check with circle obstacle
        collide_1 = gjk.collidePolyCircle(link_1, obstacle)
        circle(obstacle)
        collide_2 = gjk.collidePolyCircle(link_2, obstacle)
        circle(obstacle)
        collide_3 = gjk.collidePolyCircle(link_3, obstacle)
        circle(obstacle)

        if collide_1 or collide_2:
            collision = 1
            collision_true.append(q)
            result1.append(collision)
        else:
            collision = 0
            collision_false.append(q)
            result1.append(collision)

        if collide_2 or collide_3:
            collision = 1
            collision_true.append(q)
            result2.append(collision)
        else:
            collision = 0
            collision_false.append(q)
            result2.append(collision)
        
        if collide_1 or collide_3:
            collision = 1
            collision_true.append(q)
            result3.append(collision)
        else:
            collision = 0
            collision_false.append(q)
            result3.append(collision)

        X_s_1.append(q1_rad)
        X_s_1.append(q2_rad)

        X_s_2.append(q2_rad)
        X_s_2.append(q3_rad)

        X_s_3.append(q1_rad)
        X_s_3.append(q3_rad)
    
    t2 =time()
    print('t_rand_ang',t2-t1)
        #i += i+1
        
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
    t1 = time()
    # plt.scatter([], [], color='BLUE', marker='s', label='collision_true')
    # plt.scatter([], [], color='RED', marker='o', label='collision_false')
    fig = plt.figure(figsize=(10,7))
    ax = plt.axes(projection='3d')
    for coordinates in collision_true:
        x, y, z = coordinates
        ax.scatter(x,y,z,color='BLUE', marker='s')
        # plt.scatter(x, y, z, color='BLUE', s=10, alpha=0.5, marker='s')
    for coordinates in collision_false:
        x, y, z = coordinates
        ax.scatter(x,y,z,color='RED', marker='o')
        # plt.scatter(x, y, z, color='RED', s=10, alpha=0.5, marker='o')
    t2 = time()
    print('t_C_space',t2-t1)
    # plt.xlabel("joint 1 angle(q1, degrees)")
    # plt.ylabel("joint 2 angle(q2, degrees)")
    # plt.title("C-space")
    plt.show()


def svm_map():
    t1 = time()
    svm = SVC(kernel='rbf', random_state=1, C=20, gamma=0.001)

    X1 = np.array(X_s_1)
    X1 = np.reshape(X1, (-1, 2))
    y1 = np.array(result1)

    X2 = np.array(X_s_2)
    X2 = np.reshape(X2, (-1, 2))
    y2 = np.array(result2)

    X3 = np.array(X_s_3)
    X3 = np.reshape(X3, (-1, 2))
    y3 = np.array(result3)

    svm.fit(X1, y1)
    
    # Plotting decision regions
    t2 = time()
    print('t_SVM1',t2-t1)
    plot_decision_regions(X=X1, y=y1, clf=svm)
    plt.xlabel("joint 1 angle(q1, degrees)")
    plt.ylabel("joint 2 angle(q2, degrees)")
    plt.show()
    svm.fit(X2, y2)
    t3 = time()
    print('t_SVM2',t3-t1)
    plot_decision_regions(X=X2, y=y2, clf=svm)
    plt.xlabel("joint 2 angle(q2, degrees)")
    plt.ylabel("joint 3 angle(q3, degrees)")
    plt.show()
    svm.fit(X3, y3)
    t4 = time()
    print('t_SVM3',t4-t1)
    plot_decision_regions(X=X3, y=y3, clf=svm)
    plt.xlabel("joint 1 angle(q1, degrees)")
    plt.ylabel("joint 3 angle(q3, degrees)")
    plt.show()
    # t2 = time()
    # print('t_SVM',t2-t1)
    
    
    

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
    # t1 = time()
    # collision check for N time
    run_random_angle()
    # t2 = time()
    # print('t_rad_ang',t2-t1)
    # make C-space graph
    make_C_space()
    # t3 = time()
    # print('t_C_space',t3-t2)
    # make svm_C-space graph
    svm_map()
    # t4 = time()
    # print('t_SVM',t4-t3)


# mlxtend.plotting 라이브러리의 plot_decision_regions 이용
# 경계면을 그리는 속도 단축
