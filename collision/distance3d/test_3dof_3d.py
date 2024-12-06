"""
========================================================
Check the Collision between Links and obstacle with GJK
========================================================
"""
print(__doc__)
import time
import numpy as np
from pytransform3d.transformations import transform_from
import pytransform3d.rotations as pyrot
from distance3d import gjk, colliders
import math
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

L1_len = 10
L2_len = 5
L3_len = 3
radius = 1

collision = []
no_collision = []
result = []
X = []
data = []

def rand_choice(data1, data2):
    # 랜덤한 인덱스 생성
    random_indices = np.random.choice(data1.shape[0], 1000, replace=False)

    # 랜덤한 인덱스에 해당하는 행 추출
    random_data1 = data1[random_indices]
    random_data2 = data2[random_indices]
    # random_data3 = data3[random_indices]
    return random_data1, random_data2 #, random_data3

def collision_check():
    accumulated_time = 0.0
    center1 = np.array([0,0,L1_len])
    start = time.time()
    for i in range(360):
        theta = math.radians(i)
        # print(f"==>> theta: {theta}")
        for j in range(360):
            phi = math.radians(j)
            for k in range(360):
                gamma = math.radians(k)
                data.append([theta,phi,gamma])
    data_ = np.random.choice(len(data), 1000, replace=False)
    for l in range(data_.shape[0]):
        #random number of data
        rn = data_[l]
        theta = data[rn][0]
        phi = data[rn][1]
        gamma = data[rn][2]
        #Link1
        p=np.array([0.,0.,L1_len/2])
        a = np.array([0.0, 0.0, 1.0, np.degrees(theta)])
        R_1 = pyrot.matrix_from_axis_angle(a)
        L1_2_origin = transform_from(R_1,p)
        #Link2
        p2=np.array([L2_len/2*np.sin(phi),0.,L1_len+L2_len/2*np.cos(phi)])
        a2=np.array([0.,1.,0.,np.degrees(phi)])
        R_2 = pyrot.matrix_from_axis_angle(a2)
        L2_2_L1 = transform_from(R_2@R_1,p2)
        #Link3
        p3 = np.array([L2_len*np.sin(phi)+L3_len/2*np.sin(phi+gamma),0.,L1_len+L2_len*np.cos(phi)+L3_len/2*np.cos(phi+gamma)])
        a3 = np.array([0.,1.,0.,np.degrees(gamma)])
        R_3 = pyrot.matrix_from_axis_angle(a3)
        L3_2_L2 = transform_from(R_3@R_2@R_1,p3)
        # Links
        L1 = colliders.Cylinder(L1_2_origin, radius, L1_len)
        L2 = colliders.Cylinder(L2_2_L1, radius, L2_len)
        L3 = colliders.Cylinder(L3_2_L2, radius, L3_len)
        # Joints
        J1 = colliders.Sphere(center1, radius)
        center2 = np.array([L2_len*np.sin(phi),0.,L1_len+L2_len*np.cos(phi)])
        J2 = colliders.Sphere(center2, radius)

        # obstacle(sphere)
        obs_p = np.array([5., 2., 10.])
        obs_r = 3
        obs = colliders.Sphere(obs_p, obs_r)

        # obstacle(box)
        # obs_box = np.array([[1.,0.,0.,3.5],[0.,1.,0.,0.5],[0.,0.,1.,8.5],[0.,0.,0.,1.]])
        # obs_box_size = np.array([3.,3.,3.])
        # obs = colliders.Box(obs_box,obs_box_size)

        # obstacle(cylinder)
        # obs_cyl = np.array([[1.,0.,0.,3.],[0.,1.,0.,2.],[0.,0.,1.,7.],[0.,0.,0.,1.]])
        # obs_cyl_size = radius
        # obs = colliders.Cylinder(obs_cyl,obs_cyl_size, 6.)

        # if you want to know distance with link2 and obstacle
        #dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(L2, obs)

        # collision check
        
        s = np.degrees(theta)
        d = np.degrees(phi)
        f = np.degrees(gamma)
        X.append([s,d,f])
        check_collision_L3 = gjk.gjk_intersection(L3,obs)
        if check_collision_L3 is True:
            collision.append([s,d,f])
            result.append(1)
        else:
            check_collision_L2 = gjk.gjk_intersection(L2,obs)
            if check_collision_L2 is True:
                collision.append([s,d,f])
                result.append(1)
            else:
                check_collision_L1 = gjk.gjk_intersection(L1,obs)
                if check_collision_L1 is True:
                    collision.append([s,d,f])
                    result.append(1)
                else:  
                    no_collision.append([s,d,f])
                    result.append(0)
            
    end = time.time()
    accumulated_time += end - start

    print('collision',np.shape(collision))
    print('no collision',np.shape(no_collision))
    print(f"{accumulated_time=}")


def c_svm():
    # t1 = time.time()
    svm = SVC(kernel='rbf', random_state=1, C=50, gamma=0.0005)
    
    # X_s = np.array(X)
    X_s = list(map(lambda x: x[:2], X))
    X_s = np.array(X_s)
    
    X_a = list(map(lambda x: x[1:], X))
    X_a = np.array(X_a)
   
    X_b = list(map(lambda x: x[0:2:1], X))
    X_b = np.array(X_b)
    # X_s = np.reshape(X_s, (-1, 2))
    y = np.array(result)
    
    
    # randomly choose 1000 data
    # X_ran, Y_ran = rand_choice(X_s, y)
    # X_ran = X_s
    Y_ran = y
    # svm.fit(X_ran, Y_ran)
    # X_1 = X_a
    X_2 = X_b
    # svm.fit(X_1, Y_ran)
    svm.fit(X_2, Y_ran)
    
    # t2 = time.time()
    # print('svm time', t2-t1)

    # Plotting decision regions
    # plot_decision_regions(X=X_ran, y=Y_ran, clf=svm)
    # t2 = time.time()
    # print('t_c_SVM',t2-t1)
    # plt.xlabel("joint 1 angle(q1, degrees)")
    # plt.ylabel("joint 2 angle(q2, degrees)")
    # plt.title("C-SVM")
    # plt.show()
    # plot_decision_regions(X=X_1, y=Y_ran, clf=svm)
    # plt.xlabel("joint 2 angle(q2, degrees)")
    # plt.ylabel("joint 3 angle(q3, degrees)")
    # plt.title("C-SVM")
    # plt.show()
    # plot_decision_regions(X=X_2, y=Y_ran, clf=svm)
    # plt.xlabel("joint 1 angle(q1, degrees)")
    # plt.ylabel("joint 3 angle(q2, degrees)")
    # plt.title("C-SVM")
    # plt.show()
if __name__ == '__main__':
    collision_check()
    c_svm()