import numpy as np
import random
import math
from pytransform3d.transformations import transform_from
import pytransform3d.rotations as pyrot
from distance3d import gjk, colliders
import time

class ThreeDOF_CollisionCheck:
    def __init__(self, L1_len, L2_len, L3_len, radius, dist_x, dist_y):
        self.L1_len = L1_len
        self.L2_len = L2_len
        self.L3_len = L3_len
        self.radius = radius
        self.x = dist_x
        self.y = dist_y
      
        # self.dist_r = dist_r
    def theta_list(self):
        theta_data = []
        for i in range(36):
            theta = i*10
            # print(f"==>> theta: {theta}")
            for j in range(36):
                theta2 = j*10
                for k in range(36):
                    theta3 = k*10
                    theta_data.append([theta,theta2,theta3])
        rand_theta_data = random.sample(theta_data,4000)
        return rand_theta_data
    
    def Arm_with_Mobile(self, theta, theta2, theta3):
        center1 = np.array([0,0,self.L1_len])
        # Link1
        p=np.array([self.x, self.y, self.L1_len/2])
        a = np.array([0.0, 0.0, 1.0, np.degrees(theta)])
        R_1 = pyrot.matrix_from_axis_angle(a)
        L1_2_origin = transform_from(R_1,p)
        # Link2
        p2=np.array([self.x + self.L2_len/2*np.sin(theta2), self.y, self.L1_len+self.L2_len/2*np.cos(theta2)])
        a2=np.array([0.,1.,0.,np.degrees(theta2)])
        R_2 = pyrot.matrix_from_axis_angle(a2)
        L2_2_L1 = transform_from(R_2@R_1,p2)
        # Link3
        p3 = np.array([self.x + self.L2_len*np.sin(theta2)+self.L3_len/2*np.sin(theta2+theta3), self.y, self.L1_len+self.L2_len*np.cos(theta2)+self.L3_len/2*np.cos(theta2+theta3)])
        a3 = np.array([0.,1.,0.,np.degrees(theta3)])
        R_3 = pyrot.matrix_from_axis_angle(a3)
        L3_2_L2 = transform_from(R_3@R_2@R_1,p3)
        # Links
        L1 = colliders.Cylinder(L1_2_origin, self.radius, self.L1_len)
        L2 = colliders.Cylinder(L2_2_L1, self.radius, self.L2_len)
        L3 = colliders.Cylinder(L3_2_L2, self.radius, self.L3_len)
        # Joints
        J1 = colliders.Sphere(center1, self.radius)
        center2 = np.array([self.L2_len*np.sin(theta2),0.,self.L1_len+self.L2_len*np.cos(theta2)])
        J2 = colliders.Sphere(center2, self.radius)
        
        # mobile robot
        mobile_box_size = np.array([5.,10.,3.])
        p_m = np.array([1.,0.,-1.5])
        a_m = np.array([0.,0.,1.,theta])
        mobile_2_origin = transform_from(pyrot.matrix_from_axis_angle(a_m),p_m)
        # mobile = colliders.Box(mobile_box,mobile_box_size)
        p_mw = np.array([-3.,0.,-3.3])
        p_mw2 = np.array([5.,0.,-3.3])
        a_mw = a_m = np.array([1.,0.,0.,theta])
        mobile_wheel_2_box = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw)
        mobile_wheel_2_box2 = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw2)

        return L1, L2, L3
    
    def obstacle(self, x, y, z, radius):
        obs_p = np.array([x, y, z], dtype = float)
        obs_r = radius
        obs = colliders.Sphere(obs_p, obs_r)
        return obs

    def collision_check(self, theta_data):
        dataset = []
        accumulated_time = 0.0
        
        start = time.time()
        print('collision checking..')
        for l in range(len(theta_data)):
            
            theta = theta_data[l][0]
            theta2 = theta_data[l][1]
            theta3 = theta_data[l][2]

            # obstacle(sphere)
            # obs_p = np.array([5., 2., 10.])
            # obs_r = 3
            # obs = colliders.Sphere(obs_p, obs_r)
            obs = self.obstacle(5., 2., 10., 2)

            # if you want to know distance with link2 and obstacle
            #dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(L2, obs)

            # collision check
            # s = np.degrees(theta)
            # d = np.degrees(theta2)
            # f = np.degrees(theta3)
            L1 ,L2, L3 = self.Arm_with_Mobile(theta, theta2, theta3)
            check_collision_L3 = gjk.gjk_intersection(L3,obs)
            if check_collision_L3 is True:
                dataset.append([theta,theta2,theta3,1])
                # collision.append([s,d,f])
                # result.append(1)
            else:
                check_collision_L2 = gjk.gjk_intersection(L2,obs)
                if check_collision_L2 is True:
                    dataset.append([theta,theta2,theta3,1])
                    # collision.append([s,d,f])
                    # result.append(1)
                else:
                    check_collision_L1 = gjk.gjk_intersection(L1,obs)
                    if check_collision_L1 is True:
                        dataset.append([theta,theta2,theta3,1])
                        # collision.append([s,d,f])
                        # result.append(1)
                    else:  
                        dataset.append([theta,theta2,theta3,-1])
                        # no_collision.append([s,d,f])
                        # result.append(-1)

        end = time.time()
        accumulated_time += end - start
        return dataset