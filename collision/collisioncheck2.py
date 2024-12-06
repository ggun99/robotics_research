import numpy as np
import random
import math
# from pytransform3d.transformations import transform_from
# import pytransform3d.rotations as pyrot
# from distance3d import gjk, colliders
import time
import pytransform3d.plot_utils as ppu
import matplotlib.pyplot as plt


class ThreeDOF_CollisionCheck:
    def __init__(self, L1_len, L2_len, L3_len, radius):
        self.L1_len = L1_len
        self.L2_len = L2_len
        self.L3_len = L3_len
        self.radius = radius
        # self.x = dist_x
        # self.y = dist_y
      
        # self.dist_r = dist_r
    def theta_list(self):
        theta_data = []
        for i in range(30):
            theta = i*12
            # theta = np.radians(theta)
            # print(f"==>> theta: {theta}")
            for j in range(30):
                theta2 = j*12
                # theta2 = np.radians(theta2)
                for k in range(30):
                    theta3 = k*12
                    # theta3 = np.radians(theta3)
                    theta_data.append([theta,theta2,theta3])
        rand_theta_data = random.sample(theta_data, 3000)
        return rand_theta_data, theta_data
    

    def homogen(self, alpha, a, d, theta):
        mat = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0,0,0,1]])
        return mat
    
    # def Arm_with_Mobile(self, theta, theta2, theta3):
    #     center1 = np.array([0,0,self.L1_len])
    #     homgen_0_1 = self.homogen(0,0,3,theta)
    #     homgen_1_2 = self.homogen(90,0,0,theta2)
    #     homgen_2_3 = self.homogen(0,6,0,theta3)
    #     # Link1
    #     # p=np.array([self.x, self.y, self.L1_len/2])
    #     p = homgen_0_1@np.array([0,0,1.5,1])
    #     p = p[:3].T
    #     a = np.array([0.0, 0.0, 1.0, np.degrees(theta)])
    #     R_1 = pyrot.matrix_from_axis_angle(a)
    #     L1_2_origin = transform_from(R_1,p)
    #     # Link2
    #     # p2=np.array([self.x + self.L2_len/2*np.sin(theta2), self.y, self.L1_len+self.L2_len/2*np.cos(theta2)])
    #     p2 = homgen_1_2@np.array([3,0,0,1])
    #     p2 = p2[:3].T
    #     a2=np.array([0.,1.,0.,np.degrees(theta2)])
    #     R_2 = pyrot.matrix_from_axis_angle(a2)
    #     L2_2_L1 = transform_from(R_2@R_1,p2)
    #     # Link3
    #     # p3 = np.array([self.x + self.L2_len*np.sin(theta2)+self.L3_len/2*np.sin(theta2+theta3), self.y, self.L1_len+self.L2_len*np.cos(theta2)+self.L3_len/2*np.cos(theta2+theta3)])
    #     p3 = homgen_2_3@np.array([3,0,0,1])
    #     p3 = p3[:3].T
    #     a3 = np.array([0.,1.,0.,np.degrees(theta3)])
    #     R_3 = pyrot.matrix_from_axis_angle(a3)
    #     L3_2_L2 = transform_from(R_3@R_2@R_1,p3)
    #     # Links
    #     L1 = colliders.Cylinder(L1_2_origin, self.radius, self.L1_len)
    #     L2 = colliders.Cylinder(L2_2_L1, self.radius, self.L2_len)
    #     L3 = colliders.Cylinder(L3_2_L2, self.radius, self.L3_len)
    #     # ax = ppu.make_3d_axis(ax_s=2)
    #     # ppu.plot_cylinder(ax, A2B=L1_2_origin,radius=1, length=6, wireframe=False, color='b',alpha=0.2)
    #     # ppu.plot_cylinder(ax, A2B=L2_2_L1,radius=1, length=6, wireframe=False, color='b',alpha=0.2)
    #     # plt.show()
    #     # ppu.plot_cylinder(ax, A2B=L3_2_L2,radius=1, length=6, wireframe=False, color='b',alpha=0.2)
    #     # Joints
    #     J1 = colliders.Sphere(center1, self.radius)
    #     center2 = np.array([self.L2_len*np.sin(theta2),0.,self.L1_len+self.L2_len*np.cos(theta2)])
    #     J2 = colliders.Sphere(center2, self.radius)
        
    #     # mobile robot
    #     mobile_box_size = np.array([5.,10.,3.])
    #     p_m = np.array([1.,0.,-1.5])
    #     a_m = np.array([0.,0.,1.,theta])
    #     mobile_2_origin = transform_from(pyrot.matrix_from_axis_angle(a_m),p_m)
    #     # mobile = colliders.Box(mobile_box,mobile_box_size)
    #     p_mw = np.array([-3.,0.,-3.3])
    #     p_mw2 = np.array([5.,0.,-3.3])
    #     a_mw = a_m = np.array([1.,0.,0.,theta])
    #     mobile_wheel_2_box = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw)
    #     mobile_wheel_2_box2 = transform_from(pyrot.matrix_from_axis_angle(a_mw),p_mw2)

    #     return L1, L2, L3
    
    # def obstacle(self, x, y, z, radius):
    #     obs_p = np.array([x, y, z], dtype = float)
    #     obs_r = radius
    #     obs = colliders.Sphere(obs_p, obs_r)
    #     return obs

    def distance_point_to_line_segment(self,p, a, b):
        """
        Calculate the shortest distance from point p to the line segment ab.
        
        Parameters:
        p (numpy.ndarray): The point [x, y, z]
        a (numpy.ndarray): Start point of the line segment [x, y, z]
        b (numpy.ndarray): End point of the line segment [x, y, z]
        
        Returns:
        float: The shortest distance from point p to the line segment ab
        """
        # Vector from a to b
        ab = b - a
        # Vector from a to p
        ap = p - a
        # Project point p onto the line defined by ab, clamped to the segment [a, b]
        t = np.dot(ap, ab) / np.dot(ab, ab)
        # if t > 1:
        #     return False
        t = np.clip(t, 0, 1)
        # Calculate the closest point on the segment
        closest_point = a + t * ab
        # Return the distance from p to the closest point
        return np.linalg.norm(p - closest_point)

    def collision_check(self,theta_data, obs_x,obs_y,obs_z,obs_r,base_x):
        """
        Check if a cylinder and a sphere are colliding.
        
        Parameters:
        cylinder_start (numpy.ndarray): Start point of the cylinder's axis [x, y, z]
        cylinder_end (numpy.ndarray): End point of the cylinder's axis [x, y, z]
        cylinder_radius (float): Radius of the cylinder
        sphere_center (numpy.ndarray): Center of the sphere [x, y, z]
        sphere_radius (float): Radius of the sphere
        
        Returns:
        bool: True if the cylinder and the sphere are colliding, False otherwise
        """
        dataset = []
        indivisual_collision = []
        collision = []

        for l in range(len(theta_data)):
            
            theta = theta_data[l][0]
            theta2 = theta_data[l][1]
            theta3 = theta_data[l][2]
            theta_r = np.radians(theta)
            theta2_r = np.radians(theta2)
            theta3_r = np.radians(theta3)

            base_position = np.array([base_x,0,0])
            homgen_base = np.hstack((np.eye(3), base_position.reshape(-1,1)))
            # print(homgen_base)
            homgen_base = np.vstack((homgen_base, [0,0,0,1]))
            # print(homgen_base)
            homgen_0_1 = self.homogen(np.pi/2, 0, self.L1_len, theta*np.pi/180)
            homgen_1_2 = self.homogen(0, self.L2_len, 0, theta2*np.pi/180)
            homgen_2_3 = self.homogen(0, self.L3_len, 0, theta3*np.pi/180)
            homgen_1 = homgen_base@homgen_0_1
            homgen_2 = homgen_1@homgen_1_2
            homgen_3 = homgen_2@homgen_2_3

            cylinder_radius = 0.1
            cylinder1_start = base_position
            cylinder1_end = homgen_1[:3,3]
            cylinder2_start = cylinder1_end
            cylinder2_end = homgen_2[:3,3]
            # np.array([(self.L2_len*np.cos(theta2_r)*np.cos(theta_r)),(self.L2_len*np.cos(theta2_r)*np.sin(theta_r)),self.L2_len*np.sin(theta2_r)+self.L1_len])
            cylinder3_start = cylinder2_end 
            cylinder3_end = homgen_3[:3,3]
            cylinder3_end = np.array([cylinder2_end[0]+(self.L3_len*np.cos(theta3_r+theta2_r)*np.cos(theta_r)),cylinder2_end[1]+(self.L3_len*np.cos(theta3_r+theta2_r)*np.cos(theta_r)), cylinder2_end[2]+self.L3_len*np.sin(theta3_r+theta2_r)])
            sphere_center = np.array([obs_x,obs_y,obs_z])
            sphere_radius = obs_r
            # Check the distance from the sphere center to the cylinder's axis
            distance1 = self.distance_point_to_line_segment(sphere_center, cylinder1_start, cylinder1_end)
            distance2 = self.distance_point_to_line_segment(sphere_center, cylinder2_start, cylinder2_end)
            distance3 = self.distance_point_to_line_segment(sphere_center, cylinder3_start, cylinder3_end)

            condition1 = np.where(distance1 < sphere_radius+cylinder_radius, 1, -1)
            condition2 = np.where(distance2 < sphere_radius+cylinder_radius, 1, -1)
            condition3 = np.where(distance3 < sphere_radius+cylinder_radius, 1, -1)
            indivisual_collision.append([condition1, condition2, condition3])
            dataset.append([theta, theta2, theta3, np.max([condition1, condition2, condition3])])
            
            if dataset[l][3] ==1:
                collision.append(dataset[l])
            
            # print("CC:", indivisual_collision)

        return dataset, indivisual_collision ,collision
            #  # Check if the distance is less than the sum of the radii
            # cylinder1_axis = cylinder1_end - cylinder1_start
            # cylinder1_height = np.linalg.norm(cylinder1_axis)
            # cylinder1_axis_normalized = cylinder1_axis / cylinder1_height
            # vector_to_sphere_center = sphere_center - cylinder1_start
            # vector_to_sphere_end = sphere_center - cylinder1_end
            # projection_length = np.dot(vector_to_sphere_center, cylinder1_axis_normalized)
            # limit_length = np.dot(vector_to_sphere_end, cylinder1_axis_normalized)
            # if distance1 < (cylinder_radius + sphere_radius) and 0 <= projection_length <= cylinder1_height+limit_length:
            #     dataset.append([theta,theta2,theta3,1])
            #     # Further check if the sphere center is within the height of the cylinder
            #     # cylinder1_axis = cylinder1_end - cylinder1_start
            #     # cylinder1_height = np.linalg.norm(cylinder1_axis)
            #     # cylinder1_axis_normalized = cylinder1_axis / cylinder1_height
                
            #     # vector_to_sphere_center = sphere_center - cylinder1_start
            #     # projection_length = np.dot(vector_to_sphere_center, cylinder1_axis_normalized)
                
            #     # if 0 <= projection_length <= cylinder1_height:
            #     #     dataset.append([theta,theta2,theta3,1])
                
            # else:
            #     distance2 = self.distance_point_to_line_segment(sphere_center, cylinder2_start, cylinder2_end)
            #     cylinder2_axis = cylinder2_end - cylinder2_start
            #     cylinder2_height = np.linalg.norm(cylinder2_axis)
            #     cylinder2_axis_normalized = cylinder2_axis / cylinder2_height
            #     vector_to_sphere_center = sphere_center - cylinder2_start
            #     vector_to_sphere_end = sphere_center - cylinder2_end
            #     projection_length = np.dot(vector_to_sphere_center, cylinder2_axis_normalized)
            #     limit_length = np.dot(vector_to_sphere_end, cylinder2_axis_normalized)
            #     if distance2 < (cylinder_radius + sphere_radius) and 0 <= projection_length <= cylinder2_height+limit_length:
            #         dataset.append([theta,theta2,theta3,1])
            #         # Further check if the sphere center is within the height of the cylinder
            #         # cylinder2_axis = cylinder2_end - cylinder2_start
            #         # cylinder2_height = np.linalg.norm(cylinder2_axis)
            #         # cylinder2_axis_normalized = cylinder2_axis / cylinder2_height
                    
            #         # vector_to_sphere_center = sphere_center - cylinder2_start
            #         # projection_length = np.dot(vector_to_sphere_center, cylinder2_axis_normalized)
                    
            #         # if 0 <= projection_length <= cylinder2_height:
            #         #     dataset.append([theta,theta2,theta3,1])
                 
            #     else:
            #         distance3 = self.distance_point_to_line_segment(sphere_center, cylinder2_start, cylinder2_end)
            #         cylinder3_axis = cylinder3_end - cylinder3_start
            #         cylinder3_height = np.linalg.norm(cylinder3_axis)
            #         cylinder3_axis_normalized = cylinder3_axis / cylinder3_height
            #         vector_to_sphere_center = sphere_center - cylinder3_start
            #         vector_to_sphere_end = sphere_center - cylinder2_end
            #         projection_length = np.dot(vector_to_sphere_center, cylinder3_axis_normalized)
            #         limit_length = np.dot(vector_to_sphere_end, cylinder3_axis_normalized)
            #         if distance3 < (cylinder_radius + sphere_radius) and 0 <= projection_length <= cylinder3_height+limit_length:
            #             dataset.append([theta,theta2,theta3,1])
            #             # Further check if the sphere center is within the height of the cylinder
            #             # cylinder3_axis = cylinder3_end - cylinder3_start
            #             # cylinder3_height = np.linalg.norm(cylinder3_axis)
            #             # cylinder3_axis_normalized = cylinder3_axis / cylinder3_height
                        
            #             # vector_to_sphere_center = sphere_center - cylinder3_start
            #             # projection_length = np.dot(vector_to_sphere_center, cylinder3_axis_normalized)
                        
            #             # if 0 <= projection_length <= cylinder3_height:
            #             #     dataset.append([theta,theta2,theta3,1])
                        
            #         else:
            #             dataset.append([theta,theta2,theta3,-1])
        # return dataset

    # def collision_check(self, theta_data):
    #     dataset = []
    #     accumulated_time = 0.0
        
    #     start = time.time()
    #     print('collision checking..')
    #     for l in range(len(theta_data)):
            
    #         theta = theta_data[l][0]
    #         theta2 = theta_data[l][1]
    #         theta3 = theta_data[l][2]

    #         # obstacle(sphere)
    #         # obs_p = np.array([5., 2., 10.])
    #         # obs_r = 3
    #         # obs = colliders.Sphere(obs_p, obs_r)
    #         obs = self.obstacle(4., 4., 4., 2)

    #         # if you want to know distance with link2 and obstacle
    #         #dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(L2, obs)

    #         # collision check
    #         # s = np.degrees(theta)
    #         # d = np.degrees(theta2)
    #         # f = np.degrees(theta3)
    #         L1 ,L2, L3 = self.Arm_with_Mobile(theta, theta2, theta3)
    #         check_collision_L3 = gjk.gjk_intersection(L3,obs)
    #         if check_collision_L3 is True:
    #             dataset.append([theta,theta2,theta3,1])
    #             # collision.append([s,d,f])
    #             # result.append(1)
    #         else:
    #             check_collision_L2 = gjk.gjk_intersection(L2,obs)
    #             if check_collision_L2 is True:
    #                 dataset.append([theta,theta2,theta3,1])
    #                 # collision.append([s,d,f])
    #                 # result.append(1)
    #             else:
    #                 check_collision_L1 = gjk.gjk_intersection(L1,obs)
    #                 if check_collision_L1 is True:
    #                     dataset.append([theta,theta2,theta3,1])
    #                     # collision.append([s,d,f])
    #                     # result.append(1)
    #                 else:  
    #                     dataset.append([theta,theta2,theta3,-1])
    #                     # no_collision.append([s,d,f])
    #                     # result.append(-1)

    #     end = time.time()
    #     accumulated_time += end - start
    #     return dataset