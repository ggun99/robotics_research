import os
import sys
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))

import numpy as np
import collision_class
import matplotlib.pyplot as plt
from robot.planar_rrr import PlanarRRR

# a = collision_class.ObjPoint3D(5, 5, 2)
# b = collision_class.ObjPoint3D(5, 5, 5)
# c = collision_class.ObjPoint2D(5, 5)

# trig = collision_class.ObjTriangle([0, 0], [10, 0], [0, 10])

# collision = collision_class.intersect_point_v_point_3d(a, b)
# collision = collision_class.intersect_triangle_v_point(trig, c)
# r = 1

# O = np.array([[0], [0]])

# P = np.array([[-6], [-6]])

# Q = np.array([[-6], [6]])

# collide = collision_class.intersect_line_v_circle(r, O, P, Q)
# print(collision)

# rec1 = collision_class.ObjRec(0, 0, 5, 5)
# rec2 = collision_class.ObjRec(10, 10, 5, 5)
# collision = collision_class.intersect_rectangle_v_rectangle(rec1, rec2)
# print(collision)


# recWithAngle = collision_class.ObjRec(1,1,1,5,angle=2)
# line = collision_class.ObjLine2D(-1,1,0,5)
# collide = collision_class.intersect_line_v_rectangle(line, recWithAngle)
# print(f"==>> collide: \n{collide}")
# line.plot()
# recWithAngle.plot()
# plt.show()



robot = PlanarRRR()
theta = np.array([0.7,0,0]).reshape(3,1)

rec1 = collision_class.ObjRec(x=1.5, y=1.25, h=0.2, w=1, angle=1)
rec2 = collision_class.ObjRec(x=1.5, y=0.5, h=0.2, w=1, angle=0.3)

linkPose = robot.forward_kinematic(theta, return_link_pos=True)
linearm1 = collision_class.ObjLine2D(linkPose[0][0], linkPose[0][1], linkPose[1][0], linkPose[1][1])
linearm2 = collision_class.ObjLine2D(linkPose[1][0], linkPose[1][1], linkPose[2][0], linkPose[2][1])
linearm3 = collision_class.ObjLine2D(linkPose[2][0], linkPose[2][1], linkPose[3][0], linkPose[3][1])

obsList = [rec1, rec2]
for i in obsList:
    col1 = collision_class.intersect_line_v_rectangle(linearm1, i)
    print(f"==>> col1: \n{col1}")
    col2 = collision_class.intersect_line_v_rectangle(linearm2, i)
    print(f"==>> col2: \n{col2}")
    col3 = collision_class.intersect_line_v_rectangle(linearm3, i)
    print(f"==>> col3: \n{col3}")

robot.plot_arm(theta, plt_basis=True)
for obs in obsList:
    obs.plot()
plt.show()