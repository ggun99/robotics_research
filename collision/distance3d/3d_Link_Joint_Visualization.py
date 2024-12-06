"""
===================================
Distance between cylinders with GJK
===================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from pytransform3d.transformations import transform_from
import pytransform3d.rotations as pyrot
from distance3d import gjk, colliders
from distance3d import random
from distance3d import plotting
from scipy.spatial.transform import Rotation as R

random_state = np.random.RandomState(1)
# cylinder2origin, radius, length = random.rand_cylinder(random_state, 0.5, 1.0, 0.0)
L1_len = 10
L2_len = 5
L3_len = 3
radius = 1
theta = np.pi/2#90
phi = np.pi/3
gamma = np.pi/6

# def R_2_H(rotation_matrix, position_vector):
#     homogeneous_matrix = np.eye(4)
#     homogeneous_matrix[:3,:3] = rotation_matrix
#     homogeneous_matrix[:3,3] = position_vector
#     return homogeneous_matrix

ax = ppu.make_3d_axis(ax_s=2)
ax.set(xlim=(-10., 20.), ylim=(-10., 10.), zlim=(-5., 20.))
center1 = np.array([1,0,L1_len])
# L1_2_origin_E = R.from_euler('z',[theta],degrees=True)
# L1_2_origin_M = L1_2_origin_E.as_matrix()
# L1_2_origin = transform_from(L1_2_origin_M, p=np.array([0.,0.,L1_len/2]))

# mobile_box = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,-1.5],[0.,0.,0.,1.]])
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

p=np.array([1.,0.,L1_len/2])
a = np.array([0.0, 0.0, 1.0, theta])
L1_2_origin = transform_from(pyrot.matrix_from_axis_angle(a),p)


# L2_2_L1_E = R.from_euler('y',[phi],degrees=True)
# L2_2_L1_M = L2_2_L1_E.as_matrix()
# L2_2_L1 = transform_from(L2_2_L1_M, p=np.array([-L2_len/2*np.sin(phi),0.,L1_len+L2_len/2*np.cos(phi)]))

p2=np.array([L2_len/2*np.sin(phi)+1,0.,L1_len+L2_len/2*np.cos(phi)])
a2=np.array([0.,1.,0.,phi])
L2_2_L1 = transform_from(pyrot.matrix_from_axis_angle(a2),p2)

p3 = np.array([1+L2_len*np.sin(phi)+L3_len/2*np.sin(phi+gamma),0.,L1_len+L2_len*np.cos(phi)+L3_len/2*np.cos(phi+gamma)])
a3 = np.array([0.,1.,0.,gamma])
L3_2_L2 = transform_from(pyrot.matrix_from_axis_angle(a3)@pyrot.matrix_from_axis_angle(a2)@pyrot.matrix_from_axis_angle(a),p3)


accumulated_time = 0.0
for i in range(2):
    #cylinder2origin2, radius2, length2 = random.rand_cylinder(random_state)
    start = time.time()
    c1 = colliders.Cylinder(L1_2_origin, radius, L1_len)
    J1 = colliders.Sphere(center1, radius)
    c2 = colliders.Cylinder(L2_2_L1, radius, L2_len)
    center2 = np.array([1+L2_len*np.sin(phi),0.,L1_len+L2_len*np.cos(phi)])
    J2 = colliders.Sphere(center2, radius)
    c3 = colliders.Cylinder(L3_2_L2, radius, L3_len)
    dist, closest_point_cylinder, closest_point_cylinder2, _ = gjk.gjk(c1, c2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    # obs_cyl = np.array([[-1.04512543e-01, -1.82427323e-03, -9.94521895e-01, -2.48630474e+00],
    #           [-1.74524064e-02,  9.99847695e-01,  0.00000000e+00,  0.00000000e+00],
    #           [ 9.94370425e-01,  1.73568003e-02, -1.04528463e-01,  9.73867884e+00],
    #           [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # obs_cyl = np.array([[1.,0.,0.,3.],[0.,1.,0.,2.],[0.,0.,1.,7.],[0.,0.,0.,1.]])
    # obs_cyl_size = radius
    # obs = colliders.Cylinder(obs_cyl,obs_cyl_size, 6.)
    obs_box = np.array([[1.,0.,0.,3.5],[0.,1.,0.,0.5],[0.,0.,1.,8.5],[0.,0.,0.,1.]])
    obs_box_size = np.array([3.,3.,3.])
    obs = colliders.Box(obs_box,obs_box_size)
    if i > 1:
        continue
    plotting.plot_segment(
        ax, closest_point_cylinder, closest_point_cylinder2, c="k", lw=1)
    ppu.plot_box(ax= ax,size=mobile_box_size,A2B=mobile_2_origin,wireframe=False, alpha= 0.2)
    ppu.plot_cylinder(ax, A2B=L2_2_L1, radius=radius, length=L2_len,
                      wireframe=False, color="b", alpha=0.2)
    # pj=np.array([0.,0.,L1_len])
    ppu.plot_cylinder(ax, A2B=L3_2_L2, radius=radius, length=L3_len,
                      wireframe=False, color="b", alpha=0.2)
    ppu.plot_sphere(ax,radius,center1)
    ppu.plot_sphere(ax,radius,center2)
    ppu.plot_box(ax,obs_box_size,obs_box,wireframe=False, color="r", alpha=0.2)
    # ppu.plot_cylinder(ax, A2B=obs_cyl, radius=obs_cyl_size, length=6.,
    #                   wireframe=False, color="r", alpha=0.2)
print(f"{accumulated_time=}")

ppu.plot_cylinder(ax, A2B=L1_2_origin, radius=radius, length=L1_len,
                  wireframe=False,color="b", alpha=0.5)
ppu.plot_cylinder(ax, A2B=mobile_wheel_2_box, radius=radius*0.8, length=5.0,
                      wireframe=False, alpha=1.)
ppu.plot_cylinder(ax, A2B=mobile_wheel_2_box2, radius=radius*0.8, length=5.0,
                      wireframe=False, alpha=1.)
ppu.plot_sphere(ax,radius*0.8,np.array([5.,-2.5,-3.3]))
ppu.plot_sphere(ax,radius*0.8,np.array([-3.,-2.5,-3.3]))
plt.show()