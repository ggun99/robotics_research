import numpy as np

class Jacobian():
    def __init__(self):
        pass
    def jacobian(self, j1_r, j2_r, j3_r, j4_r, j5_r, j6_r):
        # UR5e Joint value (radian)
        j = np.array([float(j1_r), float(j2_r), float(j3_r), float(j4_r), float(j5_r), float(j6_r)])

        # UR5e DH parameters
        a = np.array([0., -0.425, -0.392, 0., 0., 0.])
        d = np.array([0.1625,0.,0.,0.1333,0.0997,0.0996])
        # d = np.array([0.1625, 0., 0., 0.1333, 0.0997, 0.0996])

        alpha = np.array([np.pi/2, 0., 0., np.pi/2, -np.pi/2, 0.])
        # Jacobian matrix
        J11 = d[5]*(np.cos(j[0])*np.cos(j[4]) + np.cos(j[1] + j[2] +j [3])*np.sin(j[0])*np.sin(j[4])) + d[3]*np.cos(j[0]) - a[1]*np.cos(j[1])*np.sin(j[0]) - d[4]*np.sin(j[1]+j[2]+j[3])*np.sin(j[0]) - a[2]*np.cos(j[1])*np.cos(j[2])*np.sin(j[0]) + a[2]*np.sin(j[0])*np.sin(j[1])*np.sin(j[2])
        J12 = -np.cos(j[0])*(a[1]*np.sin(j[1]) - d[4]*np.cos(j[1]+j[2]+j[3]) + a[2]*np.cos(j[1])*np.sin(j[2]) + a[2]*np.cos(j[2])*np.sin(j[1]) - d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J13 = np.cos(j[0])*(d[4]*np.cos(j[1]+j[2]+j[3]) - a[2]*np.cos(j[1])*np.sin(j[2]) - a[2]*np.cos(j[2])*np.sin(j[1]) + d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J14 = np.cos(j[0])*(d[4]*np.cos(j[1]+j[2]+j[3]) + d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J15 = -d[5]*(np.sin(j[0])*np.sin(j[4]) + np.cos(j[1]+j[2]+j[3])*np.cos(j[0])*np.cos(j[4]))
        J16 = 0
        J21 = d[5]*(np.cos(j[4])*np.sin(j[0]) - np.cos(j[1]+j[2]+j[3])*np.cos(j[0])*np.sin(j[4])) + d[3]*np.sin(j[0]) + a[1]*np.cos(j[0])*np.cos(j[1]) + d[4]*np.sin(j[1]+j[2]+j[3])*np.cos(j[0]) + a[2]*np.cos(j[0])*np.cos(j[1])*np.cos(j[2]) - a[2]*np.cos(j[0])*np.sin(j[1])*np.sin(j[2])
        J22 = -np.sin(j[0])*(a[1]*np.sin(j[1]) - d[4]*np.cos(j[1]+j[2]+j[3]) + a[2]*np.cos(j[1])*np.sin(j[2]) + a[2]*np.cos(j[2])*np.sin(j[1]) - d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J23 = np.sin(j[0])*(d[4]*np.cos(j[1]+j[2]+j[3]) - a[2]*np.cos(j[1])*np.sin(j[2]) - a[2]*np.cos(j[2])*np.sin(j[1]) + d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J24 = np.sin(j[0])*(d[4]*np.cos(j[1]+j[2]+j[3]) + d[5]*np.sin(j[1]+j[2]+j[3])*np.sin(j[4]))
        J25 = d[5]*(np.cos(j[0])*np.sin(j[4]) - np.cos(j[1]+j[2]+j[3])*np.cos(j[4])*np.sin(j[0]))
        J26 = 0
        J31 = 0
        J32 = d[4]*(np.cos(j[1] + j[2])*np.sin(j[3]) + np.sin(j[1] + j[2])*np.cos(j[3])) + a[2]*np.cos(j[1] + j[2]) + a[1]*np.cos(j[1]) + d[5]*np.sin(j[4])*(np.sin(j[1] + j[2])*np.sin(j[3]) - np.cos(j[1] + j[2])*np.cos(j[3]))
        J33 = a[2]*np.cos(j[1] + j[2]) + d[4]*np.sin(j[1] + j[2] + j[3]) - d[5]*np.cos(j[1] + j[2] + j[3])*np.sin(j[4])
        J34 = d[4]*np.sin(j[1] + j[2] + j[3]) - d[5]*np.cos(j[1] + j[2] + j[3])*np.sin(j[4])
        J35 = -d[5]*np.sin(j[1] + j[2] + j[3])*np.cos(j[4])
        J36 = 0
        J41 = 0
        J42 = np.sin(j[0])
        J43 = np.sin(j[0])
        J44 = np.sin(j[0])
        J45 = np.sin(j[1] + j[2] + j[3])*np.cos(j[0])
        J46 = np.cos(j[4])*np.sin(j[0]) - np.cos(j[1] + j[2] + j[3])*np.cos(j[0])*np.sin(j[4])
        J51 = 0
        J52 = -np.cos(j[0])
        J53 = -np.cos(j[0])
        J54 = -np.cos(j[0])
        J55 = np.sin(j[1] + j[2] + j[3])*np.sin(j[0])
        J56 = - np.cos(j[0])*np.cos(j[4]) - np.cos(j[1] + j[2] + j[3])*np.sin(j[0])*np.sin(j[4])
        J61 = 1
        J62 = 0
        J63 = 0
        J64 = 0
        J65 = -np.cos(j[1]+j[2]+j[3])
        J66 = -np.sin(j[1]+j[2]+j[3])*np.sin(j[4])
        J = np.array([[J11, J12, J13, J14, J15, J16], [J21, J22, J23, J24, J25, J26], [J31, J32, J33, J34, J35, J36], [J41, J42, J43, J44, J45, J46], [J51, J52, J53, J54, J55, J56], [J61, J62, J63, J64, J65, J66]])

        return J