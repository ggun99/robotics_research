import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation as R
from tf_trnasformations import quaternion_from_euler, euler_from_quaternion

# r = R.from_quat([0.976, -0.205, 0.0183, 0.07275])  #y,x,w,z
# r = R.from_quat([0.0183, -0.205, 0.976, 0.07275])  #x,y,z,w

r = R.from_quat([-0.205, 0.976, 0.07275, 0.0183])  #x,y,z,w

x=-0.20236904554519108
y=0.9764306724571273
z=0.07276186288682619
w=0.01832000543911521
# euler = r.as_euler('xyz', degrees=False)
# print(euler)
norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
print("norm:", norm)