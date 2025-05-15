import numpy as np
import osqp
from scipy import sparse
from scipy.spatial.transform import Rotation as R

# r = R.from_quat([0.976, -0.205, 0.0183, 0.07275])  #y,x,w,z
# r = R.from_quat([0.0183, -0.205, 0.976, 0.07275])  #x,y,z,w

r = R.from_quat([-0.205, 0.976, 0.07275, 0.0183])  #x,y,z,w
euler = r.as_euler('xyz', degrees=False)
print(euler)