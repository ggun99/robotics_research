import os
import sys

sys.path.append(str(os.path.abspath(os.getcwd())))

import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.plot_utils import make_3d_axis
from spatial_transformation import RigidBodyTransformation as rbt


HcameraToRobot = rbt.ht(0.3, 0.01, 0.5) @ rbt.hry(np.pi / 2) @ rbt.hrz(-np.pi / 2)

HArucoToCamera = rbt.ht(0.0, 0.0, 2.0) @ rbt.hrx(np.pi)
HArucoToRobot = HcameraToRobot @ HArucoToCamera

HRobotDesireToAruco = rbt.ht(0.0, 0.0, 0.5) @ rbt.hrx(-np.pi / 2) @ rbt.hrz(np.pi / 2)
HRobotDesireToRobot = HArucoToRobot @ HRobotDesireToAruco

# # robot base main
# ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
# plot_transform(ax=ax, s=0.1, name="robot")
# plot_transform(ax, A2B=HcameraToRobot, s=0.1, name="camera")
# plot_transform(ax, A2B=HArucoToRobot, s=0.1, name="aruco")
# plot_transform(ax, A2B=HRobotDesireToRobot, s=0.1, name="desire robot")
# plt.tight_layout()
# plt.show()


# aruco main
HCameraToAruco = rbt.hinverse(HArucoToCamera)
HRobotToAruco = rbt.hinverse(HArucoToRobot)

HRobotDesireToAruco = rbt.ht(0.0, 0.0, 0.5) @ rbt.hrx(-np.pi / 2) @ rbt.hrz(np.pi / 2)
HRobotDesireToAruco[1, 3] = HRobotToAruco[1, 3]

ax = make_3d_axis(ax_s=1, unit="m", n_ticks=6)
plot_transform(ax=ax, s=0.1, name="aruco")
plot_transform(ax, A2B=HCameraToAruco, s=0.1, name="camera")
plot_transform(ax, A2B=HRobotDesireToAruco, s=0.1, name="desire robot")
plot_transform(ax, A2B=HRobotToAruco, s=0.1, name="robot")
plt.tight_layout()
plt.show()

HRobotToRobotDesire =  rbt.hinverse(HRobotDesireToAruco) @ HRobotToAruco
print(f"> HRobotToRobotDesire: {HRobotToRobotDesire}")
