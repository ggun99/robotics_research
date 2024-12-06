from rainbow import cobot
import time
import numpy as np
import math
import matplotlib.pyplot as plt

cobot.ToCB('192.168.0.201')
e = None
while e is None:
    if cobot.GetCurrentCobotStatus() is cobot.COBOT_STATUS.IDLE:
        e = True
        break
    else:
        e = None
        continue
cobot.SetBaseSpeed(50.0)
# Initialize (activate) the cobot.
cobot.CobotInit()
cobot.SetProgramMode(cobot.PG_MODE.REAL)
time.sleep(1)

sin_list = []
act_list = []

j1 = 80.0
j2 = 3.49
j3 = -80.33
j4 = 1.79
j5 = -96.68

dj1 = 0.0
dj2 = 0.0
dj3 = 0.0
dj4 = 0.0
dj5 = 0.0
dj6 = 0.0

dx = 0.0
dy = 0.0
dz = 0.0
drx = 0.0
dry = 0.0
drz = 0.0


t1 = 0.002
t2 = 0.021
gain = 0.1
alpha = 0.5
times = 0.0
freq = np.pi/6


while times < 30.0:
    j6 = 60.0 + 15.0*np.sin(freq*times)
    # j6 = 0.0
    sin_list.append(j6)

    joint = cobot.GetCurrentJoint()
    act_list.append(joint.j5)

    point = cobot.GetCurrentTCP()
    # isTrue = cobot.ServoL(point.x+1.0, point.y, point.z, point.rx, point.ry, point.rz, t1, t2, gain, alpha)
    # isTrue = cobot.ServoJ(j1, j2, j3, j4, j5, j6, t1, t2, gain, alpha)

    # gain=1.0
    isTrue = cobot.SpeedJ(dj1, dj2, dj3, dj4, dj5, dj6+0.01, t1, t2, gain, alpha)

    
    # isTrue = cobot.SpeedL(dx+0.1, dy, dz, drx, dry, drz, t1, t2, gain, alpha)

    times += 0.3
    print(f"==>> times: {times}, {point.x}, {point.y}, {point.z}")
    

sin_list = np.array(sin_list)
print(f"==>> sin_list.shape: {sin_list.shape}")
act_list = np.array(act_list)
print(f"==>> act_list.shape: {act_list.shape}")

tt = np.linspace(0,1,len(sin_list))

plt.plot(tt,sin_list,'g',label='sin')
plt.plot(tt,act_list,'k',label='actual')
plt.legend()
plt.show()