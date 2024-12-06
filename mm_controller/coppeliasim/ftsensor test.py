import sys
import os
import numpy as np
from matplotlib import pyplot as plt
wd = os.path.abspath(os.getcwd())
sys.path.append(str(wd))
sys.path.append("/home/airlab/Documents/CoppeliaSim_Edu_V4_4_0_rev0_Ubuntu20_04/programming/zmqRemoteApi/clients/python")
from zmqRemoteApi import RemoteAPIClient

print('Program started')

client = RemoteAPIClient()
sim = client.getObject('sim')


client.step()
sim.startSimulation()

ft_sensor = sim.getObjectHandle('/UR5/joint/joint/joint/joint/joint/joint/link/connection')


# 초기 설정
# time_data, force_x_data, force_y_data, force_z_data = [], [], [], []  # 데이터를 저장할 리스트
# plt.ion()  # Interactive 모드 활성화 (실시간 업데이트 가능)
fig, ax = plt.subplots()

lines = []
for _ in range(3):  # y 값이 3개라서 3개의 선 생성
    line, = ax.plot([], [], lw=2)  # 각각의 빈 선 객체를 초기화
    lines.append(line)

# x 데이터와 y 데이터 초기화
x_data = []
y_data_list = [[], [], []]  # 각 선에 대한 y 데이터 리스트 초기화
x_max = 10

ax.set_title("Real-time Graph with While Loop")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")

# 그래프 업데이트 설정
ax.set_xlim(0, 10)  # x축 초기 범위
ax.set_ylim(-10.0, 10.0)  # y축 범위 고정 (필요시 변경 가능)

while True:
    t = sim.getSimulationTime()
    x_data.append(t) 
    print(f"==>> t: {t}")
    res, force, torque = sim.readForceSensor(ft_sensor)
    # 각 y 데이터 업데이트
    y_data_list[0].append(force[0])
    y_data_list[1].append(force[1])
    y_data_list[2].append(force[2])
    # x축이 초과되면 x축 범위 확장
    if t > x_max:
        x_max += 10  # x축 최대값 10씩 증가
        ax.set_xlim(0, x_max)
        # 각 선에 대해 데이터를 업데이트
    for line, y_data in zip(lines, y_data_list):
        line.set_data(x_data, y_data)
    
    # 그래프 업데이트
    plt.pause(0.01)
    print(res, force, torque)
    # 애니메이션 생성
    