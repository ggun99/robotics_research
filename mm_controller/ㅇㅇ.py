import matplotlib.pyplot as plt
import numpy as np

# 그래프 초기화
fig, ax = plt.subplots()
ax.set_xlim(0, 10)  # x축 범위
ax.set_ylim(-2, 2)  # y축 범위

# 여러 선 초기화
lines = []
for _ in range(3):  # y 값이 3개이므로 선 3개 생성
    line, = ax.plot([], [], lw=2)
    lines.append(line)

# x 데이터와 y 데이터 초기화
x_data = []
y_data_list = [[], [], []]  # 각 선에 대한 y 데이터 리스트 초기화

# 실시간으로 데이터 업데이트
t = 0  # 초기 시간
dt = 0.1  # 시간 증가량

while t <= 10:  # 10초 동안 업데이트
    x_data.append(t)  # x 데이터에 시간 추가
    # 각 y 데이터 업데이트
    y_data_list[0].append(np.sin(t))
    y_data_list[1].append(np.cos(t))
    y_data_list[2].append(np.sin(2 * t))
    
    # 각 선에 대해 데이터를 업데이트
    for line, y_data in zip(lines, y_data_list):
        line.set_data(x_data, y_data)
    
    # 그래프 업데이트
    plt.pause(0.01)  # 약간의 지연을 줘서 실시간처럼 보이게
    t += dt  # 시간 증가