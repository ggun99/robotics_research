import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# 데이터 초기화
x_data = []
y1_data = []  # 첫 번째 데이터
y2_data = []  # 두 번째 데이터
y3_data = []  # 세 번째 데이터

# 그래프 초기 설정
fig, ax = plt.subplots()
line1, = ax.plot([], [], label="y1: sin(x)", color="red")    # 첫 번째 라인
line2, = ax.plot([], [], label="y2: cos(x)", color="blue")   # 두 번째 라인
line3, = ax.plot([], [], label="y3: tan(x)", color="green")  # 세 번째 라인
# ax.set_xlim(0, 10)
# ax.set_ylim(-1, 1)

# 초기화 함수
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    return line1, line2, line3

# 업데이트 함수
def update(frame):
    x_data.append(frame * 0.1)   # 시간
    y1_data.append(np.sin(frame))
    y2_data.append(np.cos(frame))
    y3_data.append(np.tan(frame)) # if abs(np.tan(frame)) < 2 else np.nan)  # force values 너무 큰 값 제외
    
    line1.set_data(x_data, y1_data)
    line2.set_data(x_data, y2_data)
    line3.set_data(x_data, y3_data)

    return line1, line2, line3

def frame_generator():
    x = 0
    while True:
        yield x
        x += 0.05

# 애니메이션 설정
ani = FuncAnimation(fig, update, frames=frame_generator, init_func=init, blit=True, interval=100)

plt.show()