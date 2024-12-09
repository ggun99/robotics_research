import numpy as np
from aidin_test import AFT20D15 
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from matplotlib import pyplot as plt

class AdmittanceController:
    def __init__(self, M, B, K, dt):
        """
        Initialize the Admittance Controller.

        Parameters:
            M: Diagonal mass matrix (6x6) [kg]
            B: Diagonal damping matrix (6x6) [Ns/m]
            K: Diagonal stiffness matrix (6x6) [N/m]
            dt: Time step [s]
        """
        self.M = M
        self.B = B
        self.K = K
        self.dt = dt

        # State variables
        self.position = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.velocity = np.zeros(6)  # [vx, vy, vz, omega_roll, omega_pitch, omega_yaw]

    def update(self, force_torque):
        """
        Update the position and velocity based on the force-torque input.

        Parameters:
            force_torque: External force-torque vector (6x1) [N, Nm]

        Returns:
            position: Updated position (6x1) [m, rad]
        """
        # Calculate acceleration: a = M^(-1) * (F - B*v - K*x)
        acc = np.linalg.inv(self.M) @ (force_torque - self.B @ self.velocity - self.K @ self.position)

        # Update velocity: v = v + a * dt
        self.velocity += acc * self.dt

        # Update position: x = x + v * dt
        self.position += self.velocity * self.dt

        return self.position


# Define system parameters
M = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Mass matrix
B = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])  # Damping matrix
K = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])  # Stiffness matrix
dt = 0.001  # Time step

# PyQtGraph 애플리케이션 생성
app = QtWidgets.QApplication([])

# 창 및 그래프 초기화
win = pg.GraphicsLayoutWidget(show=True, title="Real-Time Plot")
win.resize(800, 400)
win.setWindowTitle('Real-Time Plot')

# 플롯 추가
plot = win.addPlot(title="Real-Time Data")
curve = plot.plot(pen='y')  # 노란색 선
plot.setLabel('left', 'Amplitude')
plot.setLabel('bottom', 'Time', 's')
plot.addLegend()  # 범례 추가

# 3개의 곡선 추가
curve1 = plot.plot(pen='r', name="Data 1")  # 빨간 선
curve2 = plot.plot(pen='g', name="Data 2")  # 초록 선
curve3 = plot.plot(pen='b', name="Data 3")  # 파란 선
# Initialize controller
# admittance_controller = AdmittanceController(M, B, K, dt)

time_data = []
force_data = [[], [], []]
start_t = time.time()
update_interval = 0.001  # 1ms aidin sensor's fps

# Example usage
if __name__ == "__main__":
    sensor = AFT20D15(mode="robotell")
    try:
        print("Start receiving messages")
        while True:
            ft = sensor.receive()
            # print(f"> ft: {ft}")
            
            # force_torque_input = np.array([ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]]) # [Fx, Fy, Fz, Tx, Ty, Tz]
            cur_t = time.time()
            t = cur_t - start_t
            time_data.append(t)
            force_data[0].append(ft[0])
            force_data[1].append(ft[1])
            force_data[2].append(ft[2])
            # 곡선 업데이트
            curve1.setData(time_data, force_data[0])
            curve2.setData(time_data, force_data[1])
            curve3.setData(time_data, force_data[2])

            # GUI 이벤트 처리
            app.processEvents()

            # 업데이트 간격 대기
            # time.sleep(update_interval)

            # position = admittance_controller.update(force_torque_input)
            # print(f"==>> position: {position}")

    except KeyboardInterrupt:
        sensor.shutdown()
        print("Stopped script")
        # 그래프 종료 후 상호작용 모드 끄기
        plt.ioff()
        plt.show()
