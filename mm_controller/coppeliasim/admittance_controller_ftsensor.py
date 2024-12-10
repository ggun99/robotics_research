from cv2 import threshold
import usb
import can
from matplotlib import pyplot as plt
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
from admittance_controller import AdmittanceController


class AFT20D15:
    def __init__(self, mode) -> None:
        if mode == "usb":
            self.dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
            self.bus = can.Bus(interface="gs_usb",
                               channel=self.dev.product,
                               bus=self.dev.bus,
                               address=self.dev.address,
                               bitrate=1000000)  # sensor is running on 1000kbps
        if mode == "robotell":
            self.bus = can.Bus(interface="robotell",
                               channel="/dev/ttyUSB1@115200",
                               rtscts=True,
                               bitrate=1000000)
        if mode == "socket":
            self.bus = can.Bus(channel="can0", interface="socketcan")
        self.forceid = int(0x01A)
        self.torqueid = int(0x01B)
    def byte_to_output(self, bytearray):
        intar = list(bytearray)
        xout = intar[0] * 256 + intar[1]
        yout = intar[2] * 256 + intar[3]
        zout = intar[4] * 256 + intar[5]
        return [xout, yout, zout]
    def get_force(self, data: list):
        return [data[0] / 1000.0 - 30.0,
                data[1] / 1000.0 - 30.0,
                data[2] / 1000.0 - 30.0]
    def get_torque(self, data: list):
        return [data[0] / 100000.0 - 0.3,
                data[1] / 100000.0 - 0.3,
                data[2] / 100000.0 - 0.3]
    def receive(self):
        ft = [None] * 6
        for _ in range(2):
            rxmsg: can.Message = self.bus.recv(timeout=1)
            if rxmsg is not None:
                databytes = list(rxmsg.data)
                dataints = self.byte_to_output(databytes)
                canid = rxmsg.arbitration_id
                if canid == self.forceid:
                    force = self.get_force(dataints)
                    ft[0] = force[0]
                    ft[1] = force[1]
                    ft[2] = force[2]
                if canid == self.torqueid:
                    torque = self.get_torque(dataints)
                    ft[3] = torque[0]
                    ft[4] = torque[1]
                    ft[5] = torque[2]
        return ft
    def shutdown(self):
        self.bus.shutdown()

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

# Define system parameters
M = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Mass matrix
B = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])  # Damping matrix
K = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])  # Stiffness matrix
dt = 0.001  # Time step

# Initialize controller
admittance_controller = AdmittanceController(M, B, K)

# Filter requirements.
cutoff = 5.0  # 저역통과 필터의 컷오프 주파수
fs = 60.0     # 프레임 속도 (초당 프레임)
order = 3     # 필터 차수

time_data = []
force_data = [[], [], []]
start_t = time.time()
update_interval = 0.001  # 1ms aidin sensor's fps

if __name__ == "__main__":
    sensor = AFT20D15(mode="robotell")
    try:
        print("Start receiving messages")
        while True:
            ft = sensor.receive()
            # print(f"> ft: {ft}")
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

            print('rmse', np.mean(force_data[0]), np.mean(force_data[1]), np.mean(force_data[2]))
            threshold = 0.6

            if np.abs(ft[0]+3.8) < threshold:
                ft[0] = 0
            if np.abs(ft[1]-1.9) < threshold:
                ft[1] = 0
            if np.abs(ft[2]+10.8) < threshold:
                ft[2] = 0
            force_torque_input = np.array([ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]]) # [Fx, Fy, Fz, Tx, Ty, Tz]

            # GUI 이벤트 처리
            app.processEvents()

            delta_position = admittance_controller.update(force_torque_input)
            if delta_position[0]<0.001:
                delta_position[0]=0
            if delta_position[1]<0.001:
                delta_position[1]=0
            if delta_position[2]<0.001:
                delta_position[2]=0
            print(f"==>> position: {delta_position}")

            # 업데이트 간격 대기
            time.sleep(update_interval)
        
    except KeyboardInterrupt:
        sensor.shutdown()
        print("Stopped script")
        # 그래프 종료 후 상호작용 모드 끄기
        plt.ioff()
        plt.show()
    # for _ in range(2):
    #     data = sensor.bus.recv(1)
    #     print(f"> data: {data}")
    #     # dataarry = np.array(data.data, dtype=np.uint)
    #     # print(f"> dataarry: {dataarry}")
    #     # dout = np.array(sensor.byte_to_output(dataarry))
    #     # print(f"> dout: {dout}")
    #     # force = dout/1000 -30
    #     # print(f"> force: {force}")
    #     # torque = dout/100000 - 0.3
    #     # print(f"> torque: {torque}")
    # sensor.shutdown()







