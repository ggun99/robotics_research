import usb
import can

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
if __name__ == "__main__":
    sensor = AFT20D15(mode="robotell")
    try:
        print("Start receiving messages")
        while True:
            ft = sensor.receive()
            print(f"> ft: {ft}")
    except KeyboardInterrupt:
        sensor.shutdown()
        print("Stopped script")
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







