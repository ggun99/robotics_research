import rtde_control
import rtde_receive
## INIT ##
# 로봇의 IP 주소 설정
ROBOT_IP = '192.168.0.3'  # 로봇의 IP 주소로 바꿔주세요

# RTDE 수신 객체 생성
rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

# 힘/토크 센서 값 읽기
wrench = rtde_r.getActualTCPForce()
raw_wrench = rtde_r.getFtRawWrench()
while True:  # 6D 벡터: [Fx, Fy, Fz, Tx, Ty, Tz]
    print("Force/Torque values:", wrench)
    print("Raw Force/Torque values:", raw_wrench)