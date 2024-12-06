import numpy as np
import cv2
from rainbow import cobot
import time

def find_closest_circle(image, circles, target_hsv):
    # closest_circle = None
    min_color_difference = float('inf')
    for (cx, cy, radius) in circles:
        # 동전 영역 부분 크롭 영상 만들기
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(image.shape[1], cx + radius)
        y2 = min(image.shape[0], cy + radius)
        crop = image[y1:y2, x1:x2]  # 크롭 영상 생성
        avg_color = np.mean(crop)
        # 색상 차이 계산
        color_difference = np.abs(avg_color - target_hsv)
        # 색상 차이가 현재까지의 최소 차이보다 작으면 업데이트
        if color_difference < min_color_difference:
            min_color_difference = color_difference
            closest_circle = (cx, cy, radius)
    return closest_circle

def detect_circles_and_stamp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    # (노이즈를 줄이기 위해) 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # 허프 변환(원 검출)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                            param1=50, param2=30, minRadius=20, maxRadius=50)
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv1 = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    # 원의 중점에 점, Stamped, Empty 확인
    if circles is not None:
        number_of_circles = circles.shape[1]
        print(f"Detected circles: {number_of_circles}")
        circles = np.round(circles[0, :]).astype("int")
        closest_circle = find_closest_circle(hsv1, circles, 20.0)
        if closest_circle is not None:
            won = "Empty"
            won = won + " (" + str(closest_circle[0]) + ", " + str(closest_circle[1]) + ") "
            # 중심에 원을 그리기
            cv2.circle(dst, (closest_circle[0], closest_circle[1]), 1, (0, 255, 0), -1)
            # 원 경계를 그리기
            cv2.circle(dst, (closest_circle[0], closest_circle[1]), closest_circle[2], (0, 255, 0), 2)
            # 결과를 화면에 출력
            cv2.putText(dst, won, (closest_circle[0] - 10, closest_circle[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 결과
            # cv2.imshow("hsv", hsv1)
            cv2.imshow('Result Image', dst)
            cv2.waitKey(1)
            
            # 픽셀 좌표 설정 
            x = closest_circle[0]
            y = closest_circle[1]
            z = 0.3

            # 카메라 파라미터 (실제 카메라의 파라미터로 대체해야 함)
            fx = 604.123  # x 방향 초점 길이
            fy = 604.123  # y 방향 초점 길이
            cx = 316.114  # x 방향 중점 좌표
            cy = 234.611  # y 방향 중점 좌표
            
            # z = depth[0]/1000
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy

            stamp_3d = np.array([x_3d,y_3d,z])
            
        return stamp_3d, x, y
    else:
        print("No circles detected.")
    return

cap = cv2.VideoCapture(4)

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


while(True):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if detect_circles_and_stamp(frame) is not None:
        target_3d, tarx, tary = detect_circles_and_stamp(frame)
    else:
        continue
    current_tcp_ = cobot.GetCurrentTCP()
    control_gain = 0.01
    pos_error_x = tarx.astype(float)-316.114
    pos_error_y = tary.astype(float)-234.611
    
    tx = float(current_tcp_.x - control_gain*pos_error_x)
    ty = float(current_tcp_.y + control_gain*pos_error_y)
    tz = current_tcp_.z
    t_rx = float(current_tcp_.rx)
    t_ry = float(current_tcp_.ry)
    t_rz = float(current_tcp_.rz)

    print(tx, ty, tz)
    print(np.abs(pos_error_x)+np.abs(pos_error_y))
    # print('current',current_tcp_)
    # print(',,,',tx, ty, 145.0, t_rx, t_ry, t_rz)

    cobot.ServoL(tx, ty, 145.0, t_rx, t_ry, t_rz, 0.1, 0.05, 1.0, 0.5)

    if np.abs(pos_error_x)+np.abs(pos_error_y) < 10:
        # 초기 위치로 이동.
        cobot.MoveL(-12.09 , -400.0, 145.0, -180.0, 0.0, 180.0, 60.0,-1.0)
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
##################################################