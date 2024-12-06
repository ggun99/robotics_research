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
                            param1=50, param2=30, minRadius=20, maxRadius=80)

    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dst_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv1 = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    # 원의 중점에 점, Stamped, Empty 확인
    if circles is not None:
        number_of_circles = circles.shape[1]
        print(f"Detected circles: {number_of_circles}")
        circles = np.round(circles[0, :]).astype("int")
        closest_circle = find_closest_circle(hsv1, circles, 80.0)
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

        return closest_circle
    else:
        print("No circles detected.")
    return dst


def move_RB730(self, pose):
    mat_e2c = np.array([[0.0,  0.0, 1.0,    0.0],
                        [0.0, -1.0, 0.0, -0.027],
                        [1.0,  0.0, 0.0,  0.062],
                        [0.0,  0.0, 0.0,    1.0]])
    
    current_tcp = cobot.GetCurrentTCP()
    R = np.array([[1.0,0.0,0.0],[0.0,np.cos(current_tcp[3]),-np.sin(current_tcp[3])],[0.0,np.sin(current_tcp[3]),np.cos(current_tcp[3])]])*np.array([[np.cos(current_tcp[4]),0.0,np.sin(current_tcp[4])],[0.0,1.0,0.0],[-np.sin(current_tcp[4]),0.0,np.cos(current_tcp[4])]])*np.array([[np.cos(current_tcp[5]),-np.sin(current_tcp[5]),0.0],[np.sin(current_tcp[5]),np.cos(current_tcp[5]),0.0],[0.0,0.0,1.0]])
    mat_b2e = np.array([[R[0,0],R[0,1],R[0,2],current_tcp[0]],[R[1,0],R[1,1],R[1,2],current_tcp[1]],[R[2,0],R[2,1],R[2,2],current_tcp[2]],[0.0,0.0,0.0,1.0]])
    mat_c2s = np.array([self.closest_circle[0],self.closest_circle[1],0.2,1.0])
    mat_c2s_T = np.transpose(mat_c2s)
    mat_b2s = mat_b2e*mat_e2c*mat_c2s_T
    #  cobot.MoveL(mat_b2s[0,0],mat_b2s[1,0],mat_b2s[2,0],0.0,0.0,0.0,-1.0,-1.0)



#######################ROBOT#####################
cobot.ToCB('192.168.0.201')
e = None
while e is None:
    if cobot.GetCurrentCobotStatus() is cobot.COBOT_STATUS.IDLE:
        e = True
        break
    else:
        e = None
        continue

cobot.SetBaseSpeed(1.0)
# Initialize (activate) the cobot.
cobot.CobotInit()
cobot.SetProgramMode(cobot.PG_MODE.REAL)
time.sleep(2)
##################################################




####################CAMERA#######################
cap = cv2.VideoCapture(4)
while(True):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(detect_circles_and_stamp(frame))

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
##################################################