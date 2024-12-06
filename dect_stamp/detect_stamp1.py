import numpy as np
import cv2
from rainbow import cobot
import time
import pyrealsense2 as rs

def find_closest_circle(image, circles, target_hsv):
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


def detect_square_and_stamp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일로 변환
    # (노이즈를 줄이기 위해) 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blurred, 10, 10)
    kernel = np.full((3,3), 255, np.uint8)
    canny = cv2.dilate(canny, kernel, iterations=3)

    contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntrRect = []
    for i in contours:
        epsilon = 0.03 * cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, epsilon, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            if 40 < w < 160 and w - 15 < h < w + 15:
                cntrRect.append(approx)
    print(len(cntrRect))

    rec_center_list = []

    for cnt, rect in enumerate(cntrRect):
        (x, y, w, h) = cv2.boundingRect(rect)
        rec_center_list.append([float(x+w/2), float(y+h/2)])
        std = np.std(image[y:y+h, x:x+h])
        if std > 5:
            continue
        cv2.drawContours(image, [rect], -1, (0, 255, 0), 2)
        cv2.circle(image, (int(x+w/2), int(y+h/2)), 1, (0, 255, 0), 2)
  
   
    cv2.imshow('Result Image', image)
    return rec_center_list
    

def find_2d(x,y):
    x_pix = x
    y_pix = y
    z = 285

    fx = 605.596  # x 방향 초점 길이
    fy = 604.945  # y 방향 초점 길이
    cx = 320.099  # x 방향 중점 좌표
    cy = 249.385  # y 방향 중점 좌표
    
    x_2d = -(x_pix - cx) * z / fx 
    y_2d = (y_pix - cy) * z / fy 

    offset_3d = np.array([x_2d,y_2d,z])
    return offset_3d

cap = cv2.VideoCapture(4)
# depth_cap = cv2.VideoCapture(2)
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
time.sleep(1)


while(True):
    print("---------------Init---------------")
    cobot.MoveL(0.0, -180.0, 300.0, 180.0, 0.0, -180.0, -1.0, -1.0)
    ret, frame = cap.read()
    rec_center_list = detect_square_and_stamp(frame)
    k = cv2.waitKey(1)
    if k == ord('a'):
        xy_rec = rec_center_list[0]
        print('pixel', xy_rec)
        x = xy_rec[0]
        y = xy_rec[1]
        offset_stamp = find_2d(x, y)
        # print('off', offset_stamp)
        t_stamp_x = round(offset_stamp[0] + 27. ,3)
        t_stamp_y = round(-180. + offset_stamp[1] - 60.0,3)
        t_stamp_z = 300.
        t_rx = 180.0
        t_ry = 0.0
        t_rz = -180.0
        
        print(f"Move L: {t_stamp_x, t_stamp_y, t_stamp_z, t_rx, t_ry, t_rz}")
        cobot.MoveL(t_stamp_x, t_stamp_y, t_stamp_z, t_rx, t_ry, t_rz, -1.0, -1.0)
        press_esc = 0
        time.sleep(2)
        cobot.MoveL(t_stamp_x, t_stamp_y, 100. , t_rx, t_ry, t_rz, -1.0, -1.0)
        time.sleep(1)

        
    if k == ord('s'):
        break

    

    print("----------------------------------\n\n")
   

cap.release()
cv2.destroyAllWindows()
##################################################