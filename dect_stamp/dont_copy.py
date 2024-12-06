import numpy as np
import cv2
from rainbow import cobot
import time
import pyrealsense2 as rs

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

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,101,6)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]

    # (노이즈를 줄이기 위해) 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # 허프 변환(원 검출)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                            param1=50, param2=30, minRadius=20, maxRadius=80)
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # dst_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv1 = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
    # 원의 중점에 점, Stamped, Empty 확인
    if True:
        number_of_circles = circles.shape[1]
        print(f"Detected circles: {number_of_circles}")
        circles = np.round(circles[0, :]).astype("int")
        closest_circle = find_closest_circle(hsv1, circles, 0.0)
        if True:
            won = "Empty"
            won = won + " (" + str(closest_circle[0]) + ", " + str(closest_circle[1]) + ") "
            # # 중심에 원을 그리기
            cv2.circle(dst, (closest_circle[0], closest_circle[1]), 1, (0, 255, 0), -1)
            # # 원 경계를 그리기
            cv2.circle(dst, (closest_circle[0], closest_circle[1]), closest_circle[2], (0, 255, 0), 2)

            
            # 결과를 화면에 출력
            cv2.putText(dst, won, (closest_circle[0] - 10, closest_circle[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 결과
            # cv2.imshow("hsv", hsv1)
            cv2.circle(dst,(326,238),5,(0,255,0),1)

            cv2.imshow('Result Image', thresh)
            cv2.waitKey(1)
            # RealSense 카메라 초기화
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            depth_map = cv2.imread(depth_frame, cv2.IMREAD_ANYDEPTH).astype(float)
            depth = depth_map[closest_circle[1],closest_circle[0]]
            # 픽셀 좌표 설정 
            x = 0
            y = 0
            z = 0.3

            depth = depth_frame[y,x]
            # 깊이 값 얻기
            depth = depth_frame.get_distance(x, y)

            # 3D 좌표 얻기
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            pixel = [x, y]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, pixel, depth)

            # 카메라 파라미터 (실제 카메라의 파라미터로 대체해야 함)
            fx = 387.122  # x 방향 초점 길이
            fy = 387.122  # y 방향 초점 길이
            cx = 325.950  # x 방향 중점 좌표
            cy = 238.744  # y 방향 중점 좌표
            
            # z = depth[0]/1000
            x_3d = (x - cx) * z / fx
            y_3d = (y - cy) * z / fy

            stamp_3d = np.array([x_3d,y_3d,z])
            
        return stamp_3d, x, y
    else:
        print("No circles detected.")
    return image


cap = cv2.VideoCapture(2)
# depth_cap = cv2.VideoCapture(2)

# cobot.ToCB('192.168.0.201')
# e = None
# while e is None:
#     if cobot.GetCurrentCobotStatus() is cobot.COBOT_STATUS.IDLE:
#         e = True
#         break
#     else:
#         e = None
#         continue
# cobot.SetBaseSpeed(1.0)
# # Initialize (activate) the cobot.
# cobot.CobotInit()
# cobot.SetProgramMode(cobot.PG_MODE.REAL)
# time.sleep(1)


while(True):
    
    ret, frame = cap.read()
    # ret2, frame2 = depth_cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow('cap2', frame)
    # print('depth',frame2[230,320])
    # detect_circles_and_stamp(frame)
    # # print(stamp_position)
    
    # current_tcp_ = cobot.GetCurrentTCP()

    # control_gain = 0.01
    # pos_error_x = tarx.astype(float)-320.0
    # pos_error_y = tary.astype(float)-240.0
    
    # tx = current_tcp_.x - control_gain*pos_error_x
    # ty = current_tcp_.y + control_gain*pos_error_y
    # tz = current_tcp_.z
    # t_rx = current_tcp_.rx
    # t_ry = current_tcp_.ry
    # t_rz = current_tcp_.rz
    # print(tx, ty, tz)
    # # cobot.MoveL(tx, ty, tz, t_rx, t_ry, t_rz, -1.0, -1.0)
    # print(np.abs(pos_error_x)+np.abs(pos_error_y))
    # if np.abs(pos_error_x)+np.abs(pos_error_y) < 10:
    #     # cobot.MoveL(tx, ty, 100.0, t_rx, t_ry, t_rz, -1.0, -1.0)
    #     # cobot.MoveL(0.0, -200.0, 290.0, -180.0, 0.0, 180.0, -1.0,-1.0)
    #     break


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cobot.MotionPause()
    #     cobot.MoveL(0.0, -200.0, 310.0, -180.0, 0.0, 180.0, -1.0,-1)
    #     cobot.MotionHalt()
    #     break
cap.release()
cv2.destroyAllWindows()
##################################################