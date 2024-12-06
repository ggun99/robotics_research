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
    # (노이즈를 줄이기 위해) 가우시안 블러 적용
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # 허프 변환(원 검출)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                            param1=50, param2=30, minRadius=20, maxRadius=80)
    dst = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # dst_origin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv1 = cv2.cvtColor(dst, cv2.COLOR_RGB2HSV)
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
            
            # RealSense 카메라 초기화
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)

            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            # depth_map = cv2.imread(depth_image, cv2.IMREAD_ANYDEPTH).astype(float)
            # depth = depth_map[closest_circle[1],closest_circle[0]]
            # 픽셀 좌표 설정 
            x = closest_circle[0]
            y = closest_circle[1]

            # 깊이 값 얻기
            depth = depth_frame.get_distance(x, y)

            # 3D 좌표 얻기
            depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
            pixel = [x, y]
            depth_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, pixel, depth)

            # # 카메라 파라미터 (실제 카메라의 파라미터로 대체해야 함)
            # fx = 387.122  # x 방향 초점 길이
            # fy = 387.122  # y 방향 초점 길이
            # cx = 325.950  # x 방향 중점 좌표
            # cy = 238.744  # y 방향 중점 좌표
            # z = depth
            # x_3d = (x - cx) * z / fx
            # y_3d = (y - cy) * z / fy

            # stamp_3d = np.array(x_3d,y_3d,z)
            # print(stamp_3d)
        return depth_point
    else:
        print("No circles detected.")
    return dst


cap = cv2.VideoCapture(4)
# depth_cap = cv2.VideoCapture(2)

while(True):
    ret, frame = cap.read()
    # ret2, frame2 = depth_cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print(detect_circles_and_stamp(frame))
   

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
##################################################