import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray
import pyzed.sl as sl
import math
import numpy as np
import math
import cv2 as cv
import mediapipe as mp
from ultralytics import YOLO

class Vision_Hand(Node):
    def __init__(self):
        super().__init__('Vision_Hand')
        self.position = None
        self.H_s2a = None
        self.pose_publisher = self.create_publisher(Pose,'/hand_pose', 10)
        
        # 🆕 케이블 포인트들을 위한 퍼블리셔 (PoseArray 사용)
        self.cable_points_publisher = self.create_publisher(PoseArray, '/cable_points', 10)
        self.H_s2a = None
        self.timeroffset = self.create_timer(0.1, self.offset_pub)
        # Create a Camera object
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  
        # [ULTRA depth mode : Computation mode that favors edges and sharpness. Requires more GPU memory and computation power.] 
        # [PERFORMANCE for faster processing : Computation mode optimized for speed.]
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)    
        # init_params.camera_resolution = sl.RESOLUTION.VGA  # Set resolution to 640x480  HD2K: 약 1m,  HD1080: 약 0.7m, HD720: 약 0.5m (500mm), VGA: 약 0.3m (300mm)
        # init_params.camera_fps = 100 # Optional: Set FPS (e.g., 30 FPS)
        # Open the camera   
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS: #Ensure the camera has opened succesfully
            print("Camera Open : "+repr(status)+". Exit program.")
        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat(self.zed.get_camera_information().camera_configuration.resolution.width, self.zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.U8_C4)
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

        # fx, fy
        self.focal_left_x = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
        self.focal_left_y = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
        
        self.no_hand = False
        self.previous_position = None
        
        # 🆕 빨간색 점 감지를 위한 변수들
        self.cable_points = []
        self.previous_cable_points = []
        
        # 🆕 빨간색 감지 파라미터 (트랙바로 조정 가능)
        self.setup_trackbars()

    def setup_trackbars(self):
        """빨간색 감지를 위한 HSV 트랙바 설정"""
        cv.namedWindow('Red HSV Controls')
        cv.resizeWindow('Red HSV Controls', 400, 300)
        
        # HSV 범위 초기값
        cv.createTrackbar('H Min', 'Red HSV Controls', 0, 180, lambda x: None)
        cv.createTrackbar('S Min', 'Red HSV Controls', 50, 255, lambda x: None)
        cv.createTrackbar('V Min', 'Red HSV Controls', 50, 255, lambda x: None)
        cv.createTrackbar('H Max', 'Red HSV Controls', 10, 180, lambda x: None)
        cv.createTrackbar('S Max', 'Red HSV Controls', 255, 255, lambda x: None)
        cv.createTrackbar('V Max', 'Red HSV Controls', 255, 255, lambda x: None)
        
        # 최소 면적 조정
        cv.createTrackbar('Min Area', 'Red HSV Controls', 20, 200, lambda x: None)

    def get_trackbar_values(self):
        """트랙바에서 HSV 값들을 가져오기"""
        h_min = cv.getTrackbarPos('H Min', 'Red HSV Controls')
        s_min = cv.getTrackbarPos('S Min', 'Red HSV Controls')
        v_min = cv.getTrackbarPos('V Min', 'Red HSV Controls')
        h_max = cv.getTrackbarPos('H Max', 'Red HSV Controls')
        s_max = cv.getTrackbarPos('S Max', 'Red HSV Controls')
        v_max = cv.getTrackbarPos('V Max', 'Red HSV Controls')
        min_area = cv.getTrackbarPos('Min Area', 'Red HSV Controls')
        
        return (h_min, s_min, v_min), (h_max, s_max, v_max), min_area

    def detect_red_points(self, imgbgr, imgrgb):
        """빨간색 점들을 감지하여 3D 좌표를 반환"""
        # HSV 색상 공간으로 변환
        hsv = cv.cvtColor(imgbgr, cv.COLOR_BGR2HSV)
        
        # 트랙바에서 HSV 값 가져오기
        (h_min, s_min, v_min), (h_max, s_max, v_max), min_area = self.get_trackbar_values()
        
        # 빨간색 범위 정의 (트랙바 기반 + 기본 빨간색 범위)
        lower_red1 = np.array([h_min, s_min, v_min])
        upper_red1 = np.array([h_max, s_max, v_max])
        lower_red2 = np.array([160, 50, 50])  # 기본 빨간색 범위 유지
        upper_red2 = np.array([180, 255, 255])
        
        # 빨간색 마스크 생성 (트랙바 범위 + 기본 범위)
        mask1 = cv.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv.bitwise_or(mask1, mask2)
        
        # 노이즈 제거를 위한 모폴로지 연산
        kernel = np.ones((3, 3), np.uint8)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
        red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
        
        # 컨투어 찾기
        contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        cable_points_3d = []
        
        for contour in contours:
            # 작은 노이즈 필터링 (트랙바에서 설정한 최소 면적)
            area = cv.contourArea(contour)
            if area < min_area:
                continue
            
            # 중심점 계산
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 3D 좌표 계산
                err, point_cloud_value = self.point_cloud.get_value(cx, cy)
                if isinstance(point_cloud_value, (list, np.ndarray)) and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]):
                    Z = point_cloud_value[2]  # Z축 값 (깊이)
                    X = Z * (cx - imgrgb.shape[1]/2) / self.focal_left_x
                    Y = Z * (cy - imgrgb.shape[0]/2) / self.focal_left_y
                    
                    cable_points_3d.append([X, Y, Z])
                    
                    # 시각화: 빨간색 점에 초록색 원과 번호 표시
                    cv.circle(imgbgr, (cx, cy), 10, (0, 255, 0), -1)  # 더 큰 원
                    cv.circle(imgbgr, (cx, cy), 12, (255, 255, 255), 2)  # 흰색 테두리
                    
                    # 번호 표시
                    point_num = len(cable_points_3d)
                    cv.putText(imgbgr, str(point_num), (cx - 5, cy + 5), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 3D 좌표 표시
                    cv.putText(imgbgr, f"P{point_num}({X:.0f},{Y:.0f},{Z:.0f})", 
                             (cx + 15, cy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # 컨투어 경계 그리기
                    cv.drawContours(imgbgr, [contour], -1, (0, 255, 255), 2)
        
        return cable_points_3d, red_mask

    def Image_Processing(self):
        # A new image is available if grab() returns SUCCESS
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
            imgnp = self.image.get_data()

            if imgnp is None:
                print("Image not loaded. Please check the image path or URL.")
                return
            # resize_img = cv.resize(imgnp, (640, 480))  # Resize for speed

            imgrgb = cv.cvtColor(imgnp, cv.COLOR_BGRA2RGB)
            imgbgr = cv.cvtColor(imgrgb, cv.COLOR_RGB2BGR)

            # 🆕 빨간색 점 감지
            cable_points_3d, red_mask = self.detect_red_points(imgbgr, imgrgb)
            self.cable_points = cable_points_3d
            
            if len(cable_points_3d) > 0:
                print(f"Detected {len(cable_points_3d)} red cable points")
                for i, point in enumerate(cable_points_3d):
                    print(f"  Cable point {i}: X={point[0]:.3f}, Y={point[1]:.3f}, Z={point[2]:.3f}")

            # 2. Hand detection
            results = self.hands.process(imgrgb)
            hand_positions = []
            hand_dist = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]  # wrist landmark
                    x = int(wrist.x * imgrgb.shape[1])
                    y = int(wrist.y * imgrgb.shape[0])
                    
                    # 손목 시각화: 더 큰 노란색 원과 텍스트
                    cv.circle(imgbgr, (x, y), 12, (0, 255, 255), -1)  # 노란색 원
                    cv.circle(imgbgr, (x, y), 15, (255, 255, 255), 2)  # 흰색 테두리
                    cv.putText(imgbgr, "HAND", (x - 20, y - 20), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    err, point_cloud_value = self.point_cloud.get_value(x, y)
                    # Check if point_cloud_value is valid
                    if isinstance(point_cloud_value, (list, np.ndarray)) and len(point_cloud_value) >= 3 and math.isfinite(point_cloud_value[2]):
                        distance = math.sqrt(point_cloud_value[0] ** 2 +
                                            point_cloud_value[1] ** 2 +
                                            point_cloud_value[2] ** 2)
                        Z = point_cloud_value[2]  # Z축 값 (깊이)
                        X = Z * (x - 640) / self.focal_left_x
                        Y = Z * (y - 360) / self.focal_left_y
                        print(f"3D position to Camera : X : {X}, Y : {Y}, Z : {Z}")
                        hand_positions.append([X, Y, Z])
                        hand_dist.append(distance)
                        self.previous_position = [X, Y, Z]  # Update previous position
                        
                        # 손목 3D 좌표 표시
                        cv.putText(imgbgr, f"H({X:.0f},{Y:.0f},{Z:.0f})", 
                                 (x + 20, y + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    else:
                        print(f"Invalid point cloud value at ({x}, {y}). Using previous position.")
                        if self.previous_position is not None:
                            hand_positions.append(self.previous_position)
                            hand_dist.append(math.sqrt(self.previous_position[0] ** 2 +
                                                    self.previous_position[1] ** 2 +
                                                    self.previous_position[2] ** 2))
                        else:
                            print("No valid previous position available.")

                # Convert to numpy array for easier manipulation
                if hand_positions is not None:
                    hand_dist_array = np.array(hand_dist)
                    min_index = np.argmin(hand_dist_array)
                    self.position = hand_positions[min_index]
                else:
                    self.position = None
                    self.no_hand = True
                    print("No hands detected.")
            else:
                self.position = None
                self.no_hand = True
                print("No hands detected.")

            # 4. Draw person boxes
            # for x1, y1, x2, y2 in person_boxes:
            #     cv.rectangle(imgbgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 상태 정보 표시 (더 자세하게)
            hand_status = f"Hand: {'OK' if self.position is not None else 'NONE'}"
            cable_status = f"Cable Points: {len(self.cable_points)}"
            
            # 배경 박스 그리기
            cv.rectangle(imgbgr, (5, 5), (500, 80), (0, 0, 0), -1)
            cv.rectangle(imgbgr, (5, 5), (500, 80), (255, 255, 255), 2)
            
            # 상태 텍스트
            cv.putText(imgbgr, hand_status, (15, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
                     (0, 255, 0) if self.position is not None else (0, 0, 255), 2)
            cv.putText(imgbgr, cable_status, (15, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
                     (0, 255, 0) if len(self.cable_points) > 0 else (0, 0, 255), 2)
            
            # 범례 표시
            legend_y = imgbgr.shape[0] - 60
            cv.rectangle(imgbgr, (10, legend_y - 5), (300, imgbgr.shape[0] - 10), (0, 0, 0), -1)
            cv.putText(imgbgr, "Legend:", (15, legend_y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.circle(imgbgr, (25, legend_y + 30), 8, (0, 255, 255), -1)
            cv.putText(imgbgr, "Hand (Yellow)", (40, legend_y + 35), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv.circle(imgbgr, (25, legend_y + 45), 8, (0, 255, 0), -1)
            cv.putText(imgbgr, "Cable Point (Green)", (40, legend_y + 50), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 화면 표시
            cv.imshow("Hand + Cable Detection", imgbgr)
            
            # 빨간색 마스크를 컬러로 변환해서 더 보기 좋게
            red_mask_colored = cv.applyColorMap(red_mask, cv.COLORMAP_HOT)
            
            # HSV 값들을 마스크 화면에 표시
            (h_min, s_min, v_min), (h_max, s_max, v_max), min_area = self.get_trackbar_values()
            hsv_text1 = f"HSV Range: H({h_min}-{h_max}) S({s_min}-{s_max}) V({v_min}-{v_max})"
            hsv_text2 = f"Min Area: {min_area}"
            cv.putText(red_mask_colored, hsv_text1, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv.putText(red_mask_colored, hsv_text2, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv.imshow("Red Detection Mask", red_mask_colored)
            cv.waitKey(1)

    
    def Transform(self):
        # 카메라 → 로봇 base 변환 행렬
        r2cx = 0.0
        r2cy = 0.0
        r2cz = 0.0
        H_r2c = np.array([[0,0,1,r2cx],[-1,0,0,r2cy],[0,-1,0,r2cz],[0,0,0,1]])
        
        # 손 위치 변환
        if self.position is not None:
            print(f"Hand position: {self.position}")
            # hand position based on camera
            H_c2h = np.array([[1,0,0,self.position[0]],[0,1,0,self.position[1]],[0,0,1,self.position[2]],[0,0,0,1]])
            self.H_r2h = H_r2c @ H_c2h
        else:
            self.H_r2h = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        
        # 🆕 케이블 포인트들 변환
        self.cable_points_robot = []
        if len(self.cable_points) > 0:
            print(f"Transforming {len(self.cable_points)} cable points to robot frame:")
            for i, cable_point in enumerate(self.cable_points):
                # cable point based on camera
                H_c2cable = np.array([[1,0,0,cable_point[0]],[0,1,0,cable_point[1]],[0,0,1,cable_point[2]],[0,0,0,1]])
                H_r2cable = H_r2c @ H_c2cable
                
                cable_point_robot = [H_r2cable[0][3], H_r2cable[1][3], H_r2cable[2][3]]
                self.cable_points_robot.append(cable_point_robot)
                print(f"  Cable point {i} (robot): X={cable_point_robot[0]:.3f}, Y={cable_point_robot[1]:.3f}, Z={cable_point_robot[2]:.3f}")
        else:
            print("No cable points detected.")
    
    def offset_pub(self):
        self.Image_Processing()
        self.Transform()
        
        # 손 위치 퍼블리시
        a_pose = Pose()
        if self.no_hand == True:
            print('No hand detected, no publish hand pose')
        else:
            print(f"Publishing hand pose: {self.H_r2h}")
            a_pose.position.x = float(self.H_r2h[0][3])
            a_pose.position.y = float(self.H_r2h[1][3])
            a_pose.position.z = float(self.H_r2h[2][3])
            self.pose_publisher.publish(a_pose)
        
        # 🆕 케이블 포인트들 퍼블리시
        if len(self.cable_points_robot) > 0:
            cable_pose_array = PoseArray()
            cable_pose_array.header.stamp = self.get_clock().now().to_msg()
            cable_pose_array.header.frame_id = "robot_base"  # 또는 적절한 프레임 ID
            
            for cable_point in self.cable_points_robot:
                cable_pose = Pose()
                cable_pose.position.x = float(cable_point[0])
                cable_pose.position.y = float(cable_point[1])
                cable_pose.position.z = float(cable_point[2])
                cable_pose.orientation.w = 1.0  # 기본 오리엔테이션
                cable_pose_array.poses.append(cable_pose)
            
            self.cable_points_publisher.publish(cable_pose_array)
            print(f"Published {len(self.cable_points_robot)} cable points")
        else:
            print('No cable points detected, no publish cable points')
    
def main(args = None):
    rclpy.init(args=args)
    vision_tracker = Vision_Hand()
    rclpy.spin(vision_tracker)
    vision_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()