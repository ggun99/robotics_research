import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
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
        self.pose_publisher = self.create_publisher(Pose,'/aruco_pose', 10)
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
        self.yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model
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

            # cv.imshow("YOLO + Hand Tracking", imgrgb)
            # cv.waitKey(1)

            # 1. Detect people and objects
            yolo_results = self.yolo_model(imgrgb, verbose=False)[0]
            person_boxes = []
            object_boxes = []
            for box in yolo_results.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 0:  # person
                    person_boxes.append((x1, y1, x2, y2))
                elif cls != 0:  # other object
                    object_boxes.append(((x1 + x2) // 2, (y1 + y2) // 2))

            # 2. Hand detection
            results = self.hands.process(imgrgb)
            hand_positions = []
            hand_dist = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    wrist = hand_landmarks.landmark[0]  # wrist landmark
                    x = int(wrist.x * imgrgb.shape[1])
                    y = int(wrist.y * imgrgb.shape[0])
                    cv.circle(imgbgr, (x, y), 6, (0, 255, 255), -1)
                    
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
                if hand_positions:
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
            for x1, y1, x2, y2 in person_boxes:
                cv.rectangle(imgbgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.imshow("YOLO + Hand Tracking", imgbgr)
            cv.waitKey(1)

    
    def Transform(self):
        if self.position is not None:
            print(f"Hand position: {self.position}")
            # human position based on camera
            H_c2h = np.array([[1,0,0,self.position[0]],[0,1,0,self.position[1]],[0,0,1,self.position[2]],[0,0,0,1]])

            # camera based on ur5e base
            H_r2c = np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])

            # aruco based on scout
            self.H_r2h = H_r2c @ H_c2h

        else:
            self.H_r2h = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    
    def offset_pub(self):
        self.Image_Processing()
        self.Transform()
        # if H_matrix is not None:
        a_pose = Pose()
        if self.no_hand:
            print('No hand detected, no publish')
        else:
            print(self.H_r2h)
            a_pose.position.x = float(self.H_r2h[0][3])
            a_pose.position.y = float(self.H_r2h[1][3])
            a_pose.position.z = float(self.H_r2h[2][3])
            self.pose_publisher.publish(a_pose)
    
def main(args = None):
    rclpy.init(args=args)
    vision_tracker = Vision_Hand()
    rclpy.spin(vision_tracker)
    vision_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()