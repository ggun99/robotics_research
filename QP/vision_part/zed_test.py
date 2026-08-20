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
        self.pose_publisher = self.create_publisher(Pose,'/hand_pose', 10)
        self.H_s2a = None
        # Create a Camera object
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  
        # [ULTRA depth mode : Computation mode that favors edges and sharpness. Requires more GPU memory and computation power.] 
        # [PERFORMANCE for faster processing : Computation mode optimized for speed.]
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)    
        init_params.camera_resolution = sl.RESOLUTION.HD1080  #    VGA  # Set resolution to 640x480  HD2K: 약 1m,  HD1080: 약 0.7m, HD720: 약 0.5m (500mm), VGA: 약 0.3m (300mm)
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
 

        # fx, fy
        self.focal_left_x = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
        self.focal_left_y = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
        


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

            # for x1, y1, x2, y2 in person_boxes:
            #     cv.rectangle(imgbgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv.imshow("YOLO + Hand Tracking", imgbgr)
            cv.waitKey(1)


    
def main(args = None):
    rclpy.init(args=args)
    vision_tracker = Vision_Hand()
    vision_tracker.Image_Processing()
    rclpy.spin(vision_tracker)
    vision_tracker.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()