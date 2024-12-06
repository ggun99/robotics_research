import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
import pyzed.sl as sl
import math
import numpy as np
import math
import cv2 as cv


class Vision_Tracker(Node):
    def __init__(self):
        super().__init__('Vision_Tracker')
        # Create a Camera object
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.set_from_svo_file("three.svo2")  # SVO 파일 경로 설정
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
        init_params.svo_real_time_mode = False  # 실시간이 아닌 SVO 파일 재생 모드 설정
        # Open the camera
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
            print("Camera Open : " + repr(status) + ". Exit program.")

        self.position = None
        self.H_s2a = None
        self.pose_publisher = self.create_publisher(Pose, '/aruco_pose', 10)
        self.H_s2a = None
        self.timeroffset = self.create_timer(0.016667, self.Image_Processing)

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat(self.zed.get_camera_information().camera_configuration.resolution.width,
                            self.zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.U8_C4)
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

        self.mark_5 = []
        self.mark_10 = []
        self.mark_15 = []

        mirror_ref = sl.Transform()
        mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
        # fx, fy
        self.focal_left_x = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
        self.focal_left_y = self.zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy

    def Transform(self):
        if self.position is not None:
            # aruco based on camera
            H_a2c = np.array(
                [[1, 0, 0, self.position[0]], [0, 1, 0, self.position[1]], [0, 0, 1, self.position[2]], [0, 0, 0, 1]])

            # camera to based on scout
            P_x = 0
            P_y = 0
            P_z = 0

            H_s2c = np.array([[0, 0, 1, P_x], [-1, 0, 0, P_y], [0, -1, 0, P_z], [0, 0, 0, 1]])

            # aruco based on scout
            self.H_s2a = H_s2c @ H_a2c

        else:
            self.H_s2a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    
    def middle_point(self, corners, i, imgnp):
            middle_point_x = int(
                    (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4)
            middle_point_y = int(
                    (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4)
            cv.circle(imgnp, (middle_point_x, middle_point_y), 4, [0, 255, 0], 2)
            return middle_point_x, middle_point_y

    def Image_Processing(self):
        # A new image is available if grab() returns SUCCESS
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            imgnp = self.image.get_data()
            # cv.imshow('img',imgnp)
            if imgnp is None:
                raise ValueError("Image not loaded. Please check the image path or URL.")
            # Check the number of channels in the image
            if len(imgnp.shape) == 2:  # Grayscale image
                imgnp = cv.cvtColor(imgnp, cv.COLOR_GRAY2BGR)
            elif imgnp.shape[2] == 4:  # RGBA image
                imgnp = cv.cvtColor(imgnp, cv.COLOR_RGBA2RGB)

            # Convert the image to grayscale
            gray = cv.cvtColor(imgnp, cv.COLOR_RGB2GRAY)

            # Define the dictionary and parameters
            aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
            parameters = cv.aruco.DetectorParameters()

            # Create the ArUco detector
            detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

            # Detect the markers
            corners, ids, rejected = detector.detectMarkers(gray)
            # print(np.shape(ids)[0],ids[1][0])
            # print(corners[1][0][1])

            # print("Detected markers:", ids)
            if ids is not None:
                for i in range(0,np.shape(ids)[0]):
                    if ids[i][0] == 5:
                        middle_point_5x, middle_point_5y = self.middle_point(corners, i, imgnp)
                        self.mark_5.append([middle_point_5x, middle_point_5y])
                    elif ids[i][0] == 10:
                        middle_point_10x, middle_point_10y = self.middle_point(corners, i, imgnp)
                        self.mark_10.append([middle_point_10x, middle_point_10y])
                    else:
                        middle_point_15x, middle_point_15y = self.middle_point(corners, i, imgnp)
                        self.mark_15.append([middle_point_15x, middle_point_15y])
            
                # Retrieve depth map. Depth is aligned on the left image
                self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

                # Get and print distance value in mm at the center of the image
                # We measure the distance camera - object using Euclidean distance
                x = middle_point_5x
                y = middle_point_5y
                cv.aruco.drawDetectedMarkers(imgnp, corners, ids)
                cv.imshow('Detected Markers', imgnp)
                cv.waitKey(1)

            else:
                # 아르코 마커가 없으면 가운데 픽셀 참조.
                self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
                x = round(self.image.get_width() / 2)
                y = round(self.image.get_height() / 2)

            err, point_cloud_value = self.point_cloud.get_value(x, y)

            if math.isfinite(point_cloud_value[2]):
                distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                     point_cloud_value[1] * point_cloud_value[1] +
                                     point_cloud_value[2] * point_cloud_value[2])
                X = distance * (x - 640) / self.focal_left_x
                Y = distance * (y - 360) / self.focal_left_y

                # print(f"Distance to Camera at {{{x};{y}}}: {distance} mm")
                # print(f"3D position to Camera : X : {X}, Y : {Y}, Z : {distance}")
                self.position = np.array([X, Y, distance])
            else:
                # print(f"The distance can not be computed at {{{x};{y}}}")
                self.position = None
            self.Transform()
            # if H_matrix is not None:
            a_pose = Pose()
            offset = 50  # in mm
            # print(self.H_s2a)
            a_pose.position.x = float(self.H_s2a[0][3])  # + offset)
            a_pose.position.y = float(self.H_s2a[1][3])
            a_pose.position.z = float(self.H_s2a[2][3])
            self.pose_publisher.publish(a_pose)

        else: 
            print("finished!!")
            m5 = np.array(self.mark_5)
            m10 = np.array(self.mark_10)
            m15 = np.array(self.mark_15)
            np.save('aruco_no5_x,y',arr=m5)
            np.save('aruco_no10_x,y',arr=m10)
            np.save('aruco_no15_x,y',arr=m15)
            print(m5.shape)
            print(m10.shape)
            print(m15.shape)
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    vision_tracker = Vision_Tracker()
    rclpy.spin(vision_tracker)
    vision_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()