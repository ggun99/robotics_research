#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import yaml
from threading import Lock

# =========================
# 사용자 설정
# =========================
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # 미터
IMAGE_PER_CAM = 20   # 카메라 당 캡처 수
OUTPUT_FOLDER = os.path.expanduser('~/calibration_results')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# Camera calibration node
# =========================
class MultiD435CalibNode(Node):
    def __init__(self):
        super().__init__('multi_d435_calib_node')
        self.bridge = CvBridge()
        self.cam_data = {}   # serial_number -> {'images': [], 'lock': Lock()}
        self.create_timer(1.0, self.list_cameras)

    def list_cameras(self):
        # 현재 active topic list 확인
        topics = self.get_topic_names_and_types()
        for topic, types in topics:
            if topic.endswith('/color/image_raw') and topic not in self.cam_data:
                serial = topic.split('/')[1]  # 토픽 이름 convention: /<serial>/color/image_raw
                self.get_logger().info(f'Found camera topic: {topic} (serial {serial})')
                self.cam_data[serial] = {'images': [], 'lock': Lock()}
                self.create_subscription(
                    Image, topic, lambda msg, s=serial: self.image_callback(msg, s), 10)

    def image_callback(self, msg, serial):
        with self.cam_data[serial]['lock']:
            if len(self.cam_data[serial]['images']) >= IMAGE_PER_CAM:
                return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.cam_data[serial]['images'].append(cv_image)
            self.get_logger().info(f'{serial}: captured image {len(self.cam_data[serial]["images"])}')
            
            if len(self.cam_data[serial]['images']) == IMAGE_PER_CAM:
                self.calibrate_camera(serial)

    def calibrate_camera(self, serial):
        self.get_logger().info(f'{serial}: Starting calibration...')
        objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
        objp *= SQUARE_SIZE

        objpoints = []
        imgpoints = []

        for img in self.cam_data[serial]['images']:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)

        if len(objpoints) < 3:
            self.get_logger().warn(f'{serial}: 이미지 수가 부족하여 정확한 보정이 어려울 수 있습니다.')

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        yaml_file = os.path.join(OUTPUT_FOLDER, f"{serial}_calibration.yaml")
        data = {
            'intrinsic_matrix': mtx.tolist(),
            'distortion_coefficients': dist.tolist(),
            'image_width': gray.shape[1],
            'image_height': gray.shape[0],
            'checkerboard': {'rows': CHECKERBOARD[1], 'cols': CHECKERBOARD[0], 'square_size_m': SQUARE_SIZE}
        }

        with open(yaml_file, 'w') as f:
            yaml.dump(data, f)

        self.get_logger().info(f'{serial}: Calibration 완료. 결과 -> {yaml_file}')

def main(args=None):
    rclpy.init(args=args)
    node = MultiD435CalibNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
