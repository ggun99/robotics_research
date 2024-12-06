import pyzed.sl as sl
import math
import numpy as np
import math
import cv2 as cv
import os
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

# svo 불러와서 butter 필터 써서 3개의 아르코마커의 위치를 각각 다른 numpy array로 저장하는 코드

mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75, 4.0, 0))
depth = sl.Mat()
point_cloud = sl.Mat()

def init(video_path):
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(video_path)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use meter units (for depth measurements)
    init_params.svo_real_time_mode = False  # 실시간이 아닌 SVO 파일 재생 모드 설정
    # Open the camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:  # Ensure the camera has opened succesfully
        print("Camera Open : " + repr(status) + ". Exit program.")
    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    image = sl.Mat(zed.get_camera_information().camera_configuration.resolution.width,
                        zed.get_camera_information().camera_configuration.resolution.height, sl.MAT_TYPE.U8_C4)
    
    # fx, fy
    focal_left_x = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fx
    focal_left_y = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.fy
    return runtime_parameters, image, focal_left_x, focal_left_y, zed

def Transform(position):
    if position is not None:
        # aruco based on camera
        H_a2c = np.array(
            [[1, 0, 0, position[0]], [0, 1, 0, position[1]], [0, 0, 1, position[2]], [0, 0, 0, 1]])
        # camera based on scout
        P_x = 0
        P_y = 0
        P_z = 0
        H_c2s = np.array([[0, 0, 1, P_x], [-1, 0, 0, P_y], [0, -1, 0, P_z], [0, 0, 0, 1]])
        # aruco based on scout
        H_s2a = H_c2s @ H_a2c
    else:
        H_s2a = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return H_s2a

# Butterworth 저역통과 필터 설계
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 필터 적용 함수
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def middle_point(corners, i, imgnp, data_x, data_y):
        
        middle_point_x = int(
                (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4)
        middle_point_y = int(
                (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4)
        # Filter requirements.
        cutoff = 5.0  # 저역통과 필터의 컷오프 주파수
        fs = 60.0     # 프레임 속도 (초당 프레임)
        order = 3     # 필터 차수
        
        data_x.append(middle_point_x)
        data_y.append(middle_point_y)
        # 좌표 값 버퍼 크기 조정 (필터링할 데이터 크기 유지)
        if len(data_x) > 100:
            data_x.pop(0)
            data_y.pop(0)
        # 데이터가 충분할 때 필터 적용
        if len(data_x) > order:
            filtered_x = butter_lowpass_filter(data_x, cutoff, fs, order)
            filtered_y = butter_lowpass_filter(data_y, cutoff, fs, order)
            # Filter the data, and plot both the original and filtered signals.
            cv.circle(imgnp, (int(filtered_x[-1]), int(filtered_y[-1])), 4, [0, 255, 0], 2)
        return middle_point_x, middle_point_y

def PCL_3D(focal_left_x, focal_left_y, x, y):
    err, point_cloud_value = point_cloud.get_value(x, y)
    if math.isfinite(point_cloud_value[2]):
        distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                point_cloud_value[1] * point_cloud_value[1] +
                                point_cloud_value[2] * point_cloud_value[2])
        X = distance * (x - 640) / focal_left_x
        Y = distance * (y - 360) / focal_left_y
        position = np.array([X, Y, distance])
    else:
        position = None
    H_s2a = Transform(position)
    return H_s2a




video_dir = '/home/airlab/workspace/arm_prediction/src'
for i in range(3,4):
    print(f'{i}th iteration')
    position = None
    mark_5 = []
    mark_10 = []
    mark_15 = []
    data_x_5 = []
    data_y_5 = []
    data_x_10 = []
    data_y_10 = []
    data_x_15 = []
    data_y_15 = []
    fin = True
    video_path = os.path.join(video_dir, f'{i}.svo2')
    runtime_parameters, image, focal_left_x, focal_left_y, zed = init(video_path)
# def Image_Processing(zed, runtime_parameters, image, focal_left_x, focal_left_y):
    # A new image is available if grab() returns SUCCESS
    while fin == True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            imgnp = image.get_data()
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
            if ids is not None:
                # Retrieve depth map. Depth is aligned on the left image
                zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                # Retrieve colored point cloud. Point cloud is aligned on the left image.
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                for i in range(0,np.shape(ids)[0]):
                    if ids[i][0] == 5:
                        middle_point_5x, middle_point_5y = middle_point(corners, i, imgnp, data_x_5, data_y_5)
                        H_s2a = PCL_3D(focal_left_x, focal_left_y, middle_point_5x,middle_point_5y)
                        mark_5.append([H_s2a[0][3], H_s2a[1][3], H_s2a[2][3]])
                    elif ids[i][0] == 10:
                        middle_point_10x, middle_point_10y = middle_point(corners, i, imgnp, data_x_10, data_y_10)
                        H_s2a = PCL_3D(focal_left_x, focal_left_y, middle_point_10x,middle_point_10y)
                        mark_10.append([H_s2a[0][3], H_s2a[1][3], H_s2a[2][3]])
                    else:
                        middle_point_15x, middle_point_15y = middle_point(corners, i, imgnp, data_x_15, data_y_15)
                        H_s2a = PCL_3D(focal_left_x, focal_left_y, middle_point_15x,middle_point_15y)
                        mark_15.append([H_s2a[0][3], H_s2a[1][3], H_s2a[2][3]])
                cv.aruco.drawDetectedMarkers(imgnp, corners, ids)
                cv.imshow('Detected Markers', imgnp)
                cv.waitKey(1)
            else:
                # 아르코 마커가 없으면 가운데 픽셀 참조.
                print('there is no aruco')
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                x = round(image.get_width() / 2)
                y = round(image.get_height() / 2)
                cv.imshow('No Detected Markers', imgnp)
                cv.waitKey(1)
                cv.destroyAllWindows()
                    
            
        else: 
            print("finished!!")
            m5 = np.array(mark_5)
            m10 = np.array(mark_10)
            m15 = np.array(mark_15)

            # if m5 is not None and m10 is not None and m15 is not None:
            if all(var != 0 for var in [np.shape(m5)[0], np.shape(m10)[0], np.shape(m15)[0]]):
                np.save(f'aruco_no5_x,y,z_{i}',arr=m5)
                np.save(f'aruco_no10_x,y,z_{i}',arr=m10)
                np.save(f'aruco_no15_x,y,z_{i}',arr=m15)
                print(m5.shape)
                print(m10.shape)
                print(m15.shape)
            fin = False
        # np.save('aruco_no5_x,y,z',arr=m5)
        # np.save('aruco_no10_x,y,z',arr=m10)
        # np.save('aruco_no15_x,y,z',arr=m15)
        # print(m5.shape)
        # print(m10.shape)
        # print(m15.shape)
        
# def main(args=None):
#     video_dir = '/home/airlab/workspace/arm_prediction/src'
#     for i in range(3,4):
#         video_path = os.path.join(video_dir, f'{i}.svo2')
#         runtime_parameters, image, focal_left_x, focal_left_y, zed = init(video_path)
#         m5, m10, m15 = Image_Processing(zed, runtime_parameters, image, focal_left_x, focal_left_y)
#         if m5 & m10 & m10 is not None:
#             np.save(f'aruco_no5_x,y,z_{i}',arr=m5)
#             np.save(f'aruco_no10_x,y,z_{i}',arr=m10)
#             np.save(f'aruco_no15_x,y,z_{i}',arr=m15)
#             print(m5.shape)
#             print(m10.shape)
#             print(m15.shape)
#           # SVO 파일 경로 설정
    


# if __name__ == "__main__":
#     main()
