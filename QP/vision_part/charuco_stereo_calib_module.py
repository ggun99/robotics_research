import cv2
import numpy as np

# Charuco Board 파라미터 정의 (5x4, 50mm)
SQUARES_X = 5
SQUARES_Y = 4
SQUARE_LENGTH = 0.05  # 50mm
MARKER_LENGTH = 0.037  # 40mm (SQUARE_LENGTH의 80% 가정)
DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
BOARD = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), 
                               SQUARE_LENGTH, 
                               MARKER_LENGTH, 
                               DICTIONARY)

def detect_and_collect_charuco_data(img1, img2, K1, K2):
    """
    두 이미지에서 Charuco 코너를 감지하고 매칭된 2D/3D 포인트를 반환합니다.
    """
    
    # 1. Grayscale 변환 (인식률 향상)
    gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 2. ArUco 마커 감지
    detector_params = cv2.aruco.DetectorParameters()
    corners1, ids1, _ = cv2.aruco.detectMarkers(gray_img1, DICTIONARY, parameters=detector_params)
    corners2, ids2, _ = cv2.aruco.detectMarkers(gray_img2, DICTIONARY, parameters=detector_params)
    
    if ids1 is None or ids2 is None:
        return None, None, None, None, None

    # 3. Charuco 코너 보간 (D=0 가정)
    D = np.zeros((5, 1)) 
    
    _, charuco_corners1, charuco_ids1 = cv2.aruco.interpolateCornersCharuco(
        corners1, ids1, gray_img1, BOARD, K1, D)
    
    _, charuco_corners2, charuco_ids2 = cv2.aruco.interpolateCornersCharuco(
        corners2, ids2, gray_img2, BOARD, K2, D)

    if charuco_corners1 is None or charuco_corners2 is None:
        return None, None, None, None, None

    # 4. 공통 Charuco 코너 매칭 및 3D 포인트 수집
    pts1, pts2, obj_pts = [], [], []
    board_obj_points = BOARD.getChessboardCorners() 
    
    # 공통 ID를 찾고 리스트에 추가
    ids1_flat = charuco_ids1.flatten()
    ids2_flat = charuco_ids2.flatten()
    
    for i, id1 in enumerate(ids1_flat):
        if id1 in ids2_flat:
            j = np.where(ids2_flat == id1)[0][0]
            
            pts1.append(charuco_corners1[i][0])
            pts2.append(charuco_corners2[j][0])
            obj_pts.append(board_obj_points[id1])

    if len(pts1) < 10: # 최소 10개 미만의 매칭 코너는 무시
        return None, None, None, None, None

    # 5. 시각화용 감지된 ArUco 코너 반환 (선택 사항)
    img_pts1 = np.array(pts1, dtype=np.float32)
    img_pts2 = np.array(pts2, dtype=np.float32)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    
    return img_pts1, img_pts2, obj_pts, corners1, corners2

def perform_stereo_calibration(obj_points_list, img_points_l_list, img_points_r_list, K1, K2, image_size):
    """
    수집된 데이터를 사용하여 Stereo 캘리브레이션을 수행하고 R, T를 반환합니다.
    """
    D1 = np.zeros((5, 1))
    D2 = np.zeros((5, 1))
    
    # CALIB_FIX_INTRINSIC: K, D를 고정하고 R, T만 찾도록 설정
    flags = cv2.CALIB_FIX_INTRINSIC 
    
    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        objectPoints=obj_points_list,
        imagePoints1=img_points_l_list,
        imagePoints2=img_points_r_list,
        cameraMatrix1=K1,
        distCoeffs1=D1,
        cameraMatrix2=K2,
        distCoeffs2=D2,
        imageSize=image_size,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-6)
    )
    
    if ret:
        return R, T, ret # R, T, Reprojection Error 반환
    return None, None, None