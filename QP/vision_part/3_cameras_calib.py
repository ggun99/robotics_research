"""
ArUco 기반 물체 월드 좌표 추정 (3대 카메라 예시)
- 요구사항: OpenCV (cv2), numpy, PyYAML
- 개요:
  1) 카메라별 intrinsic과 extrinsic(R_wc, t_wc)을 YAML로 준비
  2) 각 카메라에서 ArUco 마커 검출 및 마커별 카메라 좌표계에서의 pose 추정
  3) 카메라->월드 좌표로 변환
  4) 인식된 마커들의 월드 좌표와 물체(오브젝트) 프레임에 있는 마커들의 알려진 좌표를 이용해
     물체의 rigid transform (R, t) 을 Umeyama(최소자승) 방식으로 계산

사용 예시:
  python3 aruco_multi_cam_object_pose.py --cam_params cam_params.yaml --cam_ids 0 1 2

전제(사용자 설정 필요):
  - cam_params YAML 파일에 각 카메라의 intrinsic (camera_matrix, dist_coeffs) 과
    extrinsic (R_wc, t_wc) 이 들어있어야 함. (R_wc : rotation matrix 3x3, t_wc : 3-vector)
  - 물체 프레임(object frame)에서의 각 마커의 3D 좌표를 알고 있어야 함.

주의: 실제 환경에서는 카메라 보정과 카메라 간 정밀한 extrinsic 추정이 필수입니다.
"""

import cv2
import numpy as np
import yaml
import argparse
import time

# --------------------------- 유틸리티 함수들 ---------------------------

def load_cam_params(yaml_path):
    """YAML에서 카메라 파라미터 로드
    구조 예시:
    cameras:
      cam0:
        id: 0
        camera_matrix: [[fx,0,cx],[0,fy,cy],[0,0,1]]
        dist_coeffs: [k1,k2,p1,p2,k3]
        R_wc: [[..],[..],[..]]  # camera-to-world rotation (3x3)
        t_wc: [tx,ty,tz]       # camera-to-world translation (3)
      cam1: ...
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    cams = {}
    for name, c in data.get('cameras', {}).items():
        cam = {}
        cam['id'] = c.get('id', None)
        cam['camera_matrix'] = np.array(c['camera_matrix'], dtype=np.float64)
        cam['dist_coeffs'] = np.array(c.get('dist_coeffs', []), dtype=np.float64)
        cam['R_wc'] = np.array(c['R_wc'], dtype=np.float64)
        cam['t_wc'] = np.array(c['t_wc'], dtype=np.float64).reshape(3,1)
        cams[name] = cam
    return cams


def detect_aruco(frame, aruco_dict, aruco_params):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    if ids is None:
        return [], []
    return corners, ids.flatten()


def estimate_marker_poses(corners, ids, marker_length, camera_matrix, dist_coeffs):
    """각 마커에 대해 estimatePoseSingleMarkers 호출
    반환: dict: id -> (rvec, tvec, corner)
    rvec: Rodrigues 벡터 (3,1), tvec: (3,1)
    """
    poses = {}
    if len(corners) == 0:
        return poses
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
    for i, idv in enumerate(ids):
        poses[int(idv)] = (rvecs[i].reshape(3,1), tvecs[i].reshape(3,1), corners[i])
    return poses


def cam_to_world_point(R_wc, t_wc, cam_point):
    """카메라 좌표계의 3x1 포인트를 월드 좌표로 변환
    world = R_wc @ cam_point + t_wc
    """
    return R_wc @ cam_point + t_wc


def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(rvec)
    return R


def umeyama(src_pts, dst_pts, with_scaling=False):
    """Umeyama algorithm: src_pts (Nx3) 를 rigid transform 해서 dst_pts (Nx3)에 맞춘다.
    반환: R (3x3), t (3x1), s (scale)  (scale은 항상 1이면 with_scaling=False)
    참조: https://ieeexplore.ieee.org/document/88573 (Umeyama)
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    assert src.shape == dst.shape
    n, m = src.shape
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst
    cov = (dst_centered.T @ src_centered) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1,-1] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_src = (src_centered**2).sum() / n
        scale = 1.0 / var_src * np.sum(D * S.diagonal())
    else:
        scale = 1.0
    t = mean_dst.reshape(m,1) - scale * R @ mean_src.reshape(m,1)
    return R, t, scale

# --------------------------- 메인 파이프라인 ---------------------------

def main(args):
    # ArUco 설정
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()

    # 카메라 파라미터 로드
    cams = load_cam_params(args.cam_params)

    # 사용자: 물체(오브젝트) 프레임에서의 각 마커 3D 좌표 (meter 단위 권장)
    # 예: 4개의 마커가 네 모서리에 있다고 가정
    # IDs: 0,1,2,3
    object_marker_coords = {
        0: np.array([0.0, 0.0, 0.0], dtype=np.float64),
        1: np.array([0.1, 0.0, 0.0], dtype=np.float64),
        2: np.array([0.1, 0.08, 0.0], dtype=np.float64),
        3: np.array([0.0, 0.08, 0.0], dtype=np.float64),
    }

    # 카메라 입력 설정(예시는 장치 ID 사용). 실제로는 영상 파일 또는 RTSP 스트림을 사용 가능
    cam_ids = args.cam_ids
    caps = {}
    for i, cam_name in enumerate(cams.keys()):
        if i >= len(cam_ids):
            break
        cap_id = cam_ids[i]
        cap = cv2.VideoCapture(int(cap_id))
        if not cap.isOpened():
            print(f"경고: 카메라 {cap_id} 열기 실패")
        caps[cam_name] = cap

    marker_length = args.marker_length  # meters

    try:
        while True:
            # 각 카메라 프레임 처리
            observed_world_points = []  # list of (marker_id, 3x1 world coord)
            for cam_name, cam in caps.items():
                ret, frame = cam.read()
                if not ret:
                    continue
                corners, ids = detect_aruco(frame, aruco_dict, aruco_params)
                cam_param = cams[cam_name]
                poses = estimate_marker_poses(corners, ids, marker_length, cam_param['camera_matrix'], cam_param['dist_coeffs'])
                # 각 마커 pose -> 카메라 좌표계의 tvec (마커 중심)
                for mid, (rvec, tvec, corner) in poses.items():
                    # 마커의 카메라 좌표 (tvec은 마커 중심 위치)
                    cam_point = tvec.reshape(3,1)
                    # 카메라->월드 변환
                    R_wc = cam_param['R_wc']
                    t_wc = cam_param['t_wc']
                    world_point = cam_to_world_point(R_wc, t_wc, cam_point)
                    observed_world_points.append((mid, world_point.reshape(3)))
                    # 시각화: 축 그리기 (카메라 프레임에서)
                    cv2.aruco.drawDetectedMarkers(frame, [corner])
                    cv2.aruco.drawAxis(frame, cam_param['camera_matrix'], cam_param['dist_coeffs'], rvec, tvec, marker_length*0.5)
                # 간단히 보여주기
                cv2.imshow(f"cam_{cam_name}", frame)

            # 만약 인식된 마커가 3개 이상이면 (rigid transform 안정화 위해) 오브젝트 pose 계산
            if len(observed_world_points) >= 3:
                # 동일 ID로 중복 관측 가능 -> 평균 처리
                id_to_worlds = {}
                for mid, wp in observed_world_points:
                    id_to_worlds.setdefault(mid, []).append(wp)
                # correspondences
                src_pts = []  # object frame coords
                dst_pts = []  # world coords
                for mid, wps in id_to_worlds.items():
                    if mid in object_marker_coords:
                        avg_world = np.mean(wps, axis=0)
                        src_pts.append(object_marker_coords[mid])
                        dst_pts.append(avg_world)
                src_pts = np.array(src_pts)
                dst_pts = np.array(dst_pts)
                if src_pts.shape[0] >= 3:
                    R_obj2world, t_obj2world, s = umeyama(src_pts, dst_pts, with_scaling=False)
                    print("=== Object pose (object -> world) ===")
                    print("R:\n", R_obj2world)
                    print("t:\n", t_obj2world.ravel())
                    # 예: 물체의 원점 월드좌표도 출력
                    origin_world = R_obj2world @ np.zeros((3,1)) + t_obj2world
                    print("object origin (world):", origin_world.ravel())

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # 루프 주기 제어
            time.sleep(0.01)
    finally:
        for c in caps.values():
            c.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_params', type=str, required=True, help='카메라 파라미터 YAML 파일 경로')
    parser.add_argument('--cam_ids', type=int, nargs='+', required=True, help='카메라 장치 ID 리스트 (예: 0 1 2)')
    parser.add_argument('--marker_length', type=float, default=0.03, help='마커 한 변의 길이 (meters)')
    args = parser.parse_args()
    main(args)
