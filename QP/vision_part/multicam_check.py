import numpy as np
import yaml
import math

def load_transformation_matrix(yaml_file):
    """YAML 파일에서 변환 행렬 로드"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    H = np.array(data['stereo_transformation']['transformation_matrix'])
    baseline = data['stereo_transformation']['baseline_distance_m']
    timestamp = data['stereo_transformation']['timestamp']
    
    return H, baseline, timestamp

def analyze_camera_positions():
    """카메라 위치 일관성 분석"""
    
    print("=== 3-Camera Position Consistency Analysis ===\n")
    
    # 1. 변환 행렬들 로드
    try:
        H_cam1_cam2, baseline_12, ts_12 = load_transformation_matrix('stereo_calibration_results/H_camera1_camera2_current.yaml')
        H_cam1_cam3, baseline_13, ts_13 = load_transformation_matrix('stereo_calibration_results/H_camera1_camera3_current.yaml') 
        H_cam2_cam3, baseline_23, ts_23 = load_transformation_matrix('stereo_calibration_results/H_camera2_camera3_current.yaml')
        
        print("✅ 모든 변환 행렬 로드 완료")
        print(f"   H_cam1→cam2: {baseline_12:.3f}m (측정시간: {ts_12})")
        print(f"   H_cam1→cam3: {baseline_13:.3f}m (측정시간: {ts_13})")
        print(f"   H_cam2→cam3: {baseline_23:.3f}m (측정시간: {ts_23})")
        print()
        
    except Exception as e:
        print(f"❌ 파일 로드 실패: {e}")
        return
    
    # 2. 카메라 위치 계산 (Camera1을 원점으로 설정)
    print("=== Camera Positions (Camera1 = Origin) ===")
    
    # Camera1 위치 (원점)
    pos_cam1 = np.array([0, 0, 0])
    rot_cam1 = np.eye(3)
    
    # Camera2 위치 (H_cam1→cam2에서 추출)
    # H는 left→right 변환이므로, Camera2의 위치는 H의 translation 부분
    pos_cam2 = H_cam1_cam2[:3, 3]
    rot_cam2 = H_cam1_cam2[:3, :3]
    
    # Camera3 위치 (H_cam1→cam3에서 추출)
    pos_cam3 = H_cam1_cam3[:3, 3]
    rot_cam3 = H_cam1_cam3[:3, :3]
    
    print(f"Camera1 위치: [{pos_cam1[0]:7.3f}, {pos_cam1[1]:7.3f}, {pos_cam1[2]:7.3f}] m")
    print(f"Camera2 위치: [{pos_cam2[0]:7.3f}, {pos_cam2[1]:7.3f}, {pos_cam2[2]:7.3f}] m")
    print(f"Camera3 위치: [{pos_cam3[0]:7.3f}, {pos_cam3[1]:7.3f}, {pos_cam3[2]:7.3f}] m")
    print()
    
    # 3. 거리 검증 (실제 측정값 vs 계산값)
    print("=== Distance Verification ===")
    
    # 계산된 거리들
    dist_12_calc = np.linalg.norm(pos_cam2 - pos_cam1)
    dist_13_calc = np.linalg.norm(pos_cam3 - pos_cam1)
    dist_23_calc = np.linalg.norm(pos_cam3 - pos_cam2)
    
    # 측정된 거리들
    dist_12_meas = baseline_12
    dist_13_meas = baseline_13
    dist_23_meas = baseline_23
    
    print(f"Camera1 ↔ Camera2:")
    print(f"   측정값: {dist_12_meas:.4f}m")
    print(f"   계산값: {dist_12_calc:.4f}m")
    print(f"   오차:   {abs(dist_12_calc - dist_12_meas)*1000:.2f}mm")
    print()
    
    print(f"Camera1 ↔ Camera3:")
    print(f"   측정값: {dist_13_meas:.4f}m")
    print(f"   계산값: {dist_13_calc:.4f}m")
    print(f"   오차:   {abs(dist_13_calc - dist_13_meas)*1000:.2f}mm")
    print()
    
    print(f"Camera2 ↔ Camera3:")
    print(f"   측정값: {dist_23_meas:.4f}m")
    print(f"   계산값: {dist_23_calc:.4f}m")
    print(f"   오차:   {abs(dist_23_calc - dist_23_meas)*1000:.2f}mm")
    print()
    
    # 4. 삼각형 폐쇄 검증 (Triangle Closure)
    print("=== Triangle Closure Verification ===")
    
    # H_cam2→cam3 측정값
    H_cam2_cam3_measured = H_cam2_cam3
    
    # H_cam2→cam3 계산값 (H_cam1→cam2^-1 @ H_cam1→cam3 경로)
    H_cam2_cam1 = np.linalg.inv(H_cam1_cam2)  # cam1→cam2의 역행렬
    H_cam2_cam3_calculated = H_cam2_cam1 @ H_cam1_cam3
    
    # 변환 행렬 비교
    R_measured = H_cam2_cam3_measured[:3, :3]
    T_measured = H_cam2_cam3_measured[:3, 3]
    
    R_calculated = H_cam2_cam3_calculated[:3, :3]
    T_calculated = H_cam2_cam3_calculated[:3, 3]
    
    # 회전 오차 계산
    R_diff = R_calculated @ R_measured.T
    trace_R = np.trace(R_diff)
    trace_R = np.clip(trace_R, -1.0, 3.0)  # 수치 오차 방지
    rot_error_rad = math.acos((trace_R - 1) / 2)
    rot_error_deg = math.degrees(rot_error_rad)
    
    # 평행이동 오차 계산
    trans_error_vec = T_calculated - T_measured
    trans_error_norm = np.linalg.norm(trans_error_vec)
    
    print("Camera2 → Camera3 변환 비교:")
    print(f"측정된 Translation: [{T_measured[0]:7.3f}, {T_measured[1]:7.3f}, {T_measured[2]:7.3f}] m")
    print(f"계산된 Translation: [{T_calculated[0]:7.3f}, {T_calculated[1]:7.3f}, {T_calculated[2]:7.3f}] m")
    print(f"Translation 오차: {trans_error_norm*1000:.2f} mm")
    print(f"Rotation 오차: {rot_error_deg:.4f} degrees")
    print()
    
    # 5. 전체 평가
    print("=== Overall Assessment ===")
    
    max_distance_error = max(
        abs(dist_12_calc - dist_12_meas),
        abs(dist_13_calc - dist_13_meas), 
        abs(dist_23_calc - dist_23_meas)
    ) * 1000  # mm
    
    print(f"최대 거리 오차: {max_distance_error:.2f} mm")
    print(f"삼각형 폐쇄 오차: {trans_error_norm*1000:.2f} mm")
    print(f"회전 일관성 오차: {rot_error_deg:.4f} degrees")
    print()
    
    # 품질 평가
    if max_distance_error < 10 and trans_error_norm*1000 < 10 and rot_error_deg < 1.0:
        print("✅ 캘리브레이션 품질: 우수 (오차 < 10mm, < 1도)")
    elif max_distance_error < 30 and trans_error_norm*1000 < 30 and rot_error_deg < 3.0:
        print("⚠️  캘리브레이션 품질: 보통 (오차 < 30mm, < 3도)")
    else:
        print("❌ 캘리브레이션 품질: 불량 (재캘리브레이션 권장)")
    
    # 6. 카메라 배치 시각화 정보
    print("\n=== Camera Layout Analysis ===")
    
    # 카메라 간의 각도 계산
    vec_12 = pos_cam2 - pos_cam1
    vec_13 = pos_cam3 - pos_cam1
    vec_23 = pos_cam3 - pos_cam2
    
    # 벡터 정규화
    vec_12_norm = vec_12 / np.linalg.norm(vec_12)
    vec_13_norm = vec_13 / np.linalg.norm(vec_13)
    
    # 각도 계산
    angle_213 = math.degrees(math.acos(np.clip(np.dot(vec_12_norm, vec_13_norm), -1.0, 1.0)))
    
    print(f"Camera1에서 본 Camera2-Camera3 각도: {angle_213:.1f}도")
    print(f"카메라 배치 형태: ", end="")
    
    if 80 <= angle_213 <= 100:
        print("거의 직각 배치 (스테레오 비전에 적합)")
    elif angle_213 < 45:
        print("너무 좁은 각도 (깊이 정확도 낮음)")
    elif angle_213 > 135:
        print("너무 넓은 각도 (매칭 어려움)")
    else:
        print("적당한 각도")
    
    print(f"\n삼각형 변의 길이:")
    print(f"   Cam1-Cam2: {dist_12_calc:.3f}m")
    print(f"   Cam1-Cam3: {dist_13_calc:.3f}m") 
    print(f"   Cam2-Cam3: {dist_23_calc:.3f}m")
    
    # 무게중심 계산
    centroid = (pos_cam1 + pos_cam2 + pos_cam3) / 3
    print(f"\n카메라 배치의 중심점: [{centroid[0]:7.3f}, {centroid[1]:7.3f}, {centroid[2]:7.3f}] m")

if __name__ == "__main__":
    analyze_camera_positions()