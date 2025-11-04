#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import os
from scipy.spatial.transform import Rotation as R

class CameraPoseVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
        # 색상 설정
        self.colors = {
            'camera1': 'red',
            'camera2': 'blue', 
            'camera3': 'green',
            'reference': 'orange'
        }
        
        # 마커 크기
        self.marker_size = 0.05  # 5cm
    
    def load_calibration_data(self, filename='multi_camera_calibration.yaml'):
        """캘리브레이션 데이터 로드"""
        if not os.path.exists(filename):
            print(f"Error: {filename} not found!")
            return None
        
        try:
            with open(filename, 'r') as f:
                data = yaml.safe_load(f)
            
            print(f"Loaded calibration data from {filename}")
            return data
        
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
    
    def create_camera_frustum(self, position, rotation, scale=0.2, fov=60):
        """카메라 시야각(frustum) 생성"""
        # 카메라 로컬 좌표계에서 frustum 포인트들
        half_fov = np.radians(fov / 2)
        depth = scale
        
        # Frustum의 4개 모서리 (카메라 좌표계)
        frustum_points = np.array([
            [0, 0, 0],  # 카메라 중심
            [-depth * np.tan(half_fov), -depth * np.tan(half_fov), depth],
            [depth * np.tan(half_fov), -depth * np.tan(half_fov), depth],
            [depth * np.tan(half_fov), depth * np.tan(half_fov), depth],
            [-depth * np.tan(half_fov), depth * np.tan(half_fov), depth]
        ])
        
        # 월드 좌표계로 변환
        world_points = []
        for point in frustum_points:
            world_point = rotation @ point + position
            world_points.append(world_point)
        
        return np.array(world_points)
    
    def draw_coordinate_frame(self, position, rotation, scale=0.1, label=""):
        """좌표계 축 그리기"""
        # 축 벡터들 (X: 빨강, Y: 초록, Z: 파랑)
        axes = np.array([
            [scale, 0, 0],  # X축
            [0, scale, 0],  # Y축  
            [0, 0, scale]   # Z축
        ])
        
        axis_colors = ['red', 'green', 'blue']
        axis_labels = ['X', 'Y', 'Z']
        
        for i, (axis, color, axis_label) in enumerate(zip(axes, axis_colors, axis_labels)):
            # 월드 좌표로 변환
            world_axis = rotation @ axis + position
            
            # 축 그리기
            self.ax.plot([position[0], world_axis[0]], 
                        [position[1], world_axis[1]], 
                        [position[2], world_axis[2]], 
                        color=color, linewidth=2, alpha=0.8)
            
            # 축 라벨
            self.ax.text(world_axis[0], world_axis[1], world_axis[2], 
                        f'{axis_label}', fontsize=8, color=color)
        
        # 원점 표시
        self.ax.scatter(*position, color='black', s=50)
        if label:
            self.ax.text(position[0], position[1], position[2] + 0.05, 
                        label, fontsize=10, ha='center')
    
    def draw_reference_markers(self, reference_markers):
        """기준 마커들 그리기"""
        print("Drawing reference markers...")
        
        for marker_id, pos in reference_markers.items():
            position = np.array(pos)
            
            # 마커 표시
            self.ax.scatter(*position, color=self.colors['reference'], 
                           s=100, marker='s', alpha=0.8)
            
            # 마커 ID 라벨
            self.ax.text(position[0], position[1], position[2] + 0.02, 
                        f'ID:{marker_id}', fontsize=8, ha='center')
            
            print(f"  Marker {marker_id}: {position}")
        
        # 기준 마커들을 선으로 연결 (사각형 형태)
        if len(reference_markers) >= 4:
            # 마커 순서: 10-11-13-12-10 (사각형)
            connection_order = ['10', '11', '13', '12', '10']
            
            for i in range(len(connection_order) - 1):
                if connection_order[i] in reference_markers and connection_order[i+1] in reference_markers:
                    pos1 = reference_markers[connection_order[i]]
                    pos2 = reference_markers[connection_order[i+1]]
                    
                    self.ax.plot([pos1[0], pos2[0]], 
                               [pos1[1], pos2[1]], 
                               [pos1[2], pos2[2]], 
                               color='orange', linestyle='--', alpha=0.5)
    
    def draw_cameras(self, cameras_data):
        """카메라들 그리기"""
        print("Drawing cameras...")
        
        for camera_name, camera_data in cameras_data.items():
            position = np.array(camera_data['position'])
            rotation = np.array(camera_data['rotation_matrix'])
            color = self.colors.get(camera_name, 'gray')
            
            print(f"  {camera_name}:")
            print(f"    Position: {position}")
            print(f"    Looking direction: {rotation @ [0, 0, 1]}")  # Z축 방향
            
            # 카메라 위치 표시
            self.ax.scatter(*position, color=color, s=200, marker='^', 
                           alpha=0.8, edgecolors='black', linewidth=1)
            
            # 카메라 이름 라벨
            self.ax.text(position[0], position[1], position[2] + 0.08, 
                        camera_name, fontsize=10, ha='center', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            
            # 좌표계 축 그리기
            self.draw_coordinate_frame(position, rotation, scale=0.1, label="")
            
            # 카메라 시야각 그리기
            frustum_points = self.create_camera_frustum(position, rotation, scale=0.3)
            
            # Frustum 선 그리기
            # 카메라에서 4개 모서리로
            for i in range(1, 5):
                self.ax.plot([frustum_points[0][0], frustum_points[i][0]], 
                           [frustum_points[0][1], frustum_points[i][1]], 
                           [frustum_points[0][2], frustum_points[i][2]], 
                           color=color, alpha=0.3, linestyle='-')
            
            # 시야각 평면 (사각형)
            for i in range(1, 5):
                next_i = (i % 4) + 1
                self.ax.plot([frustum_points[i][0], frustum_points[next_i][0]], 
                           [frustum_points[i][1], frustum_points[next_i][1]], 
                           [frustum_points[i][2], frustum_points[next_i][2]], 
                           color=color, alpha=0.3, linestyle='-')
    
    def visualize_setup(self, calibration_data):
        """전체 셋업 시각화"""
        # Figure 생성
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        print("=== Camera Setup Visualization ===")
        
        # 기준 마커들 그리기
        if 'reference_markers' in calibration_data:
            self.draw_reference_markers(calibration_data['reference_markers'])
        
        # 카메라들 그리기
        if 'cameras' in calibration_data:
            self.draw_cameras(calibration_data['cameras'])
        
        # 월드 좌표계 원점 표시
        self.draw_coordinate_frame(np.array([0, 0, 0]), np.eye(3), 
                                  scale=0.15, label="World Origin")
        
        # 축 설정
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # 동일한 스케일 설정
        max_range = 0.5  # 기본 범위
        if 'cameras' in calibration_data:
            positions = [cam['position'] for cam in calibration_data['cameras'].values()]
            if positions:
                all_positions = np.array(positions)
                max_range = max(np.max(np.abs(all_positions)) + 0.2, max_range)
        
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range * 2])
        
        # 제목 및 범례
        self.ax.set_title('Multi-Camera Setup Visualization\n(Cameras: Triangles, Reference Markers: Squares)', 
                         fontsize=14, pad=20)
        
        # 격자 표시
        self.ax.grid(True, alpha=0.3)
        
        # 시점 설정 (등각 투영)
        self.ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
    
    def print_summary(self, calibration_data):
        """캘리브레이션 결과 요약 출력"""
        print("\n" + "="*50)
        print("CALIBRATION SUMMARY")
        print("="*50)
        
        if 'reference_markers' in calibration_data:
            print("\nReference Markers (World Coordinates):")
            for marker_id, pos in calibration_data['reference_markers'].items():
                print(f"  ID {marker_id}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        
        if 'cameras' in calibration_data:
            print("\nCamera Positions and Orientations:")
            for camera_name, camera_data in calibration_data['cameras'].items():
                pos = camera_data['position']
                rot_matrix = np.array(camera_data['rotation_matrix'])
                
                # 카메라가 보는 방향 (Z축)
                look_direction = rot_matrix @ [0, 0, 1]
                
                print(f"\n  {camera_name}:")
                print(f"    Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                print(f"    Look Direction: ({look_direction[0]:.3f}, {look_direction[1]:.3f}, {look_direction[2]:.3f})")
                
                # 오일러 각도 계산
                r = R.from_matrix(rot_matrix)
                euler = r.as_euler('xyz', degrees=True)
                print(f"    Rotation (xyz): ({euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°)")
        
        # 카메라 간 거리
        if 'cameras' in calibration_data and len(calibration_data['cameras']) > 1:
            print("\nInter-camera Distances:")
            camera_names = list(calibration_data['cameras'].keys())
            for i in range(len(camera_names)):
                for j in range(i+1, len(camera_names)):
                    cam1 = camera_names[i]
                    cam2 = camera_names[j]
                    pos1 = np.array(calibration_data['cameras'][cam1]['position'])
                    pos2 = np.array(calibration_data['cameras'][cam2]['position'])
                    distance = np.linalg.norm(pos2 - pos1)
                    print(f"  {cam1} ↔ {cam2}: {distance:.3f}m")

def main():
    """메인 함수"""
    visualizer = CameraPoseVisualizer()
    
    # 캘리브레이션 데이터 로드
    calibration_data = visualizer.load_calibration_data()
    
    if calibration_data is None:
        print("Failed to load calibration data!")
        return
    
    # 요약 정보 출력
    visualizer.print_summary(calibration_data)
    
    # 시각화
    visualizer.visualize_setup(calibration_data)
    
    print("\n" + "="*50)
    print("VISUALIZATION CONTROLS")
    print("="*50)
    print("- Mouse: Rotate view")
    print("- Mouse wheel: Zoom")
    print("- Close window to exit")
    
    plt.show()

if __name__ == '__main__':
    main()