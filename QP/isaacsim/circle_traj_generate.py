import numpy as np

class CircularTrajectoryPlanner:
    def __init__(self, radius=1.0, center=(0, 0, 0), height=1.0):
        """
        원 궤적을 생성하는 클래스.

        Args:
            radius (float): 원의 반지름.
            center (tuple): 원의 중심 좌표 (x, y, z).
            height (float): z축 높이.
        """
        self.radius = radius
        self.center = np.array(center)
        self.height = height

    def generate_circle_path(self, num_points=300):
        """
        원 궤적을 생성.

        Args:
            num_points (int): 궤적을 구성하는 점의 개수.

        Returns:
            np.ndarray: 원 궤적 (shape: num_points x 3).
        """
        t = np.linspace(0, 2 * np.pi, num_points)
        x = self.center[0] + self.radius * np.cos(t)
        y = self.center[1] + self.radius * np.sin(t)
        z = np.full_like(t, self.center[2] + self.height)
        return np.vstack((x, y, z)).T

    def generate_velocity_profile(self, path, v_max=0.2):
        """
        속도 프로파일을 생성.

        Args:
            path (np.ndarray): 궤적 (shape: num_points x 3).
            v_max (float): 최대 속도.

        Returns:
            np.ndarray: 속도 프로파일 (shape: num_points x 1).
        """
        num_points = len(path)
        velocity_profile = np.full(num_points, v_max)
        return velocity_profile

    def generate_full_trajectory_with_offset(self, path, velocity_profile, dt=0.1, time_offset=0):
        """
        전체 궤적을 생성.

        Args:
            path (np.ndarray): 궤적 (shape: num_points x 3).
            velocity_profile (np.ndarray): 속도 프로파일 (shape: num_points x 1).
            dt (float): 시간 간격.
            time_offset (float): 시간 오프셋.

        Returns:
            dict: 전체 궤적 (x, y, z, t).
        """
        num_points = len(path)
        time = np.linspace(time_offset, time_offset + num_points * dt, num_points)
        trajectory = {
            "x": path[:, 0],
            "y": path[:, 1],
            "z": path[:, 2],
            "t": time
        }
        return trajectory

# 사용 예시
if __name__ == "__main__":
    # 원 궤적 생성기 초기화
    planner = CircularTrajectoryPlanner(radius=2.0, center=(0, 0, 0), height=1.0)

    # 원 궤적 생성
    human_smoothed_path = planner.generate_circle_path(num_points=300)

    # 속도 프로파일 생성
    human_velocity_profile = planner.generate_velocity_profile(human_smoothed_path, v_max=0.2)

    # 전체 궤적 생성
    trajectories = planner.generate_full_trajectory_with_offset(human_smoothed_path, human_velocity_profile, dt=0.1, time_offset=15)

    # 결과 출력
    print("Generated Trajectory:")
    print("X:", trajectories["x"])
    print("Y:", trajectories["y"])
    print("Z:", trajectories["z"])
    print("Time:", trajectories["t"])