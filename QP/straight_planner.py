import numpy as np

class LinearTrajectoryPlanner:
    def __init__(self, start=(0, 0, 0), end=(1, 1, 1)):
        """
        직선 궤적을 생성하는 클래스.

        Args:
            start (tuple): 직선 궤적의 시작점 (x, y, z).
            end (tuple): 직선 궤적의 끝점 (x, y, z).
        """
        self.start = np.array(start)
        self.end = np.array(end)

    def generate_linear_path(self, num_points=300):
        """
        직선 궤적을 생성.

        Args:
            num_points (int): 궤적을 구성하는 점의 개수.

        Returns:
            np.ndarray: 직선 궤적 (shape: num_points x 3).
        """
        t = np.linspace(0, 1, num_points)
        x = self.start[0] + t * (self.end[0] - self.start[0])
        y = self.start[1] + t * (self.end[1] - self.start[1])
        z = self.start[2] + t * (self.end[2] - self.start[2])
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
    # 직선 궤적 생성기 초기화
    planner = LinearTrajectoryPlanner(start=(0, 0, 0), end=(5, 5, 5))

    # 직선 궤적 생성
    human_smoothed_path = planner.generate_linear_path(num_points=300)

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