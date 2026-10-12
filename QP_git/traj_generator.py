import matplotlib.pyplot as plt
import numpy as np


class TrajectoryGenerator:
    """Trajectory generation for the robot's end-effector."""

    def __init__(self, start_pose, duration, num_points, human_position=None, target_pose=None):
        self.start_pose = np.array(start_pose)
        self.duration = duration
        self.num_points = num_points
        self.human_position = np.array([0.5, 0.0, 0.5]) if human_position is None else np.array(human_position)
        self.target_pose = np.array([0.6, 0.5, 0.5]) if target_pose is None else np.array(target_pose)
        self.safe_distance = 0.2  # Minimum safe distance from the target pose

    def get_tangent_points(self, center, external_point, radius):
        """
        center        : 원의 중심 [x_t, y_t]
        external_point: 원 밖의 점 [x_h, y_h]
        radius        : 원의 반지름 d_s

        return:
            tangent_point_1, tangent_point_2
        """

        C = np.asarray(center, dtype=float)
        H = np.asarray(external_point, dtype=float)

        # 중심 -> 외부점 벡터
        v = H - C

        # 중심과 외부점 사이 거리
        D = np.linalg.norm(v)

        # 접선이 존재하는지 확인
        if D < radius:
            raise ValueError("외부점이 원 내부에 있습니다.")

        if np.isclose(D, radius):
            # 외부점이 원 위에 있는 경우 접점이 하나
            return H, H

        # 중심 -> 외부점 방향 단위벡터
        e = v / D

        # e에 수직인 단위벡터
        e_perp = np.array([-e[1], e[0]])

        # 접점 계산
        alpha = radius**2 / D
        beta = radius * np.sqrt(D**2 - radius**2) / D

        A1 = C + alpha * e + beta * e_perp
        A2 = C + alpha * e - beta * e_perp

        Robot_pose = self.start_pose[:2]  # 현재 로봇 위치 (x, y)
        A_final = A1 if np.linalg.norm(Robot_pose - A1) > np.linalg.norm(Robot_pose - A2) else A2

        return A_final

    
    # def generate_trajectory(self):
    #     """Generate a linear trajectory from start_pose to end_pose."""
    #     # Calculate the end pose based on the human position
    #     self.end_pose = self.get_tangent_points(self.target_pose[:2], self.human_position[:2], self.safe_distance)

    #     t = np.linspace(0, self.duration, self.num_points)
    #     trajectory = np.outer(t / self.duration, self.end_pose - self.start_pose[:2]) + self.start_pose[:2]
    #     return trajectory

    def generate_trajectory(self):
        """
        Generate a spiral trajectory around the target.

        - Start from the current robot position.
        - Move around the opposite side of the target.
        - Gradually decrease the radius.
        - Finally reach the tangent point.
        - The tangent point is the final point, not an intermediate waypoint.
        """

        start = np.asarray(self.start_pose[:2], dtype=float)
        target = np.asarray(self.target_pose[:2], dtype=float)
        human = np.asarray(self.human_position[:2], dtype=float)

        # =========================================================
        # 1. Get tangent point
        # =========================================================
        tangent_points = np.asarray(
            self.get_tangent_points(
                target,
                human,
                self.safe_distance
            )
        )

        # 두 접점 중 human에서 더 먼 접점 선택
        if tangent_points.ndim > 1:
            distances = np.linalg.norm(
                tangent_points - human,
                axis=1
            )
            end = tangent_points[np.argmax(distances)]
        else:
            end = tangent_points

        self.end_pose = end

        # =========================================================
        # 2. Start / End angle
        # =========================================================
        start_vec = start - target
        end_vec = end - target

        r_start = np.linalg.norm(start_vec)

        theta_start = np.arctan2(
            start_vec[1],
            start_vec[0]
        )

        theta_end = np.arctan2(
            end_vec[1],
            end_vec[0]
        )

        # =========================================================
        # 3. Shortest angular direction to tangent point
        # =========================================================
        delta = np.arctan2(
            np.sin(theta_end - theta_start),
            np.cos(theta_end - theta_start)
        )

        # =========================================================
        # 4. Rotate in the OPPOSITE direction
        # =========================================================
        if delta >= 0:
            rotation_direction = -1.0
        else:
            rotation_direction = 1.0

        # =========================================================
        # 5. Generate spiral angle
        #
        # Start
        #   ↓
        # opposite side (~180 deg)
        #   ↓
        # tangent direction
        # =========================================================

        # 추가로 돌아가는 각도
        extra_angle = np.pi

        total_angle = (
            rotation_direction
            * (abs(delta) + extra_angle)
        )

        # 실제 최종 각도가 tangent point 방향이 되도록
        theta_final = theta_start + total_angle
        print(f"rotation_direction: {rotation_direction}, theta_start: {theta_start:.4f}, theta_end: {theta_end:.4f}, delta: {delta:.4f}, total_angle: {total_angle:.4f}, theta_final: {theta_final:.4f}")

        # angle 보정
        theta_final = (
            theta_final + np.pi
        ) % (2.0 * np.pi) - np.pi

        # =========================================================
        # 6. Waypoints
        #
        # waypoint는 실제 trajectory가 아니라
        # spiral의 방향을 결정하기 위한 reference
        # =========================================================
        num_waypoints = 8

        waypoint_angles = np.linspace(
            theta_start,
            theta_start + total_angle,
            num_waypoints
        )

        # waypoint의 반지름은 실제 trajectory보다 크게/동일하게 설정
        waypoint_radii = np.linspace(
            r_start,
            self.safe_distance,
            num_waypoints
        )

        waypoints = np.array([
            target + r * np.array([
                np.cos(theta),
                np.sin(theta)
            ])
            for theta, r in zip(
                waypoint_angles,
                waypoint_radii
            )
        ])

        # =========================================================
        # 7. Generate spiral trajectory
        # =========================================================

        trajectory = np.zeros(
            (self.num_points, 2)
        )

        for i, s in enumerate(
            np.linspace(0.0, 1.0, self.num_points)
        ):

            # ---------------------------------------------
            # Smooth interpolation
            # ---------------------------------------------
            # cosine easing
            alpha = (
                1.0 - np.cos(np.pi * s)
            ) / 2.0

            # ---------------------------------------------
            # Angular position
            # ---------------------------------------------
            theta = (
                theta_start
                + total_angle * alpha
            )

            # ---------------------------------------------
            # Radial position
            # ---------------------------------------------
            radius = (self.safe_distance + (r_start - self.safe_distance) * (1 - alpha)**2) #r_start + (self.safe_distance - r_start) * alpha)

            # radius = (
            #     r_start
            #     + (self.safe_distance - r_start)
            #     * alpha**2
            # )
            
            # radius = (
            #     r_start
            #     + (self.safe_distance - r_start)
            #     * np.sqrt(alpha)
            # )

            # ---------------------------------------------
            # Spiral point
            # ---------------------------------------------
            # print(f"i: {i}, theta: {theta:.4f}, radius: {radius:.4f}")
            trajectory[i] = (
                target
                + radius * np.array([
                    np.cos(theta),
                    np.sin(theta)
                ])
            )

        # =========================================================
        # 8. Final point = exact tangent point
        # =========================================================
        trajectory[-1] = end

        return trajectory, waypoints

    def plot_trajectory(self, trajectory, waypoints):
        """Plot the generated trajectory."""
        plt.figure(figsize=(8, 6))
        plt.plot(trajectory[:, 0], trajectory[:, 1], label='Trajectory')
        plt.scatter(self.start_pose[0], self.start_pose[1], color='green', label='Start Pose')
        plt.scatter(self.end_pose[0], self.end_pose[1], color='red', label='End Pose')
        plt.scatter(self.human_position[0], self.human_position[1], color='orange', label='Human Position')
        # for i in range(waypoints.shape[0]):
        #     plt.scatter(waypoints[i, 0], waypoints[i, 1], color='green', label='Waypoints')
        plt.gca().add_patch(plt.Circle([self.target_pose[0], self.target_pose[1]], self.safe_distance/2, color='red', alpha=0.2, label='target'))
        plt.gca().add_patch(plt.Circle([self.target_pose[0], self.target_pose[1]], self.safe_distance, color='blue', alpha=0.2, label='Safe Zone'))
        plt.title('End-Effector Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid()
        plt.axis('equal')
        plt.show()

    def main(self):
        trajectory, waypoints = self.generate_trajectory()
        self.plot_trajectory(trajectory, waypoints)

if __name__ == "__main__":
    start_pose = [0.5, 0.0, 0.5]  # Starting position of the end-effector
    duration = 5.0                # Duration of the trajectory in seconds
    num_points = 100              # Number of points in the trajectory
    human_position = [0.7, 0.0, 0.5]  # Position of the human
    target_pose = [0.6, 0.5, 0.5]     # Target position for the end-effector

    traj_gen = TrajectoryGenerator(start_pose, duration, num_points, human_position, target_pose)
    traj_gen.main()
    