import matplotlib.pyplot as plt
import numpy as np


class TrajectoryGenerator:
    """Trajectory generation for the robot's end-effector."""

    def __init__(self, start_pose, duration, num_points, human_position=None, target_pose=None):
        self.start_pose = np.array(start_pose)
        self.duration = duration
        self.num_points = num_points
        if human_position is None:
            raise ValueError("human_position must be provided.")
        self.human_position = np.array(human_position)
        self.target_pose = np.array([0.6, 0.5, 0.5]) if target_pose is None else np.array(target_pose)
        self.safe_distance = 0.2  # Minimum safe distance from the target pose

    @staticmethod
    def _normalize(vector):
        vector = np.asarray(vector, dtype=float)
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            raise ValueError("Zero-length vector cannot be normalized.")
        return vector / norm

    @staticmethod
    def _build_pose(position, z_axis, previous_z_axis=None):
        z_axis = TrajectoryGenerator._normalize(z_axis)

        down_hint = np.array([0.0, 0.0, -1.0])
        y_axis = down_hint - np.dot(down_hint, z_axis) * z_axis

        if np.linalg.norm(y_axis) < 1e-8:
            reference_axis = np.array([0.0, 1.0, 0.0])
            y_axis = reference_axis - np.dot(reference_axis, z_axis) * z_axis

        if np.linalg.norm(y_axis) < 1e-8:
            reference_axis = np.array([1.0, 0.0, 0.0])
            y_axis = reference_axis - np.dot(reference_axis, z_axis) * z_axis

        if np.linalg.norm(y_axis) < 1e-8:
            if previous_z_axis is None:
                raise ValueError("Cannot construct a stable orientation frame.")
            return TrajectoryGenerator._build_pose(position, previous_z_axis, None)

        y_axis = TrajectoryGenerator._normalize(y_axis)
        x_axis = TrajectoryGenerator._normalize(np.cross(y_axis, z_axis))
        y_axis = TrajectoryGenerator._normalize(np.cross(z_axis, x_axis))

        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = position
        return pose

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

        # # 실제 최종 각도가 tangent point 방향이 되도록
        # theta_final = theta_start + total_angle

        # # angle 보정
        # theta_final = (
        #     theta_final + np.pi
        # ) % (2.0 * np.pi) - np.pi

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

        trajectory_2d = np.zeros(
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
            radius = (self.safe_distance + (r_start - self.safe_distance) * (1 - alpha)**2)

          

            # ---------------------------------------------
            # Spiral point
            # ---------------------------------------------
            # print(f"i: {i}, theta: {theta:.4f}, radius: {radius:.4f}")
            trajectory_2d[i] = (
                target
                + radius * np.array([
                    np.cos(theta),
                    np.sin(theta)
                ])
            )

        # =========================================================
        # 8. Final point = exact tangent point
        # =========================================================
        trajectory_2d[-1] = end

        z_value = float(self.start_pose[2])
        trajectory = np.column_stack([
            trajectory_2d,
            np.full(self.num_points, z_value)
        ])

        waypoints = np.column_stack([
            waypoints,
            np.full(waypoints.shape[0], z_value)
        ])

        poses = np.zeros((self.num_points, 4, 4))
        previous_z_axis = None

        for i in range(self.num_points):
            if i < self.num_points - 1:
                z_axis = trajectory[i] - trajectory[i + 1]
            else:
                z_axis = trajectory[i] - trajectory[i - 1]

            pose = self._build_pose(trajectory[i], z_axis, previous_z_axis)
            poses[i] = pose
            previous_z_axis = pose[:3, 2]

        self.end_pose = np.array([end[0], end[1], z_value])
        self.trajectory_poses = poses

        return trajectory, waypoints, poses

    @staticmethod
    def _set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        plot_radius = 0.5 * max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_trajectory(self, trajectory, waypoints, poses):
        """Plot the generated trajectory in 3D with orientation frames."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Trajectory', color='black')
        ax.scatter(self.start_pose[0], self.start_pose[1], self.start_pose[2], color='green', label='Start Pose', s=50)
        ax.scatter(self.end_pose[0], self.end_pose[1], self.end_pose[2], color='red', label='End Pose', s=50)
        ax.scatter(self.human_position[0], self.human_position[1], self.human_position[2], color='orange', label='Human Position', s=50)
        ax.scatter(self.target_pose[0], self.target_pose[1], self.target_pose[2], color='blue', label='Target Pose', s=50)

        ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], linestyle='--', color='gray', alpha=0.7, label='Waypoints')

        sphere_u = np.linspace(0.0, 2.0 * np.pi, 24)
        sphere_v = np.linspace(0.0, np.pi, 12)
        sphere_x = self.target_pose[0] + self.safe_distance * np.outer(np.cos(sphere_u), np.sin(sphere_v))
        sphere_y = self.target_pose[1] + self.safe_distance * np.outer(np.sin(sphere_u), np.sin(sphere_v))
        sphere_z = self.target_pose[2] + self.safe_distance * np.outer(np.ones_like(sphere_u), np.cos(sphere_v))
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='royalblue', alpha=0.12, linewidth=0)

        frame_step = max(1, len(poses) // 10)
        frame_length = max(0.04, self.safe_distance * 0.35)

        for i in range(0, len(poses), frame_step):
            position = poses[i, :3, 3]
            rotation = poses[i, :3, :3]

            ax.quiver(
                position[0], position[1], position[2],
                rotation[0, 0], rotation[1, 0], rotation[2, 0],
                length=frame_length, color='red', arrow_length_ratio=0.15, linewidth=1.2
            )
            ax.quiver(
                position[0], position[1], position[2],
                rotation[0, 1], rotation[1, 1], rotation[2, 1],
                length=frame_length, color='green', arrow_length_ratio=0.15, linewidth=1.2
            )
            ax.quiver(
                position[0], position[1], position[2],
                rotation[0, 2], rotation[1, 2], rotation[2, 2],
                length=frame_length, color='blue', arrow_length_ratio=0.15, linewidth=1.2
            )

        ax.set_title('End-Effector Trajectory with Orientation Frames')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.legend()
        ax.grid(True)
        self._set_axes_equal(ax)
        plt.show()

    def main(self):
        trajectory, waypoints, poses = self.generate_trajectory()
        self.plot_trajectory(trajectory, waypoints, poses)

if __name__ == "__main__":
    start_pose = [0.5, 0.0, 0.5]  # Starting position of the end-effector
    duration = 5.0                # Duration of the trajectory in seconds
    num_points = 100              # Number of points in the trajectory
    human_position = [0.7, 0.0, 0.5]  # Position of the human
    target_pose = [0.6, 0.5, 0.5]     # Target position for the end-effector

    traj_gen = TrajectoryGenerator(start_pose, duration, num_points, human_position, target_pose)
    traj_gen.main()
    