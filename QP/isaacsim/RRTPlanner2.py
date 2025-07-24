import numpy as np
import random
import math
from scipy.interpolate import CubicSpline

class RealTime3DTrajectoryPlanner:
    class Node:
        def __init__(self, pos, parent=None):
            self.pos = pos  # pos: np.array([x, y, z])
            self.parent = parent

    def __init__(self, map_size=20, num_obstacles=3, min_distance_between_obstacles=5,
                 obstacle_radius_range=(1, 3), step_size=1.0, goal_sample_rate=0.1, max_iter=1000,
                 goal_clearance=5.0, safety_margin=0.4, z_range=(0.0, 0.0)):
        self.map_size = map_size
        self.num_obstacles = num_obstacles
        self.min_distance_between_obstacles = min_distance_between_obstacles
        self.obstacle_radius_range = obstacle_radius_range
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.goal_clearance = goal_clearance
        self.safety_margin = safety_margin
        self.z_range = z_range  # (min_z, max_z)

        self.obstacles = []
        self.goal = None

    def generate_obstacles(self, goal=None):
        if goal is not None:
            self.goal = goal
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            # x = random.uniform(-self.map_size/2, self.map_size/2)
            # y = random.uniform(-self.map_size/2, self.map_size/2)
            x = random.uniform(-0.5, self.map_size/2)
            y = random.uniform(-0.5, self.map_size/2)
            z = random.uniform(*self.z_range)
            r = random.uniform(*self.obstacle_radius_range)
            candidate = (x, y, z, r)
            too_close = False

            robot_workspace_radius = 1.0  # 로봇 작업 공간 반경
            # 초기위치와 겹치는지 확인
            for ox, oy, oz, orad in obstacles:
                # dist_to_robot_center = np.linalg.norm(np.array([ox, oy, oz]) - robot_center)
                dist_to_robot_base = np.hypot(ox, oy)
                if dist_to_robot_base < orad + robot_workspace_radius :
                    too_close = True
                    break

            if self.goal is not None:
                dist_to_goal = np.linalg.norm(np.array([x, y]) - self.goal[:2])
                if dist_to_goal < self.goal_clearance + r:
                    too_close = True

            if not too_close:
                obstacles.append(candidate)
        self.obstacles = obstacles
        return np.array(self.obstacles)

    def is_in_collision(self, point):
        for ox, oy, oz, r in self.obstacles:
            dist_xy = np.hypot(point[0]-ox, point[1]-oy)
            effective_radius = r + self.safety_margin
            if dist_xy <= effective_radius:
                return TrueB
        return False

    def is_path_collision(self, p1, p2, step=0.2):
        dist = np.linalg.norm(p2 - p1)
        # print(f"Checking path collision from {p1} to {p2}, distance: {dist}")
        if dist < 1e-6:  # 너무 가까우면 점 충돌만 검사
            return self.is_in_collision(p1)
        num_steps = max(int(dist / step), 1)
        # print(f"Number of steps for collision check: {num_steps}, step size: {step}")
        for i in range(num_steps + 1):
            pos = p1 + i / num_steps * (p2 - p1)
            if self.is_in_collision(pos):
                return True
        return False

    def plan(self, start, goal):
        self.goal = goal
        nodes = [self.Node(start)]
        for _ in range(self.max_iter):
            if random.random() < self.goal_sample_rate:
                rand_point = goal
            else:
                x = random.uniform(-self.map_size/2, self.map_size/2)
                y = random.uniform(-self.map_size/2, self.map_size/2)
                z = random.uniform(*self.z_range)
                rand_point = np.array([x, y, z])
                if self.is_in_collision(rand_point):
                    continue

            dists = [np.linalg.norm(rand_point - node.pos) for node in nodes]
            nearest_node = nodes[np.argmin(dists)]
            direction = rand_point - nearest_node.pos
            if np.linalg.norm(direction) == 0:
                continue
            direction = direction / np.linalg.norm(direction)
            new_pos = nearest_node.pos + self.step_size * direction
            if self.is_in_collision(new_pos):
                continue
            if self.is_path_collision(nearest_node.pos, new_pos):
                continue

            new_node = self.Node(new_pos, nearest_node)
            nodes.append(new_node)

            if np.linalg.norm(new_pos - goal) < self.step_size:
                final_node = self.Node(goal, new_node)
                nodes.append(final_node)
                return self.extract_path(final_node)

        return None

    def extract_path(self, final_node):
        path = []
        node = final_node
        while node:
            path.append(node.pos)
            node = node.parent
        return path[::-1]

    def smooth_path(self, path, num_points=300):
        path = np.array(path)
        t = np.linspace(0, 1, len(path))
        cs_x = CubicSpline(t, path[:, 0])
        cs_y = CubicSpline(t, path[:, 1])
        cs_z = CubicSpline(t, path[:, 2])
        t_new = np.linspace(0, 1, num_points)
        smooth_x = cs_x(t_new)
        smooth_y = cs_y(t_new)
        smooth_z = cs_z(t_new)
        return np.vstack((smooth_x, smooth_y, smooth_z)).T

    def compute_curvature(self, smoothed_path):
        dx = np.gradient(smoothed_path[:,0])
        dy = np.gradient(smoothed_path[:,1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = (dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature[np.isnan(curvature)] = 0
        return np.abs(curvature)

    def generate_velocity_profile(self, smoothed_path, v_max=1.0, a_lat_max=0.5, a_long_max=1.0):
        curvature = self.compute_curvature(smoothed_path)
        v_curve = np.sqrt(np.maximum(a_lat_max / (curvature + 1e-6), 0))
        v_profile = np.minimum(v_curve, v_max)

        for i in range(1, len(v_profile)):
            ds = np.linalg.norm(smoothed_path[i] - smoothed_path[i-1])
            v_allow = np.sqrt(v_profile[i-1]**2 + 2 * a_long_max * ds)
            v_profile[i] = min(v_profile[i], v_allow)

        for i in reversed(range(len(v_profile)-1)):
            ds = np.linalg.norm(smoothed_path[i+1] - smoothed_path[i])
            v_allow = np.sqrt(v_profile[i+1]**2 + 2 * a_long_max * ds)
            v_profile[i] = min(v_profile[i], v_allow)

        return v_profile

    def generate_full_trajectory_with_offset(self, smoothed_path, v_profile, dt=0.1, time_offset=5):
        """
        사람의 궤적과 일정 시간 간격으로 로봇이 따라갈 궤적을 생성하는 함수.

        Args:
            smoothed_path (np.ndarray): 사람의 부드러운 궤적 (shape: num_points x 3).
            v_profile (np.ndarray): 사람의 속도 프로파일.
            dt (float): 시간 간격.
            time_offset (int): 로봇이 따라갈 궤적의 시간 오프셋.

        Returns:
            dict: 사람의 궤적과 로봇의 궤적을 포함하는 딕셔너리.
        """
        # 사람 궤적 생성
        distances = np.linalg.norm(smoothed_path[1:] - smoothed_path[:-1], axis=1)
        avg_v = (v_profile[1:] + v_profile[:-1]) / 2
        delta_t = distances / (avg_v + 1e-6)
        times = np.concatenate(([0], np.cumsum(delta_t)))

        dx = np.gradient(smoothed_path[:, 0])
        dy = np.gradient(smoothed_path[:, 1])
        headings = np.arctan2(dy, dx)
        heading_diff = np.gradient(headings)
        omega = heading_diff / (np.gradient(times) + 1e-6)

        total_time = times[-1]
        t_uniform = np.arange(0, total_time, dt)

        x_interp = np.interp(t_uniform, times, smoothed_path[:, 0])
        y_interp = np.interp(t_uniform, times, smoothed_path[:, 1])
        z_interp = np.interp(t_uniform, times, smoothed_path[:, 2])
        yaw_interp = np.interp(t_uniform, times, headings)
        v_interp = np.interp(t_uniform, times, v_profile)
        omega_interp = np.interp(t_uniform, times, omega)

        human_trajectory = {
            "time": t_uniform,
            "x": x_interp,
            "y": y_interp,
            "z": z_interp,
            "yaw": yaw_interp,
            "v": v_interp,
            "omega": omega_interp
        }

        # 로봇 궤적 생성 (time_offset 적용)
        robot_x = np.roll(x_interp, time_offset)
        robot_y = np.roll(y_interp, time_offset)
        robot_z = np.roll(z_interp, time_offset)

        # 초기 위치로 채우기 (time_offset 이전 값)
        robot_x[:time_offset] = x_interp[0]
        robot_y[:time_offset] = y_interp[0]
        robot_z[:time_offset] = z_interp[0]

        robot_trajectory = {
            "time": t_uniform,
            "x": robot_x,
            "y": robot_y,
            "z": robot_z,
            "yaw": yaw_interp,  # yaw는 동일하게 유지
            "v": v_interp,      # 속도는 동일하게 유지
            "omega": omega_interp  # 각속도는 동일하게 유지
        }

        return {"human_trajectory": human_trajectory, "robot_trajectory": robot_trajectory}

    def visualize_trajectory(self, trajectory):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(trajectory["x"], trajectory["y"], trajectory["z"], label="Trajectory")

        for ox, oy, oz, r in self.obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            xs = ox + r*np.cos(u)*np.sin(v)
            ys = oy + r*np.sin(u)*np.sin(v)
            zs = oz + r*np.cos(v)
            ax.plot_wireframe(xs, ys, zs, color="gray", alpha=0.5)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        plt.title("3D Trajectory")
        plt.legend()
        plt.show()
