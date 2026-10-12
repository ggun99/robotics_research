import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformBroadcaster
try:
	from geometry_msgs.msg import PoseStamped
	from sensor_msgs.msg import PointCloud
except Exception:  # pragma: no cover - allows running outside ROS
	PoseStamped = None
	PointCloud = None


class IncomingPointsPlotter(Node):
	def __init__(self):
		super().__init__('incoming_points_plotter')
		self.cable_points = np.empty((0, 3), dtype=float)
		self.human_position = None
		self.tf_broadcaster = TransformBroadcaster(self)  # TF 브로드캐스터 초기화
		# 각각의 데이터 수신 여부를 확인하기 위한 플래그로 분리
		self.received_cable = False
		self.received_human = False

		if PointCloud is not None and PoseStamped is not None:
			self.cable_sub = self.create_subscription(PointCloud, '/pointcloud', self.set_cable_positions, 10)
			self.human_sub = self.create_subscription(PoseStamped, '/hand/pose', self.set_human_position, 10)
		self.create_timer(0.05, self.svd_pub)  # 0.1초마다 svd_pub 호출
		self.target_pose_pub = self.create_publisher(PoseStamped, '/target/pose', 10)
	def set_cable_positions(self, msg):
		points = [[point.x, point.y, point.z] for point in msg.points]
		self.cable_points = np.asarray(points, dtype=float) if points else np.empty((0, 3), dtype=float)
		if len(self.cable_points) > 0:
			# self.get_logger().info(f'Received {len(self.cable_points)} cable points.')
			self.received_cable = True # 케이블 플래그 변경

	def set_human_position(self, msg):
		position = msg.pose.position
		self.human_position = np.array([position.x, position.y, position.z], dtype=float)
		# self.get_logger().info('Received human hand position.')
		self.received_human = True # 사람 플래그 변경

	@staticmethod
	def _normalize(vector):
		vector = np.asarray(vector, dtype=float)
		norm = np.linalg.norm(vector)
		if np.isclose(norm, 0.0):
			raise ValueError("Zero-length vector cannot be normalized.")
		return vector / norm
			
	def svd_pub(self):
		# 1. 데이터 확인
		if len(self.cable_points) == 0 and self.human_position is None:
			self.get_logger().warn('No data available to plot SVD.')
			return

		cable_len = len(self.cable_points)
		total_points = np.vstack([self.cable_points, self.human_position[None, :]]) if self.human_position is not None else self.cable_points
		
		# 2. 데이터 중심화 및 SVD 투영
		centroid = total_points.mean(axis=0)
		centered_points = total_points - centroid

		U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
		# 두 주성분 벡터 (2D 평면의 기저)
		pc1 = Vt[0, :]
		pc2 = Vt[1, :]
		# 세 번째 주성분 벡터 (평면의 법선 벡터 -> z_axis로 사용)
		pc3 = Vt[2, :]

		x_proj = centered_points @ pc1
		y_proj = centered_points @ pc2

		# 3. 일반 2차 곡선 최소자승법 피팅
		M = np.column_stack((x_proj**2, x_proj * y_proj, y_proj**2, x_proj, y_proj))
		b_vec = np.ones_like(x_proj)
		coeffs, _, _, _ = np.linalg.lstsq(M, b_vec, rcond=None)
		A, B, C, D, E = coeffs

		# 4. 다음 점 예측 로직 (포물선 추적 방식)
		next_point_2d = None
		p_end = None
		if cable_len >= 2 and self.human_position is not None:
			pts_cable = np.column_stack((x_proj[:cable_len], y_proj[:cable_len]))
			pt_human = np.array([x_proj[cable_len], y_proj[cable_len]])

			# 반대편 끝 점 찾기
			dists_from_human = np.linalg.norm(pts_cable - pt_human, axis=1)
			far_idx = np.argmax(dists_from_human)
			p_end = pts_cable[far_idx]

			# 평균 간격 계산
			from scipy.spatial.distance import cdist
			if len(pts_cable) > 1:
				d_matrix = cdist(pts_cable, pts_cable)
				np.fill_diagonal(d_matrix, np.inf)
				avg_dist = np.mean(np.min(d_matrix, axis=1))
			else:
				avg_dist = 0.1

			# 끝 점 주변 실제 방향 트렌드 구하기
			d_from_end = np.linalg.norm(pts_cable - p_end, axis=1)
			d_from_end[far_idx] = np.inf
			near_to_end_idx = np.argmin(d_from_end)
			p_near = pts_cable[near_to_end_idx]

			dir_vec = p_end - p_near
			dir_vec_norm = dir_vec / np.linalg.norm(dir_vec) if np.linalg.norm(dir_vec) > 0 else np.array([1.0, 0.0])

			# 대략적인 임시 점 생성 후, 포물선 선 위로 보정
			rough_next = p_end + dir_vec_norm * avg_dist

			search_res = 100
			local_x = np.linspace(rough_next[0] - avg_dist, rough_next[0] + avg_dist, search_res)
			local_y = np.linspace(rough_next[1] - avg_dist, rough_next[1] + avg_dist, search_res)
			LX, LY = np.meshgrid(local_x, local_y)
			
			LZ = A*LX**2 + B*LX*LY + C*LY**2 + D*LX + E*LY - 1
			dist_from_end_grid = np.sqrt((LX - p_end[0])**2 + (LY - p_end[1])**2)
			cost = np.abs(LZ) * 10.0 + np.abs(dist_from_end_grid - avg_dist)
			
			min_idx = np.unravel_index(np.argmin(cost), cost.shape)
			next_point_2d = np.array([LX[min_idx], LY[min_idx]])

		# -------------------------------------------------------------
		# 5. [신규 추가] 2D 예측 점을 3D 좌표 공간으로 복원 및 TF 발행
		if next_point_2d is not None and p_end is not None:
			# 5-1. 2D 좌표를 3D 벡터 공간으로 변환 (기저 벡터 pc1, pc2 결합)
			# 원래 식: centered_point = x_proj * pc1 + y_proj * pc2
			predicted_centered_3d = next_point_2d[0] * pc1 + next_point_2d[1] * pc2
			
			# 중심점(centroid)을 다시 더해 최종 world 기준 3D 위치 복원
			predicted_position_3d = predicted_centered_3d + centroid
			self.get_logger().info(f"Predicted 3D Position: {predicted_position_3d}")

			# 5-2. TF에 전달할 회전 축(Orientation) 설계
			# X축: 끝 점(p_end)에서 예측 점(next_point_2d)으로 향하는 3D 방향 벡터 (전진 접선)
			p_end_centered_3d = p_end[0] * pc1 + p_end[1] * pc2
			x_axis_3d = predicted_centered_3d - p_end_centered_3d
			
			# Z축: SVD 평면의 법선 벡터 (케이블 평면의 수직축)
			z_axis_3d = pc3
			
			# Y축: 오른손 법칙에 부합하도록 두 축을 외적(Cross Product)하여 결정
			y_axis_3d = np.cross(z_axis_3d, x_axis_3d)

			# 5-3. 기존에 가지고 계시던 TF 발행 함수 호출
			self._publish_desired_tf(predicted_position_3d, x_axis_3d, y_axis_3d, z_axis_3d)
			# self.get_logger().info("Successfully published desired_pose TF.")
		

		
	def _publish_desired_tf(self, position, x_axis, y_axis, z_axis):
		if self.tf_broadcaster is None or TransformStamped is None:
			return

		transform = TransformStamped()
		transform.header.stamp = self.get_clock().now().to_msg()
		transform.header.frame_id = 'world'
		transform.child_frame_id = 'desired_pose'
		transform.transform.translation.x = float(position[0])
		transform.transform.translation.y = float(position[1])
		transform.transform.translation.z = float(position[2])

		rotation_matrix = np.column_stack([
			self._normalize(-x_axis),
			self._normalize(z_axis),
			self._normalize(y_axis),
		])
		quat = R.from_matrix(rotation_matrix).as_quat()
		transform.transform.rotation.x = float(quat[0])
		transform.transform.rotation.y = float(quat[1])
		transform.transform.rotation.z = float(quat[2])
		transform.transform.rotation.w = float(quat[3])

		self.tf_broadcaster.sendTransform(transform)

		transform_msg = PoseStamped()
		transform_msg.header.stamp = transform.header.stamp
		transform_msg.header.frame_id = transform.header.frame_id
		transform_msg.pose.position.x = transform.transform.translation.x
		transform_msg.pose.position.y = transform.transform.translation.y
		transform_msg.pose.position.z = transform.transform.translation.z
		transform_msg.pose.orientation.x = transform.transform.rotation.x
		transform_msg.pose.orientation.y = transform.transform.rotation.y
		transform_msg.pose.orientation.z = transform.transform.rotation.z
		transform_msg.pose.orientation.w = transform.transform.rotation.w
		self.target_pose_pub.publish(transform_msg)

            
	


def main():
	rclpy.init()
	node = IncomingPointsPlotter()
	
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == '__main__':
	main()