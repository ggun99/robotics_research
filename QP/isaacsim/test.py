import numpy as np

# def get_nearest_obstacle_distance(position, obstacles):
#     """
#     Calculate the distance to the nearest obstacle from a given position.
    
#     Args:
#         position (np.ndarray): The position from which to calculate the distance.
#         obstacles (list): A list of obstacle positions.
        
#     Returns:
#         float: The distance to the nearest obstacle.
#         index (int): The index of the nearest obstacle.
#     """
#     for obs in obstacles:
#         obs[2] = position[2]  # Set the z-coordinate of the obstacle to the specified value
#     distances = [np.linalg.norm(position - obs) for obs in obstacles]
#     index = np.argmin(distances)
#     x = position[0] - obstacles[index][0]
#     y = position[1] - obstacles[index][1]
#     z = position[2] - obstacles[index][2]
#     x_norm = x / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
#     y_norm = y / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
#     z_norm = z / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
#     # Create a directional vector
#     g_mat = np.zeros(3)  
#     g_mat[0] = x_norm
#     g_mat[1] = y_norm
#     g_mat[2] = z_norm

#     return min(distances), index, g_mat 


# obstacles_positions = np.array([[0.0, 1.0, -0.2], [2.0, 1.0, -0.2], [-1.0, -1.0, -0.2]])
# position = np.array([0.0, 0.0, 0.0])
# nearest_distance, index, g_mat = get_nearest_obstacle_distance(position, obstacles_positions)
# print(f"Nearest obstacle distance: {nearest_distance:.2f}")
# print(f"Index of nearest obstacle: {index}")
# print("Directional vector matrix:")
# print(g_mat.shape)


def generate_points_between_positions(start_pos, end_pos, num_points=10):
    """
    두 3차원 위치를 이어주는 선에서 일정한 간격으로 점을 생성하는 함수.

    Args:
        start_pos (np.ndarray): 시작 위치 (3차원 좌표).
        end_pos (np.ndarray): 끝 위치 (3차원 좌표).
        num_points (int): 생성할 점의 개수 (기본값: 10).

    Returns:
        np.ndarray: 생성된 점들의 좌표 배열 (shape: num_points x 3).
    """
    # 시작 위치와 끝 위치를 연결하는 선을 따라 일정한 간격으로 점 생성
    points = np.linspace(start_pos, end_pos, num_points)
    return points
# 시작 위치와 끝 위치 정의
start_position = np.array([0.0, 0.0, 0.0])
end_position = np.array([1.0, 1.0, 1.0])

# 점 생성
points = generate_points_between_positions(start_position, end_position, num_points=10)

# 결과 출력
print("Generated Points:")
print(points)
print("Shape of generated points:", points.shape)