import numpy as np

def get_nearest_obstacle_distance(position, obstacles):
    """
    Calculate the distance to the nearest obstacle from a given position.
    
    Args:
        position (np.ndarray): The position from which to calculate the distance.
        obstacles (list): A list of obstacle positions.
        
    Returns:
        float: The distance to the nearest obstacle.
        index (int): The index of the nearest obstacle.
    """
    for obs in obstacles:
        obs[2] = position[2]  # Set the z-coordinate of the obstacle to the specified value
    distances = [np.linalg.norm(position - obs) for obs in obstacles]
    index = np.argmin(distances)
    x = position[0] - obstacles[index][0]
    y = position[1] - obstacles[index][1]
    z = position[2] - obstacles[index][2]
    x_norm = x / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    y_norm = y / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    z_norm = z / np.linalg.norm([x, y, z]) if np.linalg.norm([x, y, z]) != 0 else 0
    # Create a directional vector
    g_mat = np.zeros(3)  
    g_mat[0] = x_norm
    g_mat[1] = y_norm
    g_mat[2] = z_norm

    return min(distances), index, g_mat 


obstacles_positions = np.array([[0.0, 1.0, -0.2], [2.0, 1.0, -0.2], [-1.0, -1.0, -0.2]])
position = np.array([0.0, 0.0, 0.0])
nearest_distance, index, g_mat = get_nearest_obstacle_distance(position, obstacles_positions)
print(f"Nearest obstacle distance: {nearest_distance:.2f}")
print(f"Index of nearest obstacle: {index}")
print("Directional vector matrix:")
print(g_mat.shape)