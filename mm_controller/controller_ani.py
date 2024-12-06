import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_circular_trajectory(radius, total_time, time_step):
    t = np.arange(0, total_time, time_step)
    omega = 2 * np.pi / total_time
    x_d = radius * np.cos(omega * t)
    y_d = radius * np.sin(omega * t)
    theta_d = omega * t  # Assuming a simple angular progression
    return t, x_d, y_d, theta_d

def control_law(x, y, theta, x_d, y_d, theta_d, Kp=1.0, Kd=0.1):
    e_x = x_d - x
    e_y = y_d - y
    e_distance = np.sqrt(e_x**2 + e_y**2)
    desired_angle = np.arctan2(e_y, e_x)
    e_theta = desired_angle - theta
    e_theta = np.arctan2(np.sin(e_theta), np.cos(e_theta))  # Normalize angle
    v = Kp * e_distance
    omega = Kp * e_theta + Kd * (theta_d - theta)
    return v, omega

# Parameters
radius = 5
total_time = 10
time_step = 0.1
t, x_d, y_d, theta_d = generate_circular_trajectory(radius, total_time, time_step)

# Initial robot state
x, y, theta = 0, -radius, 0
x_traj, y_traj = [x], [y]

for i in range(1, len(t)):
    v, omega = control_law(x, y, theta, x_d[i], y_d[i], theta_d[i])
    dt = time_step
    x += v * np.cos(theta) * dt
    y += v * np.sin(theta) * dt
    theta += omega * dt
    x_traj.append(x)
    y_traj.append(y)

# Create the animation
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-radius * 1.5, radius * 1.5)
ax.set_ylim(-radius * 1.5, radius * 1.5)
ax.plot(x_d, y_d, 'r--', label="Desired Trajectory")
robot, = ax.plot([], [], 'bo', markersize=8, label="Robot")
trace, = ax.plot([], [], 'b-', lw=1)

def init():
    robot.set_data([], [])
    trace.set_data([], [])
    return robot, trace

def update(frame):
    robot.set_data(x_traj[frame], y_traj[frame])
    trace.set_data(x_traj[:frame], y_traj[:frame])
    return robot, trace

ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, repeat=False)
plt.title("Robot Following Circular Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()