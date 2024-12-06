import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def generate_circular_trajectory(radius, t):
    omega = np.pi/150
    x_d = radius * np.cos(omega * t)
    y_d = radius * np.sin(omega * t)
    x_dp = -omega*radius*np.sin(omega*t)
    y_dp = omega*radius*np.cos(omega*t)
    x_dpp = -(omega**2)*radius*np.cos(omega*t)
    y_dpp = -(omega**2)*radius*np.sin(omega*t)
    return x_d, y_d, x_dp, y_dp, x_dpp, y_dpp

def control_law(x, y, theta, x_d, y_d, x_dp, y_dp, x_dpp, y_dpp, K1=1.0, K2=0.1, K3=1.0):
    e_x = x_d - x
    e_y = y_d - y
    e_theta = np.arctan2(y_dp, x_dp) - theta
    T_e = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    # print(f"==>> T_e: {T_e}")
    # print(f"==>> T_e.shape: {T_e.shape}")
    mat_q = T_e @ np.array([[e_x],[e_y],[e_theta]])
    # print(f"==>> mat_q: {mat_q}")
    # print(f"==>> mat_q.shape: {mat_q.shape}")

    v_r = np.sqrt(x_dp**2+y_dp**2)
    w_r = (x_dp*y_dpp - y_dp*x_dpp)/(x_dp**2 + y_dp**2)

    v_c = v_r*np.cos(mat_q[2,0]) + K1*mat_q[0,0]
    # print(f"==>> v_c: {v_c}")
    w_c = w_r + K2*v_r*mat_q[1,0] + K3*np.sin(mat_q[2,0])
    # print(f"==>> w_c: {w_c}")

    return v_c, w_c

# Parameters
radius = 10
total_time = 10
time_step = 0.01

# Initial robot state
x, y, theta = 0, 0, 0
x_traj, y_traj = [x], [y]
x_d_traj = [radius]
y_d_traj = [0]

for i in range(int(total_time/time_step)):
    x_d, y_d, x_dp, y_dp, x_dpp, y_dpp = generate_circular_trajectory(radius, i)
    v_c, w_c = control_law(x, y, theta, x_d, y_d, x_dp, y_dp, x_dpp, y_dpp, K1=50.0, K2=20.0, K3=20.0)
    x += v_c * np.cos(theta) * time_step
    y += v_c * np.sin(theta) * time_step
    theta += w_c * time_step
    x_traj.append(x)
    y_traj.append(y)
    x_d_traj.append(x_d)
    y_d_traj.append(y_d)

# Create the animation
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-radius * 1.5, radius * 1.5)
ax.set_ylim(-radius * 1.5, radius * 1.5)
ax.plot(x_d, y_d, 'r--', label="Desired Trajectory")
robot, = ax.plot([], [], 'bo', markersize=8, label="Robot")
robot_d, = ax.plot([], [], 'ro', markersize=5, label="Robot_desired")
trace, = ax.plot([], [], 'b-', lw=1)

def init():
    robot.set_data([], [])
    robot_d.set_data([], [])
    trace.set_data([], [])
    return robot, trace

def update(frame):
    robot.set_data(x_traj[frame], y_traj[frame])
    robot_d.set_data(x_d_traj[frame], y_d_traj[frame])
    trace.set_data(x_traj[:frame], y_traj[:frame])
    return robot, trace

ani = FuncAnimation(fig, update, frames=int(total_time/time_step), init_func=init, blit=True, repeat=False)
plt.title("Robot Following Circular Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()