import numpy as np

class AdmittanceController:
    def __init__(self, M, B, K, dt):
        """
        Initialize the Admittance Controller.

        Parameters:
            M: Diagonal mass matrix (6x6) [kg]
            B: Diagonal damping matrix (6x6) [Ns/m]
            K: Diagonal stiffness matrix (6x6) [N/m]
            dt: Time step [s]
        """
        self.M = M
        self.B = B
        self.K = K
        self.dt = dt

        # State variables
        self.position = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.velocity = np.zeros(6)  # [vx, vy, vz, omega_roll, omega_pitch, omega_yaw]

    def update(self, force_torque):
        """
        Update the position and velocity based on the force-torque input.

        Parameters:
            force_torque: External force-torque vector (6x1) [N, Nm]

        Returns:
            position: Updated position (6x1) [m, rad]
        """
        # Calculate acceleration: a = M^(-1) * (F - B*v - K*x)
        acc = np.linalg.inv(self.M) @ (force_torque - self.B @ self.velocity - self.K @ self.position)

        # Update velocity: v = v + a * dt
        self.velocity += acc * self.dt

        # Update position: x = x + v * dt
        self.position += self.velocity * self.dt

        return self.position

# Example usage
if __name__ == "__main__":
    # Define system parameters
    M = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Mass matrix
    B = np.diag([20.0, 20.0, 20.0, 2.0, 2.0, 2.0])  # Damping matrix
    K = np.diag([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])  # Stiffness matrix
    dt = 0.01  # Time step

    # Initialize controller
    admittance_controller = AdmittanceController(M, B, K, dt)

    # Simulated force-torque input (example)
    force_torque_input = np.array([5.0, 0.0, -2.0, 0.0, 0.1, 0.0])  # [Fx, Fy, Fz, Tx, Ty, Tz]

    # Update loop
    for step in range(100):
        position = admittance_controller.update(force_torque_input)
        print(f"Step {step}: Position: {position}")
