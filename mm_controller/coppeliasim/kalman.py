import numpy as np

class Kalman():
    def __init__(self):
        pass
    # State transition matrix (includes velocity integration)
    def kalman_init(self, dt):
        # Initialize variables
        n = 6  # State size: [x, y, z, vx, vy, vz]
        m = 3  # Control size: [ax, ay, az]
        k = 3  # Observation size: [x, y, z]
        # Initial covariance matrix (uncertainty)
        covariance = np.eye(n) * 1.0
        A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        # Control input matrix (maps acceleration to velocity)
        B = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]
        ])

        # Process noise covariance (assume small noise)
        Q = np.eye(n) * 0.1

        # Observation matrix (maps state to observed position)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])

        # Observation noise covariance
        R = np.eye(k) * 0.5

        return covariance, A, B, Q, H, R

    def kalman_filter_predict(self, states, covariances, A, B, u, Q):
        """
        Kalman filter prediction step.
        
        Parameters:
            states: Current state vector (n x 1)
            covariances: Current covariance matrix (n x n)
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            u: Control input vector (m x 1)
            Q: Process noise covariance matrix (n x n)

        Returns:
            predicted_state: Predicted state vector (n x 1)
            predicted_covariance: Predicted covariance matrix (n x n)
        """
        predicted_state = A @ states + B @ u
        predicted_covariance = A @ covariances @ A.T + Q
        return predicted_state, predicted_covariance

    def kalman_filter_update(self, predicted_state, predicted_covariance, H, z, R):
        """
        Kalman filter update step.

        Parameters:
            predicted_state: Predicted state vector (n x 1)
            predicted_covariance: Predicted covariance matrix (n x n)
            H: Observation matrix (k x n)
            z: Observation vector (k x 1)
            R: Observation noise covariance matrix (k x k)

        Returns:
            updated_state: Updated state vector (n x 1)
            updated_covariance: Updated covariance matrix (n x n)
        """
        S = H @ predicted_covariance @ H.T + R  # Residual covariance
        K = predicted_covariance @ H.T @ np.linalg.inv(S)  # Kalman gain
        y = z - H @ predicted_state  # Residual
        updated_state = predicted_state + K @ y
        updated_covariance = (np.eye(predicted_covariance.shape[0]) - K @ H) @ predicted_covariance
        return updated_state, updated_covariance