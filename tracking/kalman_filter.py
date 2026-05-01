import numpy as np


class KalmanFilter:
    """
    Tracks a bounding box using a constant-velocity linear Kalman filter.

    State vector  x = [cx, cy, w, h, vx, vy]  (6-D)
    Measurement   z = [cx, cy, w, h]           (4-D)

    All bounding-box values are expected in (cx, cy, w, h) form.
    """

    def __init__(self, initial_measurement):
        cx, cy, w, h = initial_measurement

        #  state vector: position + zero initial velocity 
        self.x = np.array([cx, cy, w, h, 0.0, 0.0], dtype=float)

        #  state covariance: high uncertainty on velocity at start 
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1000.0, 1000.0])

        #  state transition matrix (constant-velocity model) 
        # x_new = F @ x:  position += velocity * dt  (dt = 1 frame)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        #  measurement matrix: we observe position/size, not velocity 
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=float)

        #  process noise (Q): how much the motion model can drift 
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])

        #  measurement noise (R): how noisy the detector is 
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])

    # Predict step  (time update)
    def predict(self):
        """Project the state one step ahead and return the predicted bbox."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()  # [cx, cy, w, h]

    # Update step  (measurement update)
    def update(self, measurement):
        """Correct the prediction with a real detection and return the updated bbox."""
        z = np.array(measurement, dtype=float)

        # Innovation (residual between measurement and prediction)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P

        return self.x[:4].copy()  # [cx, cy, w, h]

    def get_state(self):
        """Return the full state vector [cx, cy, w, h, vx, vy]."""
        return self.x.copy()
