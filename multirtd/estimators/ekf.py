"""Extended Kalman Filter (EKF) class

"""

import numpy as np


class EKF:
    """Extended Kalman Filter (EKF) class
    
    """
    def __init__(self, dynamics, sensor, x_est0=None, P0=None, Q=None, R=None):
        """Initialize EKF
        
        Parameters
        ----------
        dynamics : Dynamics
            Dynamics model
        sensor : Sensor
            Sensor model
        Q : np.array
            Process noise covariance matrix
        R : np.array
            Measurement noise covariance matrix
        C : np.array
            Noise transformation matrix
        
        """
        self.dynamics = dynamics
        self.sensor = sensor

        if Q is None:
            self.Q = dynamics.sigma**2 * np.eye(dynamics.N_noise)
        else:
            self.Q = Q
        if R is None:
            self.R = sensor.sigma**2 * np.eye(sensor.n)
        else:
            self.R = R

        # Initialize state estimate
        if x_est0 is None:
            self.x_est = np.zeros(dynamics.N_dim)
        else:
            self.x_est = x_est0
        if P0 is None:
            self.P = np.eye(dynamics.N_dim)
        else:
            self.P = P0
        self.x_est_hist = [self.x_est]
        self.P_hist = [self.P]


    def reset(self, x_est0=None, P0=None):
        """Reset state estimate
        
        Parameters
        ----------
        x_est0 : np.array
            Initial state estimate
        P0 : np.array
            Initial state estimate covariance
        
        """
        if x_est0 is None:
            self.x_est = np.zeros(self.dynamics.N_dim)
        else:
            self.x_est = x_est0
        if P0 is None:
            self.P = np.eye(self.dynamics.N_dim)
        else:
            self.P = P0
        self.x_est_hist = [self.x_est]
        self.P_hist = [self.P]


    def update(self, u, z):
        """Update state estimate
        
        Parameters
        ----------
        u : np.array
            Control input
        z : np.array
            Measurement

        """
        # Assume linearize has been called
        A = self.dynamics.A
        C = self.dynamics.C
        H = self.sensor.H

        # Predict
        x_pred = self.dynamics.step(self.x_est, u, sigma=0.0)
        P_pred = A @ self.P @ A.T + C @ self.Q @ C.T

        # Update
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + self.R)
        self.x_est = x_pred + K @ (z - self.sensor.get_measurement(x_pred, sigma=0.0))
        self.P = (np.eye(self.dynamics.N_dim) - K @ H) @ P_pred

        # Save history
        self.x_est_hist.append(self.x_est)
        self.P_hist.append(self.P)


    def get_state(self):
        """Get state estimate
        
        """
        return self.x_est


    def get_covariance(self):
        """Get state estimate covariance
        
        """
        return self.P


    def clear_history(self):
        """Clear state estimate history
        
        """
        self.x_est_hist = [self.x_est]
        self.P_hist = [self.P]