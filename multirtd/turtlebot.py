"""Turtlebot class

"""

import numpy as np

from multirtd.dynamics.diff_drive_dynamics import DiffDriveDynamics
from multirtd.sensors.position_sensor import PositionSensor
from multirtd.controllers.lqr_controller import LQRController
from multirtd.estimators.ekf import EKF


class Turtlebot:
    """Turtlebot class
    
    Modeled as differential drive with process noise in left and right wheel speeds

    init: 
        define state, state history, parameters (dt, noise sigma, wheelbase)
        define sensor (optional)
        define controller (optional, e.g. LQR)
        define state estimator (optional, e.g. EKF)
    step: 
        step forward dynamics
    linearize: 
        linearize dynamics around a point

    """

    def __init__(self, x0=np.zeros(3), dynamics=None, sensor=None, controller=None, estimator=None):
        # State
        self.x = x0  # [x, y, theta]
        self.x_hist = [x0]

        self.traj_idx = 0  # index of current trajectory point

        if dynamics is not None:
            self.dynamics = dynamics
        else:
            self.dynamics = DiffDriveDynamics(sigma=0.01)

        if sensor is not None:
            self.sensor = sensor
        else:
            self.sensor = PositionSensor(n=2, sigma=0.1)

        if controller is not None:
            self.controller = controller
        else:
            self.controller = LQRController(self.dynamics)
        
        if estimator is not None:
            self.estimator = estimator
        else:
            self.estimator = EKF(self.dynamics, self.sensor, x_est0=x0, P0=0.1*np.eye(3))


    def reset(self, x0, P0):
        self.x = x0
        self.x_hist = [x0]
        self.traj_idx = 0
        self.estimator.reset(x0, P0)


    def clear_history(self):
        self.x_hist = [self.x]


    def track(self, x_nom, u_nom):
        """Track a nominal trajectory"""
        N = len(u_nom)

        for i in range(1, N):
            # Linearize about nominal trajectory
            self.dynamics.linearize(x_nom[i], u_nom[i])
            self.dynamics.noise_matrix(x_nom[i])
            self.sensor.linearize(x_nom[i])

            # Control
            u = self.controller.get_control(u_nom[i], x_nom[i], self.estimator.x_est)
            self.x = self.dynamics.step(self.x, u)

            # Measurement
            z = self.sensor.get_measurement(self.x)
            
            # EKF
            self.estimator.update(u, z)

            self.x_hist.append(self.x)