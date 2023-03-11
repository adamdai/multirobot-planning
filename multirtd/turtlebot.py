"""Turtlebot class

"""

import numpy as np


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

        # Parameters
        self.dt = 0.1  # [s]
        self.sigma = 0.0  # assume same sigma for left and right wheel speeds
        self.wheelbase = 0.1  # [m]

        self.traj_idx = 0  # index of current trajectory point

        # Linear transformation from control to left and right wheel speeds
        self.u_to_lr = np.array([[1, -self.wheelbase / 2],
                                 [1,  self.wheelbase / 2]])

        # Optional components
        self.sensor = sensor
        self.controller = controller
        self.estimator = estimator


    def clear_history(self):
        self.x_hist = [self.x]