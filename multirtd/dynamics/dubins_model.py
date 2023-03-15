"""Dubins dynamics model

"""

import numpy as np
from sympy import symbols, lambdify, Array, sin, cos, diff

import multirtd.params as params
from multirtd.reachability.dubins_reachability_shetty import dlqr_calculate


# Generate symbolic dynamics
x, y, th, v, w = symbols('x y th v w')

x0 = Array([x, y, th])
expr = x0
for i in range(params.TRAJ_IDX_LEN):
    expr = expr + params.DT * Array([v*cos(expr[2]), v*sin(expr[2]), w])

dubins = lambdify([x, y, th, v, w], expr)


def dubins_step(x, u, dt):
    """Run one step of dynamics

    Parameters
    ----------
    x : np.array
        State vector (x, y, theta)
    u : np.array
        Control vector (v, w)
    dt : float
        Time step

    Returns
    -------
    np.array
        Updated state vector (x, y, theta)

    """
    x_dot = u[0] * np.cos(x[2])
    y_dot = u[0] * np.sin(x[2])
    theta_dot = u[1]
    x_new = x + np.array([x_dot, y_dot, theta_dot]) * dt
    return x_new


def dubins_step_new(x, u, dt):
    if u[1] < 1e-6:
        return x + np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), 0]) * dt
    else:
        dx = u[0] / u[1] * (np.sin(x[2] + u[1] * dt) - np.sin(x[2]))
        dy = u[0] / u[1] * (-np.cos(x[2] + u[1] * dt) + np.cos(x[2]))
        dtheta = u[1] * dt
        return x + np.array([dx, dy, dtheta])
    

def dubins_traj(x0, u, N, dt):
    """Compute dubins trajectory from a sequence of controls
    
    Parameters
    ----------
    x0 : np.array
        Initial state vector (x, y, theta)
    u : np.array (2)
        Control input (v, w)
    N : int
        Number of steps
    dt : float
        Time step
    
    Returns
    -------
    np.array
        Trajectory (x, y, theta)
    
    """
    traj = np.zeros((N, 3))
    traj[0] = x0
    for i in range(1, N):
        traj[i] = dubins_step(traj[i-1], u, dt)
    return traj


def dubins_traj_new(x0, U, dt):
    traj = np.zeros((len(U), 3))
    traj[0] = x0
    for i in range(1, len(U)):
        traj[i] = dubins_step_new(traj[i-1], U[i-1], dt)
    return traj


def dubins_traj_fast(x0, u, N, dt):
    """Sample points along an arc of length vT and angle wT"""
    T = N * dt
    r = u[0] / u[1]
    angles = np.linspace(0, u[1] * T, N)
    # In robot local frame
    x = r * np.sin(angles)
    y = -r * (1 - np.cos(angles))
    # Rotate and translate to global frame
    x_global = x * np.cos(x0[2]) - y * np.sin(x0[2]) + x0[0]
    y_global = x * np.sin(x0[2]) + y * np.cos(x0[2]) + x0[1]
    theta = np.linspace(x0[2], x0[2] + u[1] * T, N)
    return np.stack([x_global, y_global, theta], axis=1)


def linearize_dynamics(x, u, dt):
    """Linearize dynamics around a point

    Parameters
    ----------
    x : np.array
        State vector (x, y, theta)
    u : np.array
        Control vector (v, w)
    dt : float
        Time step

    Returns
    -------
    np.array
        Linearized dynamics matrix

    """
    G_x = np.array([[1, 0, -u[0] * np.sin(x[2]) * dt],
                    [0, 1,  u[0] * np.cos(x[2]) * dt],
                    [0, 0, 1]])
    G_u = np.array([[np.cos(x[2]) * dt, -0.5 * u[0] * dt**2 * np.sin(x[2])],
                    [np.sin(x[2]) * dt,  0.5 * u[0] * dt**2 * np.cos(x[2])],
                    [0, dt]])
    return G_x, G_u


def linearize_dynamics_new(x, u, dt):
    if u[1] > 1e-6:
        theta = x[2] + u[1] * dt
        G_x = np.array([[1, 0, u[0] / u[1] * (np.cos(theta) - np.cos(x[2]))],
                        [0, 1, u[0] / u[1] * (np.sin(theta) - np.sin(x[2]))],
                        [0, 0, 1]])
        G_u = np.array([[(np.sin(theta) - np.sin(x[2])) / u[1], (u[0]/u[1]) * (dt * np.cos(theta) + (np.sin(x[2]) - np.sin(theta)) / u[1])],
                        [(np.cos(x[2]) - np.cos(theta)) / u[1], (u[0]/u[1]) * (dt * np.sin(theta) + (np.cos(theta) - np.cos(x[2])) / u[1])],
                        [0, dt]])
    else:
        G_x = np.array([[1, 0, -u[0] * np.sin(x[2]) * dt],
                        [0, 1,  u[0] * np.cos(x[2]) * dt],
                        [0, 0, 1]])
        G_u = np.array([[np.cos(x[2]) * dt, -0.5 * u[0] * dt**2 * np.sin(x[2])],
                        [np.sin(x[2]) * dt,  0.5 * u[0] * dt**2 * np.cos(x[2])],
                        [0, dt]])
    return G_x, G_u


def diff_drive_step(x, vl, vr, sigma, d, dt):
    """Diff drive model with uncertainty in the wheel velocities.

    d : distance between wheels

    """
    noise = np.random.normal(0, sigma, 2)
    dx = np.array([0.5 * (vl + vr + noise[0] + noise[1]) * np.cos(x[2]),
                    0.5 * (vl + vr + noise[0] + noise[1]) * np.sin(x[2]),
                    (vr - vl + noise[0] - noise[1]) / d])
    return x + dx * dt



class RangeSensor:
    """Range sensor class
    
    Sensor model for range measurements to landmarks

    init:
        parameters (landmark locations, noise sigma)
    get_measurement:

    linearize:
    
    """
    def __init__(self, landmarks, sigma):
        self.landmarks = landmarks
        self.sigma = sigma



class PositionSensor:
    """Position sensor class
    
    """
    def __init__(self, n, sigma):
        self.n = n                # Position dimension (i.e. 2D or 3D)
        self.sigma = sigma        # Measurement noise
                                  # - for now, assumes same sigma across dimensions
    
    def get_measurement(self, x):
        return x[:self.n] + np.random.normal(0, self.sigma, self.n)

    def linearize(self, x):
        H = np.zeros((self.n, len(x)))
        H[:self.n, :self.n] = np.eye(self.n)
        return H
    

class LQRController:
    """LQR controller class
    
    """
    def __init__(self, K):
        self.K = K
    
    def get_control(self, x):
        return -np.dot(self.K, x)
    

class DiffDriveDynamics:
    """Differential Drive dynamics class
    
    """
    def __init__(self, dt):
        # Parameters
        self.dt = 0.1  # [s]
        self.sigma = 0.0  # assume same sigma for left and right wheel speeds
        self.wheelbase = 0.1  # [m]

        # State
        self.x = x0  # [x, y, theta]
        self.x_hist = [x0]

        # Linear transformation from control to left and right wheel speeds
        self.u_to_lr = np.array([[1, -self.wheelbase / 2],
                                 [1,  self.wheelbase / 2]])
    
    def step(self, x, u):
        """Step forward dynamics

        Parameters
        ----------
        u : np.array
            Control vector (v, w)

        """
        v_lr = self.u_to_lr @ u
        self.x = diff_drive_step(self.x, v_lr[0], v_lr[1], self.sigma, self.wheelbase, self.dt)
        self.x_hist.append(self.x)
    
    def linearize(self, x, u):
        return linearize_dynamics(x, u, self.dt)
    

"""
Robot:
    Dynamics
    Sensor
    Controller(Dynamics)
    Estimator(Dynamics, Sensor)

"""


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


    # Open-loop
    def step(self, u):
        """Step forward dynamics

        Parameters
        ----------
        u : np.array
            Control vector (v, w)

        """
        v_lr = self.u_to_lr @ u
        self.x = diff_drive_step(self.x, v_lr[0], v_lr[1], self.sigma, self.wheelbase, self.dt)
        self.x_hist.append(self.x)


    # Closed-loop
    def track(self, x_nom, u_nom):
        """Track a nominal trajectory
        
        """
        # FIXME: move the LQR stuff to the controller class
        Q_lqr = np.eye(3)
        R_lqr = np.eye(2)

        i = self.traj_idx 
        x_est = self.estimator.x

        Q_motion = self.sigma**2 * np.eye(2)
        R_sense = self.sensor.sigma**2 * np.eye(4)

        # Linearize
        A, B = self.linearize(self.x, u)
        C = self.noise_matrix(self.x)
        H = self.sensor.linearize(self.x)

        # True dynamics
        K = dlqr_calculate(A, B, Q_lqr, R_lqr)
        u = u_nom[i] + K @ (x_nom[i] - x_est)
        v_lr = self.u_to_lr @ u
        self.x = diff_drive_step(self.x, v_lr[0], v_lr[1], self.sigma, self.wheelbase, self.dt)
        self.x_hist.append(self.x)

        # Measurement
        z = self.sensor.get_measurement(self.x)

        # Estimate
        # TODO: move this to the estimator class
        x_est = diff_drive_step(x_est, v_lr[0], v_lr[1], self.sigma, self.wheelbase, self.dt)
        P = A @ P @ A.T + C @ Q_motion @ C.T
        # Update step
        # H = get_beacon_jacobian(x_est, beacon_positions)
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R_sense)
        x_est = x_est + K @ (z - self.sensor.get_measurement(x_est))
        P = (np.eye(P.shape[0]) - K @ H) @ P


    
    def noise_matrix(self, x):
        """
        Generate matrix C for process noise

        x_{t+1} = f(x_t, u_t) + C * w_t
        TODO: check this explanation

        """
        C = np.array([[0.5 * self.dt * np.cos(x[2]), 0.5 * self.dt * np.cos(x[2])], 
                      [0.5 * self.dt * np.sin(x[2]), 0.5 * self.dt * np.sin(x[2])], 
                      [-self.dt / self.wheelbase, self.dt /self.wheelbase]])
        return C

    
    def linearize(self, x, u):
        """ """
        G_x = np.array([[1, 0, -u[0] * np.sin(x[2]) * self.dt],
                        [0, 1,  u[0] * np.cos(x[2]) * self.dt],
                        [0, 0, 1]])
        G_u = np.array([[np.cos(x[2]) * self.dt, -0.5 * u[0] * self.dt**2 * np.sin(x[2])],
                        [np.sin(x[2]) * self.dt,  0.5 * u[0] * self.dt**2 * np.cos(x[2])],
                        [0, self.dt]])
        return G_x, G_u