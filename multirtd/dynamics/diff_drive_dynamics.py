"""Differential drive dynamics class

"""

import numpy as np


def diff_drive_step(x, vl, vr, sigma, d, dt):
    """Diff drive model with uncertainty in the wheel velocities.

    d : distance between wheels

    """
    noise = np.random.normal(0, sigma, 2)
    dx = np.array([0.5 * (vl + vr + noise[0] + noise[1]) * np.cos(x[2]),
                    0.5 * (vl + vr + noise[0] + noise[1]) * np.sin(x[2]),
                    (vr - vl + noise[0] - noise[1]) / d])
    return x + dx * dt


class DiffDriveDynamics:
    """Differential Drive dynamics class

    State: x = [x, y, theta]
        x - x position (meters)
        y - y position (meters)
        theta - heading angle (radians)
    Control: u = [v, w]
        v - linear velocity (m/s)
        w - angular velocity (rad/s)

    Noise: additive Gaussian noise on left and right wheel speeds
    
    """
    def __init__(self, dt=0.1, sigma=0.0, wheelbase=0.1):
        """Initialize dynamics
        
        Parameters
        ----------
        dt : float
            Time discretization [s]
        sigma : float
            Standard deviation of Gaussian noise on left and right wheel speeds
        wheelbase : float
            Distance between wheels [m]

        """
        # Constants
        self.N_dim = 3  # state dimension
        self.N_ctrl = 2  # control dimension
        self.N_noise = 2  # noise dimension

        # Parameters
        self.dt = dt  # [s]
        self.sigma = sigma  # assume same sigma for left and right wheel speeds
        self.wheelbase = wheelbase  # [m]

        # Linear transformation from control to left and right wheel speeds
        self.u_to_lr = np.array([[1, -self.wheelbase / 2],
                                 [1,  self.wheelbase / 2]])

        # Linearized dynamics matrices (updated when linearize() is called)
        self.A = None
        self.B = None
        self.C = None
    

    def step(self, x, u, sigma=None):
        """Step forward dynamics

        Use sigma=0.0 for expected dynamics model

        Parameters
        ----------
        x : np.array
            State vector
        u : np.array
            Control vector (v, w)
        sigma : float
            Standard deviation of Gaussian noise on left and right wheel speeds

        """
        if sigma is None:
            sigma = self.sigma
        v_lr = self.u_to_lr @ u
        return diff_drive_step(x, v_lr[0], v_lr[1], sigma, self.wheelbase, self.dt)
    

    def linearize(self, x, u):
        """Linearize Dynamics """
        G_x = np.array([[1, 0, -u[0] * np.sin(x[2]) * self.dt],
                        [0, 1,  u[0] * np.cos(x[2]) * self.dt],
                        [0, 0, 1]])
        G_u = np.array([[np.cos(x[2]) * self.dt, -0.5 * u[0] * self.dt**2 * np.sin(x[2])],
                        [np.sin(x[2]) * self.dt,  0.5 * u[0] * self.dt**2 * np.cos(x[2])],
                        [0, self.dt]])
        self.A = G_x
        self.B = G_u
        return G_x, G_u

    
    def noise_matrix(self, x):
        """
        Generate matrix C for process noise

        x_{t+1} = f(x_t, u_t) + C * w_t
        TODO: check this explanation

        This matrix transforms left/right wheel speed noise in R^2 to state noise in R^3

        """
        C = np.array([[0.5 * self.dt * np.cos(x[2]), 0.5 * self.dt * np.cos(x[2])], 
                      [0.5 * self.dt * np.sin(x[2]), 0.5 * self.dt * np.sin(x[2])], 
                      [-self.dt / self.wheelbase, self.dt /self.wheelbase]])
        self.C = C
        return C
    

    def nominal_traj(self, x0, u, N):
        """Compute nominal trajectory with no noise 
        
        Applies constant control u for N timesteps

        Parameters
        ----------
        x0 : np.array
            Initial state
        u : np.array
            Control vector (v, w)

        Returns
        -------
        traj : np.array
            Trajectory (N, 3)
        
        """
        v_lr = self.u_to_lr @ u
        traj = np.zeros((N, 3))
        traj[0] = x0
        for i in range(1, N):
            traj[i] = diff_drive_step(traj[i-1], v_lr[0], v_lr[1], 0.0, self.wheelbase, self.dt)
        return traj