import numpy as np

from multirtd.dynamics.diff_drive_dynamics import DiffDriveDynamics


# TODO: make all agents inherit from Agent base class
class Agent:
    def __init__(self, x_init=np.zeros(2)):
        pass
    def step(self):
        pass


class RedAgent:
    """
    Jerky dynamics. 

    """
    def __init__(self, x_init=np.zeros(2)):
        # Intialize 
        pass

    def step(self):
        pass


class GreenAgent:
    """
    Smooth (Dubins) dynamics. 
    Constant v, switches omega every 2 seconds

    """
    def __init__(self, x_init=np.zeros(3)):
        self.dt = 0.1
        self.x = x_init
        self.x_hist = []
        self.t_switch = 2.0  # seconds
        self.n_switch = self.t_switch // self.dt  # timesteps
        self.dynamics = DiffDriveDynamics()

        self.V = 1.0  # m/s
        self.OMEGA_RANGE = [-0.5, 0.5]  # rad/s
        self.omega = np.random.rand() # TODO
        self.u = np.array([self.V, self.omega])

    def step(self, t):
        if t % self.n_switch == 0:
            self.omega = np.random.rand() # TODO
            self.u = np.array([self.V, self.omega])
        
        self.x = self.dynamics.step(self.x, self.u)
        self.x_hist.append(self.x)
