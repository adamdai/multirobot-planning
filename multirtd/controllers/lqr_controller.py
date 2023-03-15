"""LQR controller class

"""

import numpy as np

from multirtd.reachability.dubins_reachability_shetty import dlqr_calculate


class LQRController:
    """LQR controller class

    Q and R are the cost matrices for the state and control inputs, respectively.
    
    """
    def __init__(self, dynamics, Q=None, R=None):
        self.dynamics = dynamics
        if Q is None:
            self.Q = np.eye(dynamics.N_dim)
        else:
            self.Q = Q
        if R is None:
            self.R = np.eye(dynamics.N_ctrl)
        else:
            self.R = R
    
    
    def compute_K(self):
        # Assumes linearize() has been called previously for this state
        A = self.dynamics.A
        B = self.dynamics.B
        K = dlqr_calculate(A, B, self.Q, self.R)
        return K


    def get_control(self, u_nom, x_nom, x_est):
        """Get control input
        
        """
        # Assumes linearize() has been called previously for this state
        A = self.dynamics.A
        B = self.dynamics.B
        K = dlqr_calculate(A, B, self.Q, self.R)
        return u_nom - K @ (x_est - x_nom)