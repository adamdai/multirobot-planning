"""Position sensor class

"""

import numpy as np


class PositionSensor:
    """Position sensor class
    
    """
    def __init__(self, n=2, sigma=0.1):
        self.n = n                # Position dimension (i.e. 2D or 3D)
        self.sigma = sigma        # Measurement noise
                                  # - for now, assumes same sigma across dimensions
    
    def get_measurement(self, x, sigma=None):
        """Get measurement from state

        Use sigma=0.0 for expected measurement model
        
        Parameters
        ----------
        x : np.array
            State vector
        sigma : float
            Measurement noise
        
        """
        if sigma is None:
            sigma = self.sigma
        return x[:self.n] + np.random.normal(0, sigma, self.n)

    def linearize(self, x):
        H = np.zeros((self.n, len(x)))
        H[:self.n, :self.n] = np.eye(self.n)
        return H