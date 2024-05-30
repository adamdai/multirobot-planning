"""SimplePlanner class

This module defines the SimplePlanner class.

"""

import numpy as np
import torch

from multirtd.dynamics.diff_drive_dynamics import DiffDriveDynamics


class ImplicitPlanner:
    """

    Inputs: other agent states
    Outputs: sequence of controls (plan)

    Encapsulates ego agent

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, x_init):

        self.horizon = 10  # time steps
        self.x = x_init
        self.x_hist = []

        self.dynamics = DiffDriveDynamics()

        # TODO: define architecture of planner 
        # (1-layer MLP w/256 neurons)
        self.network = torch.nn.Module  



    def replan(self, agent_states):
        """Replan based on other agents

        Parameters
        ----------
        agent_states : list
            List of agent states (ordered by class)

        Returns
        -------
        np.array 
            Sequence of control actions (plan)
        
        """
        pass
        # Wrap agent_states into tensor
        states_tensor = torch.tensor(agent_states)

        # Inference
        out = self.network(states_tensor)

        # Transform output however necessary
        
        # Get first control and apply it
        u = out[0]
        self.x = self.dynamics.step(self.x, u)
        self.x_hist.append(self.x)


    
    