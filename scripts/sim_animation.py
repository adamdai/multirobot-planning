import plotly.graph_objects as go
import numpy as np

from multirtd.dynamics.diff_drive_dynamics import DiffDriveDynamics
from multirtd.plotting import plot_trajectories
from multirtd.agents import GreenAgent
from multirtd.planners.learned_planner import ImplicitPlanner
from multirtd.utils import rand_in_bounds


if __name__ == '__main__':

    BOUNDS = [-5.0, 5.0, -5.0, 5.0]  # xmin, xmax, ymin, ymax

    # Initialize agents, which act independently
    a1 = GreenAgent(x_init=rand_in_bounds(BOUNDS, 1).flatten())

    # Initialize planner, which takes agent states (position, heading, velocity, angular velocity) as inputs, 
    # and outputs a plan (sequence of controls over horizon)
    planner = ImplicitPlanner(x_init=rand_in_bounds(BOUNDS, 1).flatten())

    
    N = 100
    
    # For loop over time steps
    for t in range(N):
        # Step agents
        a1.step(t)

        # Feed agent states to planner
    

    x_hist = np.array(a1.x_hist)

    # Convert trajectories to the required format
    trajectories = [x_hist]

    # Plot trajectories
    fig = plot_trajectories(trajectories)

    # Set X and Y axis limits
    fig.update_xaxes(range=[-1.5, 1.5])
    fig.update_yaxes(range=[-1.5, 1.5])

    # Axes equal
    fig.update_layout(xaxis=dict(scaleanchor='y', scaleratio=1), yaxis=dict(scaleanchor='x', scaleratio=1))

    fig.show()