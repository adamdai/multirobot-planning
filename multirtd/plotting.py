import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2
import plotly.graph_objects as go


### ---------------------- MATPLOTLIB ---------------------- ###

def plot_ellipse(ax, c, Sigma, conf=0.95):
    """Plot 2D confidence ellipse from center and covariance matrix
    
    """
    s = chi2.ppf(conf, 2)
    vals, vecs = np.linalg.eig(Sigma)
    vals = np.maximum(vals, 1e-6)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(s) * np.sqrt(vals)
    ellip = Ellipse(xy=c, width=width, height=height, angle=theta, alpha=0.5)
    ax.add_artist(ellip)


def plot_trajectory(ax, traj, **kwargs):
    """Plot a trajectory
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    traj : np.array
        Trajectory to plot
    **kwargs : dict
        Keyword arguments to pass to ax.plot()
        
    """
    ax.plot(traj[:,0], traj[:,1], **kwargs)


def plot_environment(ax, obstacles, unc_regions, start, goal):
    """Plot environment
    """
    ax.grid()
    ax.axis('equal')
    
    # Plot obstacles
    if obstacles is not None:
        for obs in obstacles:
            ax.add_patch(plt.Circle(tuple(obs[0]), obs[1], color='r', alpha=0.5, zorder=2))
    
    # Plot uncertainty regions
    if unc_regions is not None:
        for reg in unc_regions:
            ax.add_patch(plt.Circle(tuple(reg[0]), reg[1], color='r', alpha=0.1, zorder=2))

    # Plot goal
    ax.scatter(goal[0], goal[1], s=100, marker='*', color='g')

    # Plot start
    ax.plot(start[0], start[1], 'bo')

    return ax


### ---------------------- PLOTLY ---------------------- ###

def plot_trajectories(trajectories):
    # Create scatter traces for the particles
    N = len(trajectories[0])
    scatters = [go.Scatter(x=traj[:,0], y=traj[:,1], mode='markers') for traj in trajectories]

    # Create a figure and add frames
    fig = go.Figure(data=scatters)
    frames = [go.Frame(data=[go.Scatter(x=traj[:,0][:i+1], y=traj[:,1][:i+1], mode='markers') 
                             for traj in trajectories]) for i in range(N)]
    fig.frames = frames

    # Create animation settings and add to figure
    animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, 
                                        buttons=[dict(label='Play', method='animate', args=[None, animation_settings])])])
    return fig