import numpy as np
import time
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from multirtd.LPM import LPM
import multirtd.params as params

class Trajectory:
    """Trajectory class

    Planned trajectory
    
    Attributes
    ----------
    length : int
        Length of trajectory (i.e. number of timesteps)
    N_dim : int
        State dimension of the trajectory (i.e. 2D or 3D)
    t : np.array (1 x N)
        Time array
    p : np.array (N_dim x N)
        Positions
    v : np.array (N_dim x N)
        Velocities
    a : np.array (N_dim x N)
        Accelerations

    """

    def __init__(self, T, N_dim):
        """Initialize a trajectory with 0 position, velocity, and acceleration"""
        self.T = T
        self.length = len(T)
        self.N_dim = N_dim
        self.P = np.zeros((N_dim, self.length))
        self.V = np.zeros((N_dim, self.length))
        self.A = np.zeros((N_dim, self.length))


def remove_zero_columns(A):
    """Remove all zeros columns from an array
    
    Parameters
    ----------
    A : np.array (2D)
        Input array
    
    Returns
    -------
    np.array 
        Array with all zeros columns removed

    """
    zero_idx = np.argwhere(np.all(A[...,:]==0, axis=0))
    return np.delete(A, zero_idx, axis=1)


def check_obs_collision(positions, obs, r_collision):
    """Check a sequence of positions against a single obstacle for collision.

    Obstacles are cylinders represented as (center, radius)

    Parameters
    ----------
    positions : np.array
    obs : tuple

    Returns
    -------
    bool
        True if the plan is safe, False is there is a collision

    """
    c_obs, r_obs = obs
    d_vec = np.linalg.norm(positions - c_obs, axis=1)
    if any(d_vec <= r_collision + r_obs):
        return False
    else:
        return True


def rand_in_bounds(bounds, n):
    """Generate random samples within specified bounds

    Parameters
    ----------
    bounds : list
        List of min and max values for each dimension.
    n : int
        Number of points to generate.

    Returns
    -------
    np.array 
        Random samples

    """
    x_pts = np.random.uniform(bounds[0], bounds[1], n)
    y_pts = np.random.uniform(bounds[2], bounds[3], n)
    # 2D 
    if len(bounds) == 4:
        return np.hstack((x_pts[:,None], y_pts[:,None]))
    # 3D
    elif len(bounds) == 6:
        z_pts = np.random.uniform(bounds[4], bounds[5], n)
        return np.hstack((x_pts[:,None], y_pts[:,None], z_pts[:,None]))
    else:
        raise ValueError('Please pass in bounds as either [xmin xmax ymin ymax] '
                            'or [xmin xmax ymin ymax zmin zmax] ')


def prune_vel_samples(V, v_0, max_norm, max_delta):
    """Prune Velocity Samples
    
    """
    V_mag = np.linalg.norm(V, axis=1)
    delta_V = np.linalg.norm(V - v_0, axis=1)
    keep_idx = np.logical_and(V_mag < max_norm, delta_V < max_delta)
    return V[keep_idx]


def dlqr_calculate(G, H, Q, R):
    """
    Discrete-time Linear Quadratic Regulator calculation.
    State-feedback control  u[k] = -K*x[k]
    Implementation from  https://github.com/python-control/python-control/issues/359#issuecomment-759423706
    How to apply the function:    
        K = dlqr_calculate(G,H,Q,R)
        K, P, E = dlqr_calculate(G,H,Q,R, return_solution_eigs=True)
    Inputs:
      G, H, Q, R  -> all numpy arrays  (simple float number not allowed)
      returnPE: define as True to return Ricatti solution and final eigenvalues
    Returns:
      K: state feedback gain
      P: Ricatti equation solution
      E: eigenvalues of (G-HK)  (closed loop z-domain poles)
      
    """
    from scipy.linalg import solve_discrete_are, inv
    P = solve_discrete_are(G, H, Q, R)  #Solução Ricatti
    K = inv(H.T@P@H + R)@H.T@P@G    #K = (B^T P B + R)^-1 B^T P A 

    return K


def normalize(v):
    """Normalize a vector

    Parameters
    ----------
    v : np.array
        Vector to normalize

    Returns
    -------
    np.array
        Normalized vector

    """
    if np.linalg.norm(v) == 0:
        return v
    return v / np.linalg.norm(v)


def signed_angle_btwn_vectors(v1, v2):
    """Signed angle between two 2D vectors

    Counter-clockwise is positive, clockwise is negative.

    Parameters
    ----------
    v1 : np.array
        Vector 1
    v2 : np.array
        Vector 2

    Returns
    -------
    float
        Angle between vectors in radians

    """
    v1_ = normalize(v1.flatten())
    v2_ = normalize(v2.flatten())
    return np.sign(np.cross(v1_, v2_)) * np.arccos(np.dot(v1_, v2_))


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


def rot_mat_2D(theta):
    """2D rotation matrix
    
    Parameters
    ----------
    theta : float
        Rotation angle in radians
    
    Returns
    -------
    np.array
        2D rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])