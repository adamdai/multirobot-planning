"""Dubins dynamics model

"""

import numpy as np
from sympy import symbols, lambdify, Array, sin, cos, diff


# Generate symbolic dynamics
x, y, th, v, w = symbols('x y th v w')

dt = 0.1
N = 15
x0 = Array([x, y, th])
expr = x0
for i in range(N):
    expr = expr + dt * Array([v*cos(expr[2]), v*sin(expr[2]), w])

dubins = lambdify([x, y, th, v, w], expr)


def dubins_step(x, u, dt, sigma=np.zeros((3, 3))):
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
    x_new = x + np.array([x_dot, y_dot, theta_dot]) * dt + np.random.multivariate_normal(np.zeros(3), sigma)
    return x_new


def dubins_step_new(x, u, dt):
    if u[1] < 1e-6:
        return x + np.array([u[0] * np.cos(x[2]), u[0] * np.sin(x[2]), 0]) * dt
    else:
        dx = u[0] / u[1] * (np.sin(x[2] + u[1] * dt) - np.sin(x[2]))
        dy = u[0] / u[1] * (-np.cos(x[2] + u[1] * dt) + np.cos(x[2]))
        dtheta = u[1] * dt
        return x + np.array([dx, dy, dtheta])
    

def dubins_traj(x0, U, dt, sigma=np.zeros((3, 3))):
    """Compute dubins trajectory from a sequence of controls
    
    Parameters
    ----------
    x0 : np.array
        Initial state vector (x, y, theta)
    U : np.array
        Control sequence (v, w)
    dt : float
        Time step
    
    Returns
    -------
    np.array
        Trajectory (x, y, theta)
    
    """
    traj = np.zeros((len(U), 3))
    traj[0] = x0
    for i in range(1, len(U)):
        traj[i] = dubins_step(traj[i-1], U[i-1], dt, sigma)
    return traj


def dubins_traj_new(x0, U, dt):
    traj = np.zeros((len(U), 3))
    traj[0] = x0
    for i in range(1, len(U)):
        traj[i] = dubins_step_new(traj[i-1], U[i-1], dt)
    return traj


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