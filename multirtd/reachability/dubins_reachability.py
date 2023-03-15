"""Dubins reachability functions

"""

import numpy as np

import multirtd.params as params
from multirtd.reachability.zonotope import Zonotope
from multirtd.utils import remove_zero_columns


def RRBT(x_nom, u_nom, Sigma0, Qs, Rs, agent):
    """RRBT for a dubins nominal trajectory

    Dynamics and sensor
    Sequence of motion and sensing sigmas (Qs and Rs)

    Parameters
    ----------
    x_nom : np.array
        Nominal trajectory (N, 3)
    u_nom : np.array
        Nominal control inputs (N, 2)
    Z0 : Zonotope
        Initial zonotope

    """
    N = len(x_nom)

    Sigma = Sigma0  # state estimation covariance
    Lambda = np.zeros((3, 3))  # uncertainty from not having yet taken observations

    RRBT_Sigmas = [Sigma0]

    for i in range(1,N):        
        # Linearize about nominal trajectory
        A, B = agent.dynamics.linearize(x_nom[i], u_nom[i])
        C = agent.dynamics.noise_matrix(x_nom[i])
        H = agent.sensor.linearize(x_nom[i])
        K = agent.controller.compute_K()
        
        # Covariance prediction
        Sigma = A @ Sigma @ A.T + C @ Qs[i] @ C.T
        L = Sigma @ H.T @ np.linalg.inv(H @ Sigma @ H.T + Rs[i])

        # Covariance update
        Lambda = (A - B @ K) @ Lambda @ (A - B @ K).T + L @ H @ Sigma
        Sigma = Sigma - L @ H @ Sigma

        RRBT_Sigmas.append(Sigma + Lambda)
