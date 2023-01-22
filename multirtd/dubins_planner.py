"""SimplePlanner class

This module defines the SimplePlanner class.

"""

import numpy as np
import time 

import multirtd.params as params
from multirtd.LPM import LPM
import multirtd.utils as utils


class DubinsPlanner:
    """Dubins Planner class

    Modified linear planner for dubin's vehicle

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, lpm_file, p_0):
        # Initialize LPM object
        self.LPM = LPM(lpm_file)
        self.N_T_PLAN = len(self.LPM.time)  # planned trajectory length
        self.DT = self.LPM.t_sample  # trajectory discretization time interval

        # Initial conditions [m],[m/s],[m/s^2]
        self.p_0 = p_0
        self.v_0 = np.zeros(params.N_DIM)
        self.a_0 = np.zeros(params.N_DIM)

        self.theta_0 = 0  # Initial heading, 0 is along x-axis, positive is CCW, -pi to pi

        # Goal position [m]
        self.p_goal = np.zeros(params.N_DIM)

        # Obstacles
        self.obstacles = []


    def check_obstacle_collisions(self, positions):
        """ Check a sequence of positions against the current list of nearby obstacles for collision.

        Parameters
        ----------
        positions : np.array
            Trajectory positions to check against obstacles.

        Returns
        -------
        bool
            True if plan is safe, False if there is a collision.

        """
        for obs in self.obstacles:
            if not utils.check_obs_collision(positions, obs, 2*params.R_BOT):
                return False
        return True


    def traj_opt(self, t_start_plan):
        """Trajectory Optimization

        Attempt to find a collision-free plan (v_peak) which brings the agent 
        closest to its goal.

        Parameters
        ----------
        t_start_plan : float
            Time at which planning started for this iteration

        Returns
        -------
        np.array or None
            Optimal v_peak or None if failed to find one
        
        """

        # Generate potential v_peak samples
        V_peak = utils.rand_in_bounds(params.V_BOUNDS, params.N_PLAN_MAX)
        # Eliminate samples that exceed the max velocity and max delta from initial velocity
        V_peak = utils.prune_vel_samples(V_peak, self.v_0, params.V_MAX_NORM, params.DELTA_V_PEAK_MAX)

        # Calculate the endpoints for the sample v_peaks
        P_endpoints = self.LPM.compute_endpoints(self.v_0, self.a_0, V_peak) + self.p_0
        
        # Sort V_peaks by distance to goal
        dist_to_goal = np.linalg.norm(P_endpoints - self.p_goal, axis=1)
        V_sort_idxs = np.argsort(dist_to_goal)
        V_peak = V_peak[V_sort_idxs]

        # Iterate through V_peaks until we find a feasible one
        for i in range(len(V_peak)):
        
            # Get candidate trajectory positions for current v_peak
            v_peak = V_peak[i]
            k = np.vstack((self.v_0, self.a_0, v_peak))
            cand_traj = self.LPM.compute_positions(k) + self.p_0

            # check against obstacles
            check_obs = self.check_obstacle_collisions(cand_traj)

            if check_obs:
                return v_peak

            if (time.time() - t_start_plan > params.T_PLAN):
                print("Ran out of time for planning, idx = ", i)
                break

        # No v_peaks are feasible (or we ran out of time)
        return None


    def replan(self, initial_conditions):
        """Replan

        Periodically called to perform trajectory optimization.

        Parameters
        ----------
        initial_conditions : Tuple


        Returns
        -------
        
        """
        t_start_plan = time.time()

        # Update initial conditions
        self.p_0, self.v_0, self.a_0 = initial_conditions

        self.theta_0 = np.arctan2(self.v_0[1], self.v_0[0])

        # Find a new v_peak
        v_peak = self.traj_opt(t_start_plan)

        if v_peak is None:
            # Failed to find new plan
            return None
        else:
            # Generate new trajectory
            k = np.vstack((self.v_0, self.a_0, v_peak))
            P,V,A = self.LPM.compute_trajectory(k)
            P = P + self.p_0  # translate to p_0

            
            speed = np.linalg.norm(v_peak)
            #theta = utils.signed_angle_btwn_vectors(self.v_0, v_peak)
            theta_pk = np.arctan2(v_peak[1], v_peak[0])
            dtheta = (theta_pk - self.theta_0) / params.T_PK
            print("v = ", speed)
            print("dtheta = ", dtheta)
            return P,V,A, speed, dtheta
