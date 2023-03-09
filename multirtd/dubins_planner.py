"""SimplePlanner class

This module defines the SimplePlanner class.

"""

import numpy as np
import time 
from scipy.optimize import minimize, NonlinearConstraint

import multirtd.params as params
from multirtd.LPM import LPM
from multirtd.utils import rand_in_bounds, check_obs_collision

from multirtd.dubins_model import dubins_traj


class DubinsPlanner:
    """Dubins Planner class

    Modified linear planner for dubin's vehicle

    traj_opt_sample: sample v and w uniformly offline. at runtime shift to robot frame

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self):

        # Goal position [m]
        self.p_goal = np.zeros(params.N_DIM)

        # Obstacles
        self.obstacles = []


    def traj_opt(self, init_pose, t_start_plan):
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
        def cost(u):
            traj = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)
            dist = np.linalg.norm(traj[-1,:-1] - self.p_goal)
            return dist

        def constraint(u):
            traj = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)
            dists = []
            for obs_c, obs_r in self.obstacles:
                dist = np.linalg.norm(traj[:,:-1] - obs_c, axis=1) - (obs_r + params.R_BOT)
                dists.append(dist)
            # for peer in self.peer_traj:
            #     dist = np.linalg.norm(traj[:,:-1] - self.peer_traj[peer][:,:-1], axis=1) - (2 * params.R_BOT)
            #     dists.append(dist)
            return np.hstack(dists)

        start_time = time.time()
        cons = NonlinearConstraint(constraint, 0, np.inf)
        u0 = rand_in_bounds([-params.V_MAX, params.V_MAX, -params.W_MAX, params.W_MAX], 1)[0]
        res = minimize(cost, np.array([0, 0]), method='SLSQP', bounds=[(-params.V_MAX, params.V_MAX), (-params.W_MAX, params.W_MAX)], constraints=cons, 
                    options={'disp': False,
                             'ftol': 1e-6})
        # print("Time elapsed: {:.3f} s".format(time.time() - start_time))
        # print(res.x)
        return res.x

    
    def check_collisions(self, traj):
        """Check if the trajectory collides with any obstacles."""
        for obs_c, obs_r in self.obstacles:
            dist = np.linalg.norm(traj[:,:-1] - obs_c, axis=1)
            if np.any(dist < obs_r + params.R_BOT):
                return True
        return False


    def traj_opt_sample(self, init_pose, t_start_plan):
        """Sampling-based Trajectory Optimization

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
        start_time = time.time()
        u_samples = rand_in_bounds([0, params.V_MAX, -params.W_MAX, params.W_MAX], params.N_PLAN_MAX)
        #traj_samples = np.zeros((params.N_PLAN_MAX, params.TRAJ_IDX_LEN, 3))
        endpoints = np.zeros((params.N_PLAN_MAX, 2))
        for i, u in enumerate(u_samples):
            #traj_samples[i,:,:] = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)
            traj = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)
            endpoints[i] = traj[-1,:-1]

        #endpoints = traj_samples[:,-1,:-1]
        dists = np.linalg.norm(endpoints - self.p_goal, axis=1)
        sort_idxs = np.argsort(dists)
        u_samples_sorted = u_samples[sort_idxs]
        #traj_samples_sorted = traj_samples[sort_idxs]

        # Check collisions
        for i, u in enumerate(u_samples_sorted):
            #traj = traj_samples_sorted[i]
            traj = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)
            if self.check_collisions(traj):
                continue
            else:
                # print("found plan ", u)
                # print("Time elapsed: {:.3f} s".format(time.time() - start_time))
                return u
        print("No feasible plan found")
        return None


    def replan(self, init_pose):
        """Replan
        
        Periodically called to perform trajectory optimization.
        
        """
        t_start_plan = time.time()
        traj = None

        # Find a new plan
        u = self.traj_opt(init_pose, t_start_plan)

        if u is None:
            # Failed to find new plan
            print("Failed to find new plan")
        else:
            # Generate new trajectory
            traj = dubins_traj(init_pose, u, params.TRAJ_IDX_LEN, params.DT)

            # print("Found new trajectory, u = " + str(np.round(u, 2)))
            # print(" Start point: " + str(np.round(traj[0], 2)))
            # print(" End point: " + str(np.round(traj[-1], 2)))
        
            # # Check for goal-reached
            # if np.linalg.norm(traj[-1,:-1] - self.p_goal) < params.R_GOAL_REACHED:
            #     print("Goal reached")
            #     self.done = True

        return traj