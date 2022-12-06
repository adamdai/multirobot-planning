import numpy as np

# Timing parameters
T_REPLAN = 5.0  # [s] amount of time between replans
T_PLAN = 5.0  # [s] amount of time allotted for planning itself 
              #     (allow buffer for actual tranmission of plan)

N_DIM = 2  # workspace dimension (i.e. 2D or 3D)

R_BOT = 0.2  # [m]

# Max velocity constraints [m/s]
V_MAX = 2.0  # L1 velocity constraints
V_BOUNDS = np.tile(np.array([-V_MAX, V_MAX]), (1,N_DIM))[0]
V_MAX_NORM = 2.0  # L2 velocity constraints
DELTA_V_PEAK_MAX = 3.0  # Delta from initial velocity constraint

R_GOAL_REACHED = 0.3  # [m] stop planning when within this dist of goal

N_PLAN_MAX = 1000  # Max number of plans to evaluate


# Reachability
ERS_MAG = 0.5             # Error reachable set size
K_DIM = np.array([2, 3])  # Trajectory parameter dimensions (row idxs) in FRS
                              # TODO: generalize this
OBS_DIM = np.arange(N_DIM)  # Obstacle dimensions (row idxs) in FRS
                            # - obstacles exist in position space