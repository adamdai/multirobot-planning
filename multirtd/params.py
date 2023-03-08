import numpy as np

# Timing parameters
DT = 0.1  # [s] timestep for trajectory discretization
T_REPLAN = 5.0  # [s] amount of time between replans
T_PLAN = 5.0  # [s] amount of time allotted for planning itself 
              #     (allow buffer for actual tranmission of plan)
T_PK = 1.5  # [s] time along trajectory of peak velocity
TRAJ_TIME_LEN = 3.0  # [s] Trajectory total duration
TRAJ_IDX_LEN = int(TRAJ_TIME_LEN / DT) + 1  # Trajectory length in timestep

N_DIM = 2  # workspace dimension (i.e. 2D or 3D)

R_BOT = 0.2  # [m]

# Max velocity constraints [m/s]
V_MAX = 0.22  # L1 velocity constraints
V_BOUNDS = np.tile(np.array([-V_MAX, V_MAX]), (1,N_DIM))[0]
V_MAX_NORM = 2.0  # L2 velocity constraints
DELTA_V_PEAK_MAX = 3.0  # Delta from initial velocity constraint

W_MAX = 1.0  # [rad/s] max angular velocity (turning rate)

R_GOAL_REACHED = 0.3  # [m] stop planning when within this dist of goal

N_PLAN_MAX = 1000  # Max number of plans to evaluate


# Reachability
ERS_MAG = 0.25             # Error reachable set size
K_DIM = np.array([2, 3])  # Trajectory parameter dimensions (row idxs) in FRS
                              # TODO: generalize this
OBS_DIM = np.arange(N_DIM)  # Obstacle dimensions (row idxs) in FRS
                            # - obstacles exist in position space