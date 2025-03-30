from dynamics import BasicGridWorld, BlockedGridWorld
from utils.bellman import soft_bellman_operation
from noisy_solvers import solve_PROBLEM_3_noisy
from solvers import solve_PROBLEM_3_feasibility
# from solvers import solve_milp
from utils.bellman import state_only_soft_bellman_operation, soft_bellman_operation, time_varying_value_iteration
import numpy as np
from utils.checks import is_markovian
import pickle
import sys
import time
import datetime
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from utils.sample import estimate_pi_and_visits_numba, compute_likelihood

path_to_dynamic_irl = '~\Desktop'
repo2_path = os.path.expanduser(path_to_dynamic_irl)  # Adjust path if necessary
sys.path.append(repo2_path)

repo2_path = os.path.expanduser("~/Desktop/dynamic_irl")  # Adjust the path to your setup
if repo2_path not in sys.path:
    sys.path.append(repo2_path)

repo2_path = os.path.expanduser("~/Desktop/dynamic_irl/src")  # Adjust the path to your setup
if repo2_path not in sys.path:
    sys.path.append(repo2_path)

repo2_path = os.path.expanduser("~/Desktop/dynamic_irl/src/optimize_weights")  # Adjust the path to your setup
if repo2_path not in sys.path:
    sys.path.append(repo2_path)


# import dynamic_irl

# from dynamic_irl.src.envs  import  gridworld

# from dynamic_irl.src.simulate_data_gridworld import generate_expert_trajectories
# from dynamic_irl.src.simulate_data_gridworld import create_goal_maps
# from dynamic_irl.src.dirl_for_gridworld import fit_dirl_gridworld
GEN_DIR_NAME = 'data'

from main import run_methods, plot_results
from solvers import solve_PROBLEM_2, solve_PROBLEM_3, solve_PROBLEM_3_RNNM, solve_PROBLEM_3_RTH, solve_PROBLEM_3_regularized
from scipy.optimize import minimize

def generate_weight_trajectories(sigmas, weights0, T):
    '''Simulates time varying weights, for a given sigmas array

    Args:
        sigma: array of length K, the smoothness of each reward weight
        weights0: values of time-varying weights parameters at t=0
        T: length of trajectory to generate (i.e. number of states visited in the gridworld)

    Returns:
        rewards: array of KxT reward parameters
    '''
    K = len(sigmas)
    noise = np.random.normal(scale=sigmas, size=(T, K))
    # home port
    # np.random.seed(50)
    # noise[:,0] = np.random.normal(0.01, scale=sigmas[0], size=(T,))
    # # water port
    # # np.random.seed(100)
    # noise[:,1] = np.random.normal(-0.02, scale=sigmas[1], size=(T,))
    # noise[0,:] = weights0
    weights = np.cumsum(noise, axis=0)
    return weights #array of size (TxK)

# def optimal_ratio(true, recovered):
#     """
#     Finds the optimal ratio that minimizes the squared distance between true and recovered weights.
    
#     Parameters:
#     - true: numpy array of true weights for a specific state
#     - recovered: numpy array of recovered weights for the same state
    
#     Returns:
#     - ratio: optimal ratio to scale the recovered weights
#     """
#     def loss(ratio):
#         return np.sum((true - recovered * ratio) ** 2)

#     # Minimize the squared difference using scipy's minimize function
#     result = minimize(loss, 1.0)  # Initial guess is 1.0
#     return result.x[0]  # Optimal ratio

# def plot_time_varying_weights(true_weights, results, rw_ashwood, T):
#     """
#     Plots the time-varying weights for the home state and water state, comparing true and recovered weights using the simple method and Ashwood et al. method.

#     Parameters:
#     - true_weights: numpy array of shape (T, num_maps), the true time-varying weights
#     - rw_simple: numpy array of shape (T, num_maps), the recovered time-varying weights using the simple method
#     - rw_ashwood: numpy array of shape (T, num_maps), the recovered time-varying weights using the Ashwood et al. method
#     - T: int, number of time steps
#     """
#     # Enable LaTeX rendering
#     plt.rcParams.update({
#         "text.usetex": True,  # Use LaTeX for text rendering
#         "font.family": "serif",  # Use a serif font
#         "font.serif": ["Computer Modern"],  # Default LaTeX font
#     })
#     plt.figure(figsize=(12, 18))
 
#     standardized_results = []
#     # Standardize
#     standardized_true_home = (true_weights[:, 0] - np.mean(true_weights[:, 0])) / np.std(true_weights[:, 0])
#     standardized_ashwood_home = (rw_ashwood[:, 0] - np.mean(rw_ashwood[:, 0])) / np.std(rw_ashwood[:, 0])
#     standardized_true_water = (true_weights[:, 1] - np.mean(true_weights[:, 1])) / np.std(true_weights[:, 1])
#     standardized_ashwood_water = (rw_ashwood[:, 1] - np.mean(rw_ashwood[:, 1])) / np.std(rw_ashwood[:, 1])

#     for (label, rw_simple) in results:
#         standardized_simple_home = (rw_simple[:, 0] - np.mean(rw_simple[:, 0])) / np.std(rw_simple[:, 0])
#         standardized_simple_water = (rw_simple[:, 1] - np.mean(rw_simple[:, 1])) / np.std(rw_simple[:, 1])
#         standardized_results.append((label, standardized_simple_home,  standardized_simple_water))

#     # Calculate RMS errors
#     # rms_error_simple_home = np.sqrt(np.mean((standardized_true_home - standardized_simple_home) ** 2))
#     rms_error_ashwood_home = np.sqrt(np.mean((standardized_true_home - standardized_ashwood_home) ** 2))
#     # rms_error_simple_water = np.sqrt(np.mean((standardized_true_water - standardized_simple_water) ** 2))
#     rms_error_ashwood_water = np.sqrt(np.mean((standardized_true_water - standardized_ashwood_water) ** 2))

#     for (label, standardized_simple_home, _) in standardized_results:
#         rms_error_simple_home = np.sqrt(np.mean((standardized_true_home - standardized_simple_home) ** 2))
#         print(f"RMS Error for Home State ({label}): {rms_error_simple_home}")

#     print(f"RMS Error for Home State (Ashwood Method): {rms_error_ashwood_home}")
#     for (label, _, standardized_simple_water) in standardized_results:
#         rms_error_simple_water = np.sqrt(np.mean((standardized_true_water - standardized_simple_water) ** 2))
#         print(f"RMS Error for Water State ({label}): {rms_error_simple_water}")
#     print(f"RMS Error for Water State (Ashwood Method): {rms_error_ashwood_water}")

#     # Plotting
#     plt.subplot(2, 1, 1)
#     plt.plot(range(T), standardized_true_home, label=r'\textbf{True Reward}', linestyle='-', linewidth=4)
#     for (label, standardized_simple_home,  _) in standardized_results:
#         plt.plot(range(T), standardized_simple_home, label=f"Recovered Reward - {label}", linestyle=':', linewidth=4)
    
#     plt.plot(range(T), standardized_ashwood_home, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022)}', linestyle='-', linewidth=4)
#     plt.xlabel(r'\textbf{Time}', fontsize=24)
#     plt.ylabel(r'\textbf{Weight}', fontsize=24)
#     plt.title(r'\textbf{Time-Varying Weight for Home State (Standardized)}', fontsize=28)
#     plt.legend(fontsize=20)
#     plt.grid(True, linestyle='-', alpha=0.7)

#     plt.subplot(2, 1, 2)
#     plt.plot(range(T), standardized_true_water, label=r'\textbf{True Reward}', linestyle='-', linewidth=4)
#     for (label, _,  standardized_simple_water) in standardized_results:
#         plt.plot(range(T), standardized_simple_water, label=f"Recovered Reward - {label}", linestyle=':', linewidth=4)

#     plt.plot(range(T), standardized_ashwood_water, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022)}', linestyle='-', linewidth=4)
#     plt.xlabel(r'\textbf{Time}', fontsize=24)
#     plt.ylabel(r'\textbf{Weight}', fontsize=24)
#     plt.title(r'\textbf{Time-Varying Weight for Water State (Standardized)}', fontsize=28)
#     plt.legend(fontsize=20)
#     plt.grid(True, linestyle='-', alpha=0.7)

#     plt.tight_layout()

#     # Create directories if they don't exist
#     if not os.path.exists("results"):
#         os.makedirs("results")
#     if not os.path.exists("results/problem3"):
#         os.makedirs("results/problem3")

#     # Generate a timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     # Save the figure with a timestamp
#     plt.savefig(f"results/problem3/exp3.png", dpi=300, bbox_inches='tight')

# def row_space_basis(matrix, top_k=2, tol=1e-10):
#     """
#     Compute a basis for the row space of `matrix`, keeping only the `top_k` largest singular values.

#     Args:
#         matrix (numpy.ndarray): The input matrix.
#         top_k (int): Number of singular vectors to retain.
#         tol (float): Tolerance for rank determination.

#     Returns:
#         numpy.ndarray: Basis for the row space.
#     """
#     U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
#     rank = min(top_k, np.sum(S > tol))  # Keep at most `top_k` nonzero singular values
#     return Vt[:rank, :]  # The first `rank` rows of Vt form the row space basis

# def get_coordinates_wrt_row_basis(matrix, basis):
#     """
#     Compute the coordinate representation of each row of `matrix` with respect to `basis`.

#     Args:
#         matrix (numpy.ndarray): The original matrix.
#         basis (numpy.ndarray): The row space basis.

#     Returns:
#         numpy.ndarray: The coordinate representation of each row.
#     """
#     return np.linalg.lstsq(basis.T, matrix.T, rcond=None)[0].T  # Solve for row coordinates

# def change_of_basis_matrix(B, B_prime):
#     """
#     Compute the change of basis matrix P such that B P ≈ B'.
#     """
#     P = np.linalg.pinv(B) @ B_prime
#     return P

if __name__ == "__main__":
    
    # np.random.seed(int(time.time()))
    # np.random.seed(0)


    grid_size = 5
    wind = 0.1
    # discount = 0.9
    horizon = 49

    start_state = 10


    HOME_STATE = 0
    WATER_STATE = 14
    # N_experts = 200
    gridworld_H = 5
    gridworld_W = 5
    T = horizon
    GAMMA = 0.9
   
    num_trajectories = 50_000_000


    # select number of maps
    n_features = 2
    
    # choose noise covariance for the random walk priors over weights corresponding to these maps
    sigmas = [2**-(3.5)]*n_features

    # print(sigmas)
    weights0 = np.zeros(n_features) #initial weights at t=0
    weights0[1] = 1


    time_varying_weights  = pickle.load(open(GEN_DIR_NAME + 
                                    "/output_data.pickle", 'rb'))['gen_time_varying_weights']   
    
    time_varying_weights = time_varying_weights.T

    print("time_varying_weights", time_varying_weights.shape)

    gw = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)

    # print("gw.horizon", gw.horizon)
    # construct U
    U = np.zeros(shape=(gw.n_states*gw.n_actions, n_features))

    REWARD_SCALE = 2.
    U[HOME_STATE, 0] = 1.0* REWARD_SCALE
    U[HOME_STATE + gw.n_states, 0] = 1.0* REWARD_SCALE
    U[HOME_STATE + 2*gw.n_states, 0] = 1.0* REWARD_SCALE
    U[HOME_STATE + 3*gw.n_states, 0] = 1.0* REWARD_SCALE
    U[HOME_STATE + 4*gw.n_states, 0] = 1.0* REWARD_SCALE

    U[WATER_STATE, 1] = 1.0* REWARD_SCALE
    U[WATER_STATE + gw.n_states, 1] = 1.0* REWARD_SCALE
    U[WATER_STATE + 2*gw.n_states, 1] = 1.0* REWARD_SCALE
    U[WATER_STATE + 3*gw.n_states, 1] = 1.0* REWARD_SCALE
    U[WATER_STATE + 4*gw.n_states, 1] = 1.0* REWARD_SCALE

    # Desired action
    # desired_action = 4 # stay
    # U[HOME_STATE + desired_action*gw.n_states, 2] = 1.0
    # U[WATER_STATE + desired_action*gw.n_states, 2] = 1.0

    # Append a third column to time_varying_weights
    # third_column = np.zeros((time_varying_weights.shape[0], 1))
    # third_column[-1, 0] = 1
    # time_varying_weights = np.hstack((time_varying_weights, third_column))
   

    # Generate time-varying weights for all map

    true_reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    # print("true_reward", true_reward.shape, "time_varying_weights", time_varying_weights.shape)
    for t in range(gw.horizon):
        for s in range(gw.n_states):
             for a in range(gw.n_actions):
                    true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                        + U[s + a * gw.n_states, 1] * time_varying_weights[t,1] 
                        # + U[s + a * gw.n_states, 2] * time_varying_weights[t,2]
    
    true_reward_matrix = np.zeros((gw.horizon, gw.n_states * gw.n_actions))

    for t in range(gw.horizon):
        for s in range(gw.n_states):
            for a in range(gw.n_actions):
                idx = s + a * gw.n_states
                true_reward_matrix[t, idx] = (
                    U[idx, 0] * time_varying_weights[t, 0] +
                    U[idx, 1] * time_varying_weights[t, 1]
                )
    # # for t in range(gw.horizon):
    # #     print(f"At time step {t}:")
    # #     print(f"true reward: {true_reward[t, :, :]}")
    
    # print(true_reward)
    V, Q, pi = soft_bellman_operation(gw, true_reward)

    delta = 1-1e-4


    P = np.asarray(gw.P, dtype=np.float64)     # shape (A, S, S)
    pi = np.asarray(pi, dtype=np.float64)   # shape (H, S, A)
    action_counts, visit_counts = estimate_pi_and_visits_numba(P, pi, gw.horizon, num_trajectories)

    if (visit_counts == 0).any():
        print("Still there are t,s pairs not visited")

    # Normalize to get pi_hat
    pi_hat = np.zeros_like(pi)/gw.n_actions
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_hat = np.divide(action_counts, visit_counts[:, :, None], where=visit_counts[:, :, None] != 0)

    log_term = np.log(2 / (1 - delta))

    b = np.full((gw.horizon, gw.n_states, gw.n_actions), 1e3)  # default to inf for unvisited

    # Where visits > 0, compute epsilon and b
    visited_mask = visit_counts > 0
    epsilons = np.zeros_like(visit_counts, dtype=np.float64)
    epsilons[visited_mask] = np.sqrt(1 / (2 * visit_counts[visited_mask]) * log_term)

    # ### Find a single alpha
    # sum_array = pi_hat - epsilons[:, :, np.newaxis]
    # # Find the minimal value
    # alpha = min(np.min(sum_array), pi_hat.min())

    # print(f"{alpha=}, {pi.min()=}, {pi_hat.min()=}")

    # b[visited_mask] = epsilons[visited_mask] / alpha


    ### Find a alpha(t,s,a)
    # Compute alpha[t, s, a] = pi_hat[t, s, a] - epsilons[t, s]
    epsilon_broadcast = np.broadcast_to(epsilons[:, :, np.newaxis], pi_hat.shape)  # shape (H, S, A)
    alpha = pi_hat - epsilon_broadcast

    # Assert all alpha values are positive where visited
    visited_mask_3d = np.broadcast_to(visited_mask[:, :, np.newaxis], alpha.shape)
    print(f"{alpha.min()=}")
    assert np.all(alpha[visited_mask_3d] > 0), "Alpha contains non-positive values!"

    # Avoid division by zero or negative alpha values
    safe_mask = visited_mask_3d & (alpha > 0)

    # b = np.full_like(pi_hat, 1e3)
    b[safe_mask] = epsilon_broadcast[safe_mask] / alpha[safe_mask]
    print(b.shape)

    # Check b for irregularities and print them
    irregular_mask = (b < 0) | ~np.isfinite(b) | np.isnan(b)
    irregular_indices = np.argwhere(irregular_mask)
    for t, s, a in irregular_indices:
        print(f"Irregular b at (t={t}, s={s}, a={a}): b={b[t, s, a]}, epsilon={epsilons[t, s]}, alpha={alpha[t, s, a]}")

    # Compute fraction of (t, s, a) entries where deviation exceeds epsilon
    violation_fraction = np.sum(np.abs(pi - pi_hat) > epsilon_broadcast) / (gw.horizon * gw.n_states * gw.n_actions)

    print("Violation fraction: ", violation_fraction)
    print(f"{epsilons.max()=}, {epsilons.min()=}")

    # Solve the problem with RTH method
    r_recovered, nu_recovered = solve_PROBLEM_3_noisy(gw, U, sigmas, pi_hat, b)
    r_recovered = r_recovered.reshape(horizon, gw.n_states, gw.n_actions, order='F')



    r_recovered_feasibility, nu_recovered_feasibility = solve_PROBLEM_3_feasibility(gw, U, sigmas, pi)
    r_recovered_feasibility = r_recovered_feasibility.reshape(horizon, gw.n_states, gw.n_actions, order='F')
    rec_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/output_data.pickle", 'rb'))['rec_rewards']
    
    rec_weights_MaxEntIRL = np.load('data/static_MaxEntIRL_reward.npy')
    


    # print(rec_weights.shape)
    r_recovered_ashwood = np.zeros((gw.horizon, gw.n_states, gw.n_actions))
    r_recovered_MaxEntIRL = np.zeros((gw.horizon, gw.n_states, gw.n_actions))
    # for t in range(gw.horizon):
    #     for s in range(gw.n_states):
    #         for a in range(gw.n_actions):
    #             r_recovered_ashwood[t, s ,a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
    #                     + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]

    for t in range(gw.horizon):
            for s in range(gw.n_states):
                for a in range(gw.n_actions):
                    r_recovered_ashwood[t, s ,a] = rec_weights[t, s]
                    r_recovered_MaxEntIRL[t, s ,a] = rec_weights_MaxEntIRL[t, s]


    gw2 = BlockedGridWorld(grid_size, wind, GAMMA, horizon, None)
    _, _, pi2 = soft_bellman_operation(gw2, true_reward)
    _, _, pi_recovered_2  = soft_bellman_operation(gw2, r_recovered)
    _, _, pi_recovered_ashwood = soft_bellman_operation(gw2, r_recovered_ashwood)
    _, _, pi_recovered_MaxEntIRL = soft_bellman_operation(gw2, r_recovered_MaxEntIRL)
    _, _, pi_recovered_feasibility = soft_bellman_operation(gw2, r_recovered_feasibility)
    # _, _, pi_hat_hat = soft_bellman_operation(gw, r_recovered)

    P = np.asarray(gw2.P, dtype=np.float64)     # shape (A, S, S)
    pi2 = np.asarray(pi2, dtype=np.float64)   # shape (H, S, A)
    action_counts_val, visit_counts_val = estimate_pi_and_visits_numba(P, pi2, gw2.horizon, 10)

    pi_hat_hat_likelihood = compute_likelihood(pi_recovered_2, visit_counts_val, action_counts_val)
    pi_hat_likelihood = compute_likelihood(pi_hat, visit_counts_val, action_counts_val)
    pi_likelihood = compute_likelihood(pi, visit_counts_val, action_counts_val)
    pi2_likelihood = compute_likelihood(pi2, visit_counts_val, action_counts_val)
    pi_recovered_ashwood_likelihood = compute_likelihood(pi_recovered_ashwood, visit_counts_val, action_counts_val)
    pi_recovered_MaxEntIRL_likelihood = compute_likelihood(pi_recovered_MaxEntIRL, visit_counts_val, action_counts_val)
    pi_recovered_feasibility_likelihood = compute_likelihood(pi_recovered_feasibility, visit_counts_val, action_counts_val)
    uniform_likelihood = compute_likelihood(np.ones_like(pi2)/gw2.n_actions, visit_counts_val, action_counts_val)

    print("Log likelihoods")
    print("Pi_1: ", pi_likelihood)
    print("Pi_2: ", pi2_likelihood)
    print("Pi Hat: ", pi_hat_likelihood)
    print("Pi Double Hat: ", pi_hat_hat_likelihood)
    print("Pi Recovered Ashwood: ", pi_recovered_ashwood_likelihood)
    print("Pi Recovered MaxEntIRL: ", pi_recovered_MaxEntIRL_likelihood)
    print("Pi Recovered Feasibility: ", pi_recovered_feasibility_likelihood)
    print("Uniform Policy: ", uniform_likelihood)

    

    # print(rec_weights.shape)
    # print(rec_weights.shape)
    # rw_ashwood = rec_weights.T
    
    # plot_time_varying_weights(time_varying_weights,  results, rw_ashwood, T)
    