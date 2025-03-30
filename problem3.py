from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
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
from solvers import solve_PROBLEM_2, solve_PROBLEM_3, solve_PROBLEM_3_feasibility 
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

def optimal_ratio(true, recovered):
    """
    Finds the optimal ratio that minimizes the squared distance between true and recovered weights.
    
    Parameters:
    - true: numpy array of true weights for a specific state
    - recovered: numpy array of recovered weights for the same state
    
    Returns:
    - ratio: optimal ratio to scale the recovered weights
    """
    def loss(ratio):
        return np.sum((true - recovered * ratio) ** 2)

    # Minimize the squared difference using scipy's minimize function
    result = minimize(loss, 1.0)  # Initial guess is 1.0
    return result.x[0]  # Optimal ratio

def plot_time_varying_weights(true_weights, rw_simple, rw_ashwood, T):
    """
    Plots the time-varying weights for the home state and water state, comparing true and recovered weights using the simple method and Ashwood et al. method.

    Parameters:
    - true_weights: numpy array of shape (T, num_maps), the true time-varying weights
    - rw_simple: numpy array of shape (T, num_maps), the recovered time-varying weights using the simple method
    - rw_ashwood: numpy array of shape (T, num_maps), the recovered time-varying weights using the Ashwood et al. method
    - T: int, number of time steps
    """
    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
        "font.serif": ["Computer Modern"],  # Default LaTeX font
    })
    plt.figure(figsize=(12, 18))
 
    # Standardize
    standardized_true_home = (true_weights[:, 0] - np.mean(true_weights[:, 0])) / np.std(true_weights[:, 0])
    standardized_simple_home = (rw_simple[:, 0] - np.mean(rw_simple[:, 0])) / np.std(rw_simple[:, 0])
    standardized_ashwood_home = (rw_ashwood[:, 0] - np.mean(rw_ashwood[:, 0])) / np.std(rw_ashwood[:, 0])
    standardized_true_water = (true_weights[:, 1] - np.mean(true_weights[:, 1])) / np.std(true_weights[:, 1])
    standardized_simple_water = (rw_simple[:, 1] - np.mean(rw_simple[:, 1])) / np.std(rw_simple[:, 1])
    standardized_ashwood_water = (rw_ashwood[:, 1] - np.mean(rw_ashwood[:, 1])) / np.std(rw_ashwood[:, 1])

    # Calculate RMS errors
    rms_error_simple_home = np.sqrt(np.mean((standardized_true_home - standardized_simple_home) ** 2))
    rms_error_ashwood_home = np.sqrt(np.mean((standardized_true_home - standardized_ashwood_home) ** 2))
    rms_error_simple_water = np.sqrt(np.mean((standardized_true_water - standardized_simple_water) ** 2))
    rms_error_ashwood_water = np.sqrt(np.mean((standardized_true_water - standardized_ashwood_water) ** 2))

    print(f"RMS Error for Home State (Simple Method): {rms_error_simple_home}")
    print(f"RMS Error for Home State (Ashwood Method): {rms_error_ashwood_home}")
    print(f"RMS Error for Water State (Simple Method): {rms_error_simple_water}")
    print(f"RMS Error for Water State (Ashwood Method): {rms_error_ashwood_water}")

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(range(T), standardized_true_home, label=r'\textbf{True Reward at Home State}',  linewidth=4)
    plt.plot(range(T), standardized_simple_home, label=r'\textbf{Recovered Reward at Home State (P3)}', linewidth=3)
    plt.plot(range(T), standardized_ashwood_home, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022) at Home State}', linewidth=3)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Home State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20, loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(range(T), standardized_true_water, label=r'\textbf{True Reward at Water State}',  linewidth=4)
    plt.plot(range(T), standardized_simple_water, label=r'\textbf{Recovered Reward at Water State (P3)}',  linewidth=3)
    plt.plot(range(T), standardized_ashwood_water, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022) at Water State}',  linewidth=3)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Water State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20, loc='upper left')
    plt.grid(True, linestyle='-', alpha=0.7)

    plt.tight_layout()

    # Create directories if they don't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/problem3"):
        os.makedirs("results/problem3")

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the figure with a timestamp
    plt.savefig(f"results/problem3/exp3.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    
    np.random.seed(int(time.time()))

    grid_size = 5
    wind = 0.1
    # discount = 0.9
    horizon = 49

    start_state = 10


    HOME_STATE = 0
    WATER_STATE = 14
    N_experts = 200
    gridworld_H = 5
    gridworld_W = 5
    T = horizon
    GAMMA = 0.9
   

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
    # Save the policy to pi_expert as a .npy file
    pi_expert = pi
    np.save('pi_expert.npy', pi_expert)

    # Solve the problem with RTH method
    r_recovered, nu_recovered = solve_PROBLEM_3(gw, U, sigmas, pi)
    r_recovered_feasibility, nu_recovered_feasibility = solve_PROBLEM_3_feasibility(gw, U, sigmas, pi)
    np.save('data/r_recovered_feasibility.npy', r_recovered_feasibility)

   
  

    
    rec_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/output_data_alpha_times_2_comp.pickle", 'rb'))['final_rec_weights']
 
    # print(rec_weights.shape)
    rw_ashwood = rec_weights.T


    def row_space_basis(matrix, top_k=2, tol=1e-10):
        """
        Compute a basis for the row space of `matrix`, keeping only the `top_k` largest singular values.

        Args:
            matrix (numpy.ndarray): The input matrix.
            top_k (int): Number of singular vectors to retain.
            tol (float): Tolerance for rank determination.

        Returns:
            numpy.ndarray: Basis for the row space.
        """
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
        rank = min(top_k, np.sum(S > tol))  # Keep at most `top_k` nonzero singular values
        return Vt[:rank, :]  # The first `rank` rows of Vt form the row space basis

    def get_coordinates_wrt_row_basis(matrix, basis):
        """
        Compute the coordinate representation of each row of `matrix` with respect to `basis`.

        Args:
            matrix (numpy.ndarray): The original matrix.
            basis (numpy.ndarray): The row space basis.

        Returns:
            numpy.ndarray: The coordinate representation of each row.
        """
        return np.linalg.lstsq(basis.T, matrix.T, rcond=None)[0].T  # Solve for row coordinates

    # Compute row space basis for both r_recovered_RTH and r_recovered_simple
    # basis_RTH = row_space_basis(r_recovered_RTH)
    basis_simple = row_space_basis(r_recovered, top_k=n_features)
    basis_simple_feasibility = row_space_basis(r_recovered_feasibility, top_k=n_features)
    def change_of_basis_matrix(B, B_prime):
        """
        Compute the change of basis matrix P such that B P ≈ B'.
        """
        P = np.linalg.pinv(B) @ B_prime
        return P

  
    P_simple = change_of_basis_matrix(basis_simple, U.T)
    P_simple_feasibility = change_of_basis_matrix(basis_simple_feasibility, U.T)
    # Compute coordinates with respect to the new basis for both RTH and simple
    
    rw_simple = get_coordinates_wrt_row_basis(r_recovered, np.dot(basis_simple, P_simple))
    rw_feasibility = get_coordinates_wrt_row_basis(r_recovered_feasibility, np.dot(basis_simple_feasibility, P_simple_feasibility))
   
    plot_time_varying_weights(time_varying_weights,  rw_simple, rw_ashwood, T)
    

    # projected_features = np.dot(basis_simple, P_simple)

    # # print(projected_features)
    # first_row = projected_features[0, :]
    # second_row = projected_features[1, :]
    # third_row = projected_features[2, :]
    # n_states = gw.n_states  # Assuming n_states is defined in the context

    # vectors_first_row = [first_row[i * n_states:(i + 1) * n_states] for i in range(5)]
    # vectors_second_row = [second_row[i * n_states:(i + 1) * n_states] for i in range(5)]
    # vectors_third_row = [third_row[i * n_states:(i + 1) * n_states] for i in range(5)]
    
    # # Plot the heatmaps
    # fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    # for i, ax in enumerate(axes[0]):
    #     sns.heatmap(vectors_first_row[i].reshape(5, 5), ax=ax, cbar=True, annot=True, fmt=".2f")
    #     ax.set_title(f'Action {i}')
    #     ax.set_yticks(range(5))
    #     ax.set_yticklabels(range(5))
    #     ax.set_xticks([])

    # for i, ax in enumerate(axes[1]):
    #     sns.heatmap(vectors_second_row[i].reshape(5, 5).T, ax=ax, cbar=True, annot=True, fmt=".2f")
    #     ax.set_title(f'Action {i}')
    #     ax.set_yticks(range(5))
    #     ax.set_yticklabels(range(5))
    #     ax.set_xticks([])

    # for i, ax in enumerate(axes[2]):
    #     sns.heatmap(vectors_third_row[i].reshape(5, 5), ax=ax, cbar=True, annot=True, fmt=".2f")
    #     ax.set_title(f'Action {i}')
    #     ax.set_yticks(range(5))
    #     ax.set_yticklabels(range(5))
    #     ax.set_xticks([])

    # plt.suptitle('Recovered Features')
    # plt.tight_layout()

    # plt.savefig('results/problem3/features.png')




    # final_rec_goal_maps = pickle.load(open(GEN_DIR_NAME + 
    #                                 "/output_data.pickle", 'rb'))['final_rec_goal_maps']

    # feature_maps_0 = final_rec_goal_maps[0]
    # feature_maps_1 = final_rec_goal_maps[1]

    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # # Define grid dimensions
    # grid_H, grid_W = 5, 5

    # # Plot the heatmap on the left
    # sns.heatmap(vectors_first_row[0].reshape(grid_H, grid_W), ax=axes[0], cbar=True, annot=True, fmt=".2f", annot_kws={"size": 16})
    # axes[0].set_yticks(range(grid_H))
    # axes[0].set_yticklabels(range(grid_H), fontsize=16)
    # axes[0].set_xticks([])

    # # Plot the heatmap on the right using sns
    # sns.heatmap(np.reshape(final_rec_goal_maps[0], (grid_H, grid_W), order='F'), ax=axes[1], cbar=True, annot=True, fmt=".2f", annot_kws={"size": 16})
    # axes[1].set_yticks(range(grid_H))
    # axes[1].set_yticklabels(range(grid_H), fontsize=16)
    # axes[1].set_xticks([])

    # plt.tight_layout()

    # plt.savefig('results/problem3/comparison_home.png')

    # fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # # Plot the heatmap on the left
    # sns.heatmap(vectors_second_row[0].reshape(grid_H, grid_W).T, ax=axes[0], cbar=True, annot=True, fmt=".2f", annot_kws={"size": 16})
    # axes[0].set_yticks(range(grid_H))
    # axes[0].set_yticklabels(range(grid_H), fontsize=16)
    # axes[0].set_xticks([])

    # # Plot the heatmap on the right using sns
    # sns.heatmap(np.reshape(final_rec_goal_maps[1], (grid_H, grid_W), order='F'), ax=axes[1], cbar=True, annot=True, fmt=".2f", annot_kws={"size": 16})
    # axes[1].set_yticks(range(grid_H))
    # axes[1].set_yticklabels(range(grid_H), fontsize=16)
    # axes[1].set_xticks([])

    # plt.tight_layout()

    # plt.savefig('results/problem3/comparison_water.png')
