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

from main import run_methods, plot_results
from solvers import solve_PROBLEM_2, solve_PROBLEM_3, solve_PROBLEM_3_RNNM, solve_PROBLEM_3_RTH
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

def plot_time_varying_weights(true_weights, rw_simple, T):
    """
    Plots the time-varying weights for the home state and water state, comparing true and recovered weights using the simple method.

    Parameters:
    - true_weights: numpy array of shape (T, num_maps), the true time-varying weights
    - rw_simple: numpy array of shape (T, num_maps), the recovered time-varying weights using the simple method
    - T: int, number of time steps
    """
    # Enable LaTeX rendering
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
        "font.serif": ["Computer Modern"],  # Default LaTeX font
    })
    plt.figure(figsize=(12, 12))
 
    # Standardize
    standardized_true_home = (true_weights[:, 0] - np.mean(true_weights[:, 0])) / np.std(true_weights[:, 0])
    standardized_simple_home = (rw_simple[:, 0] - np.mean(rw_simple[:, 0])) / np.std(rw_simple[:, 0])
    standardized_true_water = (true_weights[:, 1] - np.mean(true_weights[:, 1])) / np.std(true_weights[:, 1])
    standardized_simple_water = (rw_simple[:, 1] - np.mean(rw_simple[:, 1])) / np.std(rw_simple[:, 1])

    # Plotting
    plt.subplot(2, 1, 1)
    plt.plot(range(T), standardized_true_home, label=r'\textbf{True Reward at Home State (Standardized)}', linestyle='--', linewidth=2)
    plt.plot(range(T), standardized_simple_home, label=r'\textbf{Recovered Reward at Home State (Standardized)}', linestyle='-.', linewidth=2)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Home State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(range(T), standardized_true_water, label=r'\textbf{True Reward at Water State (Standardized)}', linestyle='--', linewidth=2)
    plt.plot(range(T), standardized_simple_water, label=r'\textbf{Recovered Reward at Water State (Standardized)}', linestyle='-.', linewidth=2)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Water State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

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
    horizon = 50

    start_state = 10


    HOME_STATE = 0
    WATER_STATE = 14
    N_experts = 200
    gridworld_H = 5
    gridworld_W = 5
    T = 50
    GAMMA = 0.9
   

    # select number of maps
    n_features = 2
    
    # choose noise covariance for the random walk priors over weights corresponding to these maps
    sigmas = [2**-(3.5)]*n_features

    print(sigmas)
    weights0 = np.zeros(n_features) #initial weights at t=0
    weights0[1] = 1

    # goal_maps = create_goal_maps(gridworld_H*gridworld_W, WATER_STATE, HOME_STATE)
    #size is num_maps x num_states
    #generate time-varying weights
    time_varying_weights = generate_weight_trajectories(sigmas,
                                                        weights0,
                                                        T) #T x num_maps
    # plot_time_varying_weights(time_varying_weights, horizon)
    gw = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)


    # construct U
    U = np.zeros(shape=(gw.n_states*gw.n_actions, n_features))

    U[HOME_STATE, 0] = 1.0
    U[HOME_STATE + gw.n_states, 0] = 1.0
    U[HOME_STATE + 2*gw.n_states, 0] = 1.0
    U[HOME_STATE + 3*gw.n_states, 0] = 1.0
    U[HOME_STATE + 4*gw.n_states, 0] = 1.0

    U[WATER_STATE, 1] = 1.0
    U[WATER_STATE + gw.n_states, 1] = 1.0
    U[WATER_STATE + 2*gw.n_states, 1] = 1.0
    U[WATER_STATE + 3*gw.n_states, 1] = 1.0
    U[WATER_STATE + 4*gw.n_states, 1] = 1.0

    true_reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))

    for t in range(gw.horizon):
        for s in range(gw.n_states):
             for a in range(gw.n_actions):
                    true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                        + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]
    
    true_reward_matrix = np.zeros((gw.horizon, gw.n_states * gw.n_actions))

    for t in range(gw.horizon):
        for s in range(gw.n_states):
            for a in range(gw.n_actions):
                idx = s + a * gw.n_states
                true_reward_matrix[t, idx] = (
                    U[idx, 0] * time_varying_weights[t, 0] +
                    U[idx, 1] * time_varying_weights[t, 1]
                )
    # for t in range(gw.horizon):
    #     print(f"At time step {t}:")
    #     print(f"true reward: {true_reward[t, :, :]}")
    
    # print(true_reward)
    V, Q, pi = soft_bellman_operation(gw, true_reward)
    # print(np.round(pi[T-1,:,:],4))
    
    # r_recovered_RTH, nu_recovered_RTH  = solve_PROBLEM_3_RTH(gw, U, sigmas, pi,max_iter=5)
    r_recovered_simple, nu_recovered_simple  = solve_PROBLEM_3(gw, U, sigmas, pi)
    

    # r_recovered_reshaped =  np.zeros((gw.horizon, gw.n_states , gw.n_actions))
    # for t in range(gw.horizon-1):
    #     for s in range(gw.n_states):
    #         for a in range(gw.n_actions):
    #             idx = s + a * gw.n_states
    #             r_recovered_reshaped[t+1, s, a] = r_recovered_simple[t,idx]
    
    # # Find the policy for r_recovered
    # V_recovered, Q_recovered, pi_recovered = soft_bellman_operation(gw, r_recovered_reshaped)

    # # Calculate the norm difference between the true policy and the recovered policy
    # for t in range(gw.horizon):
    #     norm_difference = np.linalg.norm(pi[t] - pi_recovered[t])
    #     print(f"Norm difference between the true policy and the recovered policy at time step {t}: {norm_difference}")
        # print(f"Norm difference between the true policy and the recovered policy at time step {t}: {norm_difference}")

    # Compute singular values for both r_recovered_RTH and r_recovered_simple
    # singular_values_RTH = np.linalg.svd(r_recovered_RTH, compute_uv=False)
    # rounded_singular_values_RTH = np.round(singular_values_RTH, 4)

 

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
    basis_simple = row_space_basis(r_recovered_simple)

    def change_of_basis_matrix(B, B_prime):
        """
        Compute the change of basis matrix P such that B P ≈ B'.
        """
        P = np.linalg.pinv(B) @ B_prime
        return P

    # Compute change of basis matrices for both RTH and simple
    # P_RTH = change_of_basis_matrix(basis_RTH, U.T)
    P_simple = change_of_basis_matrix(basis_simple, U.T)

    # Compute coordinates with respect to the new basis for both RTH and simple
    # coordinates_RTH = get_coordinates_wrt_row_basis(r_recovered_RTH, np.dot(basis_RTH, P_RTH))
    coordinates_simple = get_coordinates_wrt_row_basis(r_recovered_simple, np.dot(basis_simple, P_simple))
    # coordinates_RTH = get_coordinates_wrt_row_basis(r_recovered_simple, basis_simple)
   
    plot_time_varying_weights(time_varying_weights,  coordinates_simple, T)
    # plot_time_varying_weights(time_varying_weights, coordinates_simple, T)



    projected_features = np.dot(basis_simple, P_simple)

    # print(projected_features)
    first_row = projected_features[0, :]
    second_row = projected_features[1, :]
    n_states = gw.n_states  # Assuming n_states is defined in the context

    vectors_first_row = [first_row[i * n_states:(i + 1) * n_states] for i in range(5)]
    vectors_second_row = [second_row[i * n_states:(i + 1) * n_states] for i in range(5)]

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, ax in enumerate(axes[0]):
        sns.heatmap(vectors_first_row[i].reshape(5, 5), ax=ax, cbar=True, annot=True, fmt=".2f")
        ax.set_title(f'Action {i}')
        ax.set_yticks(range(5))
        ax.set_yticklabels(range(5))
        ax.set_xticks([])

    for i, ax in enumerate(axes[1]):
        sns.heatmap(vectors_second_row[i].reshape(5, 5).T, ax=ax, cbar=True, annot=True, fmt=".2f")
        ax.set_title(f'Action {i}')
        ax.set_yticks(range(5))
        ax.set_yticklabels(range(5))
        ax.set_xticks([])

    plt.suptitle('Recovered Features')
    plt.tight_layout()

    plt.savefig('results/problem3/features.png')
