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
    noise[0,:] = weights0
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

def plot_time_varying_weights(true_weights, recovered_weights, T):
    """
    Plots the time-varying weights for the home state and water state, comparing true and recovered weights.

    Parameters:
    - true_weights: numpy array of shape (T, num_maps), the true time-varying weights
    - recovered_weights: numpy array of shape (T, num_maps), the recovered time-varying weights
    - T: int, number of time steps
    """
    plt.figure(figsize=(18, 12))

    # Option 1: Standardize
    standardized_true_home = (true_weights[:-1, 0] - np.mean(true_weights[:-1, 0])) / np.std(true_weights[:-1, 0])
    standardized_recovered_home = (recovered_weights[:-1, 0] - np.mean(recovered_weights[:-1, 0])) / np.std(recovered_weights[:-1, 0])
    standardized_true_water = (true_weights[:-1, 1] - np.mean(true_weights[:-1, 1])) / np.std(true_weights[:-1, 1])
    standardized_recovered_water = (recovered_weights[:-1, 1] - np.mean(recovered_weights[:-1, 1])) / np.std(recovered_weights[:-1, 1])

    # Option 2: Min-Max Scaling
    min_max_true_home = (true_weights[:-1, 0] - np.min(true_weights[:-1, 0])) / (np.max(true_weights[:-1, 0]) - np.min(true_weights[:-1, 0]))
    min_max_recovered_home = (recovered_weights[:-1, 0] - np.min(recovered_weights[:-1, 0])) / (np.max(recovered_weights[:-1, 0]) - np.min(recovered_weights[:-1, 0]))
    min_max_true_water = (true_weights[:-1, 1] - np.min(true_weights[:-1, 1])) / (np.max(true_weights[:-1, 1]) - np.min(true_weights[:-1, 1]))
    min_max_recovered_water = (recovered_weights[:-1, 1] - np.min(recovered_weights[:-1, 1])) / (np.max(recovered_weights[:-1, 1]) - np.min(recovered_weights[:-1, 1]))

    # Option 3: Optimal Ratio
    ratio_home = optimal_ratio(true_weights[:-1, 0], recovered_weights[:-1, 0])
    adjusted_recovered_home = recovered_weights[:-1, 0] * ratio_home
    ratio_water = optimal_ratio(true_weights[:-1, 1], recovered_weights[:-1, 1])
    adjusted_recovered_water = recovered_weights[:-1, 1] * ratio_water

    # Plotting
    for i, (true_home, recovered_home, true_water, recovered_water, method) in enumerate([
        (standardized_true_home, standardized_recovered_home, standardized_true_water, standardized_recovered_water, "Standardized"),
        (min_max_true_home, min_max_recovered_home, min_max_true_water, min_max_recovered_water, "Min-Max Scaled"),
        (true_weights[:-1, 0], adjusted_recovered_home, true_weights[:-1, 1], adjusted_recovered_water, "Optimal Ratio")
    ]):
        plt.subplot(3, 2, 2*i + 1)
        plt.plot(range(T-1), true_home, label=f'True Reward at Home State ({method})', linestyle='--', linewidth=2)
        plt.plot(range(T-1), recovered_home, label=f'Recovered Reward at Home State ({method})', linewidth=2)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.title(f'Time-Varying Weight for Home State ({method})', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(3, 2, 2*i + 2)
        plt.plot(range(T-1), true_water, label=f'True Reward at Water State ({method})', linestyle='--', linewidth=2)
        plt.plot(range(T-1), recovered_water, label=f'Recovered Reward at Water State ({method})', linewidth=2)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.title(f'Time-Varying Weight for Water State ({method})', fontsize=14)
        plt.legend(fontsize=10)
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
    plt.savefig(f"results/problem3/time_varying_weights_{timestamp}.png", dpi=300, bbox_inches='tight')

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
    
    # r_recovered, nu_recovered  = solve_PROBLEM_3_RTH(gw, U, sigmas, pi)
    r_recovered, nu_recovered  = solve_PROBLEM_3(gw, U, sigmas, pi)

    r_recovered_reshaped =  np.zeros((gw.horizon, gw.n_states , gw.n_actions))
    for t in range(gw.horizon):
        for s in range(gw.n_states):
            for a in range(gw.n_actions):
                idx = s + a * gw.n_states
                r_recovered_reshaped[t, s, a] = r_recovered[t,idx]
    
    # Find the policy for r_recovered
    V_recovered, Q_recovered, pi_recovered = soft_bellman_operation(gw, r_recovered_reshaped)

    # Calculate the norm difference between the true policy and the recovered policy
    for t in range(gw.horizon):
        norm_difference = np.linalg.norm(pi[t] - pi_recovered[t])
        assert norm_difference < 1e-6, f"Norm difference between the true policy and the recovered policy at time step {t}: {norm_difference}"
        # print(f"Norm difference between the true policy and the recovered policy at time step {t}: {norm_difference}")

    singular_values = np.linalg.svd(r_recovered, compute_uv=False)
    rounded_singular_values = np.round(singular_values, 4)
    # print(f"Singular values of r_recovered (rounded to 4 decimal points): {rounded_singular_values}")
    

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

    # Example usage
 
    basis = row_space_basis(r_recovered)

    print("basis: ", basis.shape)

    def change_of_basis_matrix_v1(B, B_prime):
        """
        Compute the change of basis matrix P such that B P ≈ B'.

        Parameters:
            B (np.ndarray): Original basis matrix (n x m).
            B_prime (np.ndarray): Target basis matrix (n x m).

        Returns:
            P (np.ndarray): Change of basis matrix (m x m).
        """
        # Compute P using the least-squares solution: P = (B^T B)^{-1} B^T B'
        P = np.linalg.pinv(B)@ B_prime
        return P



    def change_of_basis_matrix_v2(B, B_prime):
        """
        Computes the change of basis matrix P such that B' ≈ P B.
        Uses the Moore-Penrose pseudoinverse if B is not square.
        """
        B = np.array(B, dtype=np.float64)
        B_prime = np.array(B_prime, dtype=np.float64)
        
        if B.shape != B_prime.shape:
            print("Error: B and B' must have the same dimensions.")
            return None

        # Compute the pseudoinverse of B
        B_pseudo_inv = np.linalg.pinv(B)
        
        # Compute P
        P = np.dot(B_prime, B_pseudo_inv)
        
        # Compute error
        error = np.linalg.norm(np.dot(P, B) - B_prime)
        print(f"Transformation error: {error:.6f}")
        
        if np.allclose(np.dot(P, B), B_prime):
            print("Exact change of basis matrix found.")
        else:
            print("Only an approximate transformation exists.")

        return P


    P = change_of_basis_matrix_v1(basis, U.T)

    # print("P@basis: ", np.round(np.dot(P, basis), 3))
    # print("The shape of the basis is: ", basis.shape)
    # rounded_basis = np.round(basis, 3)
    coordinates = get_coordinates_wrt_row_basis(r_recovered, np.dot( basis, P))
    print("coordinates: ", coordinates.shape)
    # # print(f"Coordinates of r_recovered with respect to the basis: {coordinates.shape}")
    # n_states = gw.n_states  # Assuming `gw` is the gridworld object with the attribute `n_states`
    # for i, vector in enumerate(rounded_basis):
    #     print(f"Basis vector {i + 1} (rounded to 3 decimal points):")
    #     for j in range(0, len(vector), n_states):
    #         print(vector[j:j + n_states] - min(vector))
    #     print()  # Add an empty line for better readability between basis vectors

   
    # idxs = [0,14]
    # r = r_recovered[:,idxs]
    plot_time_varying_weights(time_varying_weights, coordinates, T)
