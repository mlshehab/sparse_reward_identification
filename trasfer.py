from dynamics import BasicGridWorld, BlockedGridWorld, StickyGridWorld
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

# from main import run_methods, plot_results
# from solvers import solve_PROBLEM_2, solve_PROBLEM_3, solve_PROBLEM_3_RNNM, solve_PROBLEM_3_RTH, solve_PROBLEM_3_regularized
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
    plt.plot(range(T), standardized_true_home, label=r'\textbf{True Reward at Home State (Standardized)}', linestyle='-', linewidth=4)
    plt.plot(range(T), standardized_simple_home, label=r'\textbf{Recovered Reward at Home State (Standardized)}', linestyle='-', linewidth=4)
    plt.plot(range(T), standardized_ashwood_home, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022) at Home State (Standardized)}', linestyle='-', linewidth=4)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Home State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='-', alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(range(T), standardized_true_water, label=r'\textbf{True Reward at Water State (Standardized)}', linestyle='-', linewidth=4)
    plt.plot(range(T), standardized_simple_water, label=r'\textbf{Recovered Reward at Water State (Standardized)}', linestyle='-', linewidth=4)
    plt.plot(range(T), standardized_ashwood_water, label=r'\textbf{dynamic\_irl (Ashwood et al., 2022) at Water State (Standardized)}', linestyle='-', linewidth=4)
    plt.xlabel(r'\textbf{Time}', fontsize=24)
    plt.ylabel(r'\textbf{Weight}', fontsize=24)
    plt.title(r'\textbf{Time-Varying Weight for Water State (Standardized)}', fontsize=28)
    plt.legend(fontsize=20)
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

    # gw1 = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)
    # bgw = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)
    fgw = FrozenGridWorld(grid_size, wind, GAMMA, horizon, 0)

    for state in range(fgw.n_states):
        for action in range(fgw.n_actions):
            next_states = np.where(fgw.transition_probability[state, action] > 0)[0]
            for next_state in next_states:
                prob = fgw.transition_probability[state, action, next_state]
                print(f"State: {state}, Action: {fgw.action_dict_inverse[action]}, Next State: {next_state}, Probability: {prob:.4f}")


    # np.save("results/problem3/gw_P.npy", bgw.P)
    


    # # # print("gw.horizon", gw.horizon)
    # # # print("gw.horizon", gw.horizon)
    # # construct U
    # U = np.zeros(shape=(bgw.n_states*bgw.n_actions, n_features))

    # U[HOME_STATE, 0] = 1.0
    # U[HOME_STATE + bgw.n_states, 0] = 1.0
    # U[HOME_STATE + 2*bgw.n_states, 0] = 1.0
    # U[HOME_STATE + 3*bgw.n_states, 0] = 1.0
    # U[HOME_STATE + 4*bgw.n_states, 0] = 1.0

    # U[WATER_STATE, 1] = 1.0
    # U[WATER_STATE + bgw.n_states, 1] = 1.0
    # U[WATER_STATE + 2*bgw.n_states, 1] = 1.0
    # U[WATER_STATE + 3*bgw.n_states, 1] = 1.0
    # U[WATER_STATE + 4*bgw.n_states, 1] = 1.0



    # true_reward = np.zeros(shape=(bgw.horizon, bgw.n_states, bgw.n_actions))
    # # print("true_reward", true_reward.shape, "time_varying_weights", time_varying_weights.shape)
    # for t in range(bgw.horizon):
    #     for s in range(bgw.n_states):
    #          for a in range(bgw.n_actions):
    #                 true_reward[t, s, a] = U[s + a * bgw.n_states,0] * time_varying_weights[t,0] \
    #                     + U[s + a * bgw.n_states, 1] * time_varying_weights[t,1] 
    #                     # + U[s + a * gw1.n_states, 2] * time_varying_weights[t,2]
    #                 if t == horizon - 1:
    #                     true_reward[t, s, a] *= 100
                        
    
    # true_reward_matrix = np.zeros((bgw.horizon, bgw.n_states * bgw.n_actions))

    # for t in range(bgw.horizon):
    #     for s in range(bgw.n_states):
    #         for a in range(bgw.n_actions):
    #             idx = s + a * bgw.n_states
    #             true_reward_matrix[t, idx] = (
    #                 U[idx, 0] * time_varying_weights[t, 0] +
    #                 U[idx, 1] * time_varying_weights[t, 1]
    #             )
  
    # V, Q, pi = soft_bellman_operation(bgw, true_reward)
    # start_state = 7
    # traj = bgw.simulate_trajectory(7, pi)
    # # print(traj)
    # bgw.visualize_trajectory(traj)

