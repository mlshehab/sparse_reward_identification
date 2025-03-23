from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
# from solvers import solve_milp
from utils.bellman import state_only_soft_bellman_operation, soft_bellman_operation, time_varying_value_iteration
import numpy as np
from utils.checks import is_markovian
import pickle
import os
import sys
import time
import datetime
import matplotlib.pyplot as plt

# path_to_dynamic_irl = '~\Desktop'
# repo2_path = os.path.expanduser(path_to_dynamic_irl)  # Adjust path if necessary
# sys.path.append(repo2_path)

# repo2_path = os.path.expanduser("~/Desktop/dynamic_irl")  # Adjust the path to your setup
# if repo2_path not in sys.path:
#     sys.path.append(repo2_path)

# repo2_path = os.path.expanduser("~/Desktop/dynamic_irl/src")  # Adjust the path to your setup
# if repo2_path not in sys.path:
#     sys.path.append(repo2_path)

# repo2_path = os.path.expanduser("~/Desktop/dynamic_irl/src/optimize_weights")  # Adjust the path to your setup
# if repo2_path not in sys.path:
#     sys.path.append(repo2_path)


# import dynamic_irl

# from dynamic_irl.src.envs  import  gridworld

# from dynamic_irl.src.simulate_data_gridworld import generate_expert_trajectories
# from dynamic_irl.src.simulate_data_gridworld import create_goal_maps
# # from dynamic_irl.src.dirl_for_gridworld import fit_dirl_gridworld

# from main import run_methods, plot_results
from noisy_solvers import solve_PROBLEM_2_noisy#, solve_PROBLEM_3
from solvers import solve_PROBLEM_2

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


def plot_time_varying_weights(true_weights, recovered_weights, T):
    """
    Plots the time-varying weights for the home state and water state, comparing true and recovered weights.

    Parameters:
    - true_weights: numpy array of shape (T, num_maps), the true time-varying weights
    - recovered_weights: numpy array of shape (T, num_maps), the recovered time-varying weights
    - T: int, number of time steps
    """
    # Enable LaTeX rendering
    # plt.rcParams.update({
    #     "text.usetex": True,  # Use LaTeX for text rendering
    #     "font.family": "serif",  # Use a serif font
    #     "font.serif": ["Computer Modern"],  # Default LaTeX font
    # })
    plt.figure(figsize=(12, 6))

    # Plot weight 0 (reward at home state)
    plt.figure(figsize=(14, 7), dpi=150)  # Increase figure size and DPI for better quality
    plt.subplot(1, 2, 1)
    plt.plot(range(T), true_weights[:, 0], label='True Reward at Home State', linestyle='--', linewidth=2)
    plt.plot(range(T), recovered_weights[:, 0], label='Recovered Reward at Home State', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Time-Varying Weight for Home State', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability

    # Plot weight 1 (reward at water state)
    plt.subplot(1, 2, 2)
    plt.plot(range(T), true_weights[:, 1], label='True Reward at Water State', linestyle='--', linewidth=2)
    plt.plot(range(T), recovered_weights[:, 1], label='Recovered Reward at Water State', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Weight', fontsize=12)
    plt.title('Time-Varying Weight for Water State', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)  # Add grid for better readability

    plt.tight_layout()
    # plt.show()
    # Save the figure
    import os
    from datetime import datetime

    # Create directories if they don't exist
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/problem2"):
        os.makedirs("results/problem2")

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the figure with a timestamp
    plt.savefig(f"results/noisy/problem2/feas_time_varying_weights_{timestamp}.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    
    # np.random.seed(int(time.time()))
    np.random.seed(0)

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
    

    # for t in range(gw.horizon):
    #     print(f"At time step {t}:")
    #     print(f"true reward: {true_reward[t, :, :]}")
    
    # print(true_reward)
    V, Q, pi = soft_bellman_operation(gw, true_reward)


        ## Sample from pi to obtain demonstration
    pi_hat = np.zeros_like(pi)
    alpha = pi.min()
    print(f"{alpha=}")
    print(f"{gw.n_actions=}")
    delta = 1-1e-4
    NUM_SAMPLES = 50_000_000
    for t in range(gw.horizon):
        for s in range(gw.n_states):
            # Sample NUM_SAMPLES actions from the distribution pi[t, s, :]

            # actions = np.random.choice(gw.n_actions, size=NUM_SAMPLES, p=pi[t, s, :])
            
            # Count occurrences of each action
            counts = np.random.multinomial(NUM_SAMPLES, pi[t, s, :])    
            
            # Normalize to get empirical distribution
            pi_hat[t, s, :] = counts / NUM_SAMPLES

    epsilon = np.sqrt(2/NUM_SAMPLES * np.log((2**(gw.n_actions)-2)/(1-delta)))

    new_epsilon = np.sqrt(1/(2*NUM_SAMPLES) * np.log((2/(1-delta))))


    print(f"{epsilon=}")
    print(f"{new_epsilon=}")
    b = np.ones(shape=(gw.horizon, gw.n_states))*new_epsilon/alpha
    print(f"{epsilon/alpha=}")
    print(f"{new_epsilon/alpha=}")

    print(f"{gw.horizon*gw.n_states*gw.n_actions}")
    print("Ratio of bound exceeds:")
    print(np.sum(np.abs(pi-pi_hat) > epsilon)/gw.horizon/gw.n_states/gw.n_actions)            
    print("Ratio of bound exceeds - new epsilon:")
    print(np.sum(np.abs(pi-pi_hat) > new_epsilon)/gw.horizon/gw.n_states/gw.n_actions)      


    print("Ratio of bound exceeds - log:")
    print(np.sum(np.abs(np.log(pi)-np.log(pi_hat)) > epsilon/alpha)/gw.horizon/gw.n_states/gw.n_actions)
    
    alpha_values, *sol  = solve_PROBLEM_2_noisy(gw, U, sigmas, pi, np.ones_like(b)*1e-4)
    # alpha_values, *sol  = solve_PROBLEM_2(gw, U, sigmas, pi)#, np.ones_like(b)*1e-4)

  
    plot_time_varying_weights(time_varying_weights, alpha_values, T)
