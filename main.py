from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
from solvers import solve_milp, solve_L_1, solve_L_inf, solve_L2
import numpy as np
import pickle
GEN_DIR_NAME = 'data'
import time
import matplotlib.pyplot as plt
import sys
import os
# Add the path to the dynamic_irl module
path_to_dynamic_irl = '~\Desktop'
repo2_path = os.path.expanduser(path_to_dynamic_irl)  # Adjust path if necessary
sys.path.append(repo2_path)
# print(sys.path)
# sys.path.append('/home/mlshehab/Desktop/dynamic_irl')
import dynamic_irl

from dynamic_irl.src.envs  import  gridworld




def run_methods(gw, pi, methods):
    results = {}
    for method in methods:
        start_time = time.time()
        if method == "MILP":
            r, nu, z = solve_milp(gw, pi)
            results["MILP"] = (r, nu, z)
            print(f"MILP done in {time.time() - start_time:.2f} seconds")
        elif method == "L1":
            r, nu = solve_L_1(gw, pi)
            results["L1"] = (r, nu)
            print(f"L1 done in {time.time() - start_time:.2f} seconds")
        elif method == "Linf":
            r, nu = solve_L_inf(gw, pi)
            results["Linf"] = (r, nu)
            print(f"Linf done in {time.time() - start_time:.2f} seconds")
        elif method == "L2":
            r, nu = solve_L2(gw, pi)
            results["L2"] = (r, nu)
            print(f"L2 done in {time.time() - start_time:.2f} seconds")
    return results

def plot_results(gw, reward, results):
    print("The shape of reward is", reward.shape)
    plt.figure(figsize=(10, 6))
    true_reward_avg = np.mean(reward[:, 0, :], axis=1)
    plt.plot(range(gw.horizon), true_reward_avg, label='True Reward', linestyle='--')

    for method, (r, _) in results.items():
        reward_avg = np.mean(r[:, 0, :], axis=1)
        reward_avg -= reward_avg[0]  # Adjust to start from 0
        plt.plot(range(gw.horizon), 2*reward_avg, label=method)

    plt.xlabel('Time')
    plt.ylabel('Average Reward at State 0')
    plt.title('Average Reward at State 0 over Time for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":


    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    start_state = 10

    gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
    
    gen_time_varying_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['time_varying_weights']

    # now obtain time-varying reward maps
    rewards = np.zeros((gw.horizon, gw.n_states)) #array of size Txnum_states
    rewards = rewards.T

    r_map = np.reshape(np.array(rewards[:, 0]), (gw.grid_size, gw.grid_size), order='F')
    # r_map = 0
    env = gridworld.GridWorld(r_map, {},)  # instantiate
    # gridworld environment.  {} indicates that there are no terminal states
    P_a = env.get_transition_mat()

    P = [P_a[:,:,0], P_a[:,:,1], P_a[:,:,2], P_a[:,:,3], P_a[:,:,4]]
    
    gw.P = P

    HOME_STATE = 0
    WATER_STATE = 14

    reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    for t in range(gw.horizon):
        reward[t, HOME_STATE, :] = gen_time_varying_weights[t][0]
        reward[t, WATER_STATE, :] = gen_time_varying_weights[t][1]

    V, Q, pi = soft_bellman_operation(gw, reward)


    # # Choose which methods to run
    # methods_to_run = ["L1", "L2", "Linf"]
    # results = run_methods(gw, pi, methods_to_run)

    # # reward_L1 = results["L1"][0]
    # # reward_L2 = results["L2"][0]
    # # reward_Linf = results["Linf"][0]
    # # print("The shape of reward_L1 is", reward_L1.shape)

    # # V_L1, Q_L1, pi_L1 = soft_bellman_operation(gw, reward_L1)
    # # V_L2, Q_L2, pi_L2 = soft_bellman_operation(gw, reward_L2)
    # # V_Linf, Q_Linf, pi_Linf = soft_bellman_operation(gw, reward_Linf)
    # # Compare the policies
    # def compare_policy_norms(pi, pi_other, horizon):
    #     norm_diffs = np.zeros(horizon)
    #     for t in range(horizon):
    #         norm_diffs[t] = np.linalg.norm(pi[t] - pi_other[t])
    #     return norm_diffs

    # # norm_diffs_L1 = compare_policy_norms(pi, pi_L1, gw.horizon)
    # # norm_diffs_L2 = compare_policy_norms(pi, pi_L2, gw.horizon)
    # # norm_diffs_Linf = compare_policy_norms(pi, pi_Linf, gw.horizon)

    # # print("Norm differences between true policy and L1, L2, Linf policies at each time step:")
    # # for t in range(gw.horizon):
    # #     print(f"Time {t}: L1: {norm_diffs_L1[t]:.6f}, L2: {norm_diffs_L2[t]:.6f}, Linf: {norm_diffs_Linf[t]:.6f}")

    # # Plot the results
    # plot_results(gw, reward, results)

