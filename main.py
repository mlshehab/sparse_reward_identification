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


    path_to_policy = os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'dynamic_irl',
        'data',
        'simulated_gridworld_data',
        'exclude_explore_share_weights_1',
    )

    # Print the files in path_to_policy
    # print("Files in path_to_policy:")
    # for root, dirs, files in os.walk(path_to_policy):
    #     for file in files:
    #         print(file)


    policy_r = pickle.load(open(path_to_policy + "\generative_parameters.pickle", 'rb'))['time_varying_policy']
    print("The shape of policy_r is", policy_r.shape)
    print("The shape of pi is", pi.shape)
    for t in range(gw.horizon):
        norm_diff = np.linalg.norm(pi[t] - policy_r[t])
        print(f"Time {t}: Norm difference between pi and policy_r: {norm_diff:.6f}")

#     path_to_expert_trajectories = os.path.join(
#     os.path.expanduser('~'),
#     'Desktop',
#     'dynamic_irl',
#     'data',
#     'simulated_gridworld_data',
#     'exclude_explore_share_weights_1',
#     'expert_trajectories.pickle'
# )   
#     rec_dir_name = os.path.join(
#         os.path.expanduser('~'),
#         'Desktop',
#         'dynamic_irl',
#         'recovered_parameters',
#         'gridworld_recovered_params',
#         'exclude_explore_1',
#         'maps_2_lr_0.001_0.05'
#     )

#     rec_goal_maps = np.load(rec_dir_name + "\goal_maps_trajs_200_seed_1_iters_100.npy")[-1]


#     # load recovered parameters for this seed
#     rec_weights = np.load(rec_dir_name + "\weights_trajs_200_seed_1_iters_100.npy")[-1] 

#     rec_rewards = np.repeat((rec_weights.T @ rec_goal_maps)[:, :, np.newaxis], gw.n_actions, axis=2)
#     rec_rewards = np.concatenate((rec_rewards, rec_rewards[-1:, :, :]), axis=0)

    
#     print("The shape of rec_rewards is", rec_rewards.shape)

#     def compute_and_print_log_likelihood(gw, reward, path_to_expert_trajectories):
#         V, Q, pi = soft_bellman_operation(gw, reward)

#         trajectories = pickle.load(open(path_to_expert_trajectories, 'rb'))

#         # concatenate expert trajectories
#         assert(len(trajectories) > 0), "no expert trajectories found!"
#         state_action_pairs = []
#         for num, traj in enumerate(trajectories):
#             states = np.array(traj['states'])[:, np.newaxis]
#             actions = np.array(traj['actions'])[:, np.newaxis]
#             if len(states) == len(actions) + 1:
#                 states = np.array(traj['states'][:-1])[:, np.newaxis]
#             assert len(states) == len(actions), "states and action sequences don't have the same length"
#             T = len(states)
#             state_action_pairs_this_traj = np.concatenate((states, actions), axis=1)
#             assert state_action_pairs_this_traj.shape[0] == len(states), "error in concatenation of s,a,s' tuples"
#             assert state_action_pairs_this_traj.shape[1] == 2, "states and actions are not integers?"
#             state_action_pairs.append(state_action_pairs_this_traj)

#         # compute the log likelihood for all trajectories
#         num_trajectories = len(state_action_pairs)
#         log_likelihood = 0
#         for i in range(num_trajectories):
#             states, actions = state_action_pairs[i][:, 0], state_action_pairs[i][:, 1]
#             log_likelihood += sum(np.log(pi[range(T), states, actions]))

#         print("The log likelihood is", log_likelihood)

    




#     # now i need to plot the expert trajectories
#     # plt.figure(figsize=(10, 6))
#     # # for trajectory in expert_trajectories:
#     # plt.plot(expert_trajectories[0]['states'])
#     # plt.show()





#     # Choose which methods to run
#     methods_to_run = ["L1", "L2", "Linf"]
#     results = run_methods(gw, pi, methods_to_run)

#     reward_L1 = results["L1"][0]
#     reward_L2 = results["L2"][0]
#     reward_Linf = results["Linf"][0]

#     # compute the log likelihood for the recovered rewards
#     compute_and_print_log_likelihood(gw, rec_rewards, path_to_expert_trajectories)
#     # compute the log likelihood for the true rewards
#     compute_and_print_log_likelihood(gw, reward, path_to_expert_trajectories)
#     compute_and_print_log_likelihood(gw, reward_L1, path_to_expert_trajectories)
#     compute_and_print_log_likelihood(gw, reward_L2, path_to_expert_trajectories)
#     compute_and_print_log_likelihood(gw, reward_Linf, path_to_expert_trajectories)



#     # # print("The shape of reward_L1 is", reward_L1.shape)

#     # # V_L1, Q_L1, pi_L1 = soft_bellman_operation(gw, reward_L1)
#     # # V_L2, Q_L2, pi_L2 = soft_bellman_operation(gw, reward_L2)
#     # # V_Linf, Q_Linf, pi_Linf = soft_bellman_operation(gw, reward_Linf)
#     # # Compare the policies
#     # def compare_policy_norms(pi, pi_other, horizon):
#     #     norm_diffs = np.zeros(horizon)
#     #     for t in range(horizon):
#     #         norm_diffs[t] = np.linalg.norm(pi[t] - pi_other[t])
#     #     return norm_diffs

#     # # norm_diffs_L1 = compare_policy_norms(pi, pi_L1, gw.horizon)
#     # # norm_diffs_L2 = compare_policy_norms(pi, pi_L2, gw.horizon)
#     # # norm_diffs_Linf = compare_policy_norms(pi, pi_Linf, gw.horizon)

#     # # print("Norm differences between true policy and L1, L2, Linf policies at each time step:")
#     # # for t in range(gw.horizon):
#     # #     print(f"Time {t}: L1: {norm_diffs_L1[t]:.6f}, L2: {norm_diffs_L2[t]:.6f}, Linf: {norm_diffs_Linf[t]:.6f}")

#     # # Plot the results
#     # plot_results(gw, reward, results)

