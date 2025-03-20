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


import dynamic_irl

from dynamic_irl.src.envs  import  gridworld

from dynamic_irl.src.simulate_data_gridworld import generate_expert_trajectories
from dynamic_irl.src.simulate_data_gridworld import create_goal_maps
# from dynamic_irl.src.dirl_for_gridworld import fit_dirl_gridworld

from main import run_methods, plot_results

GEN_DIR_NAME = 'data'


if __name__ == "__main__":

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
   


    gen_time_varying_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['time_varying_weights']


    # select number of maps
    num_maps = 2

    # choose noise covariance for the random walk priors over weights corresponding to these maps
    sigmas = [2**-(3.5)]*num_maps

    weights0 = np.zeros(num_maps) #initial weights at t=0
    weights0[1] = 1

    goal_maps = create_goal_maps(gridworld_H*gridworld_W, WATER_STATE, HOME_STATE)
    #size is num_maps x num_states

    #generate time-varying weights
    time_varying_weights = gen_time_varying_weights
    # now obtain time-varying reward maps
    rewards = time_varying_weights@goal_maps #array of size Txnum_states
    rewards = rewards.T

    r_map = np.reshape(np.array(rewards[:, 0]), (gridworld_H, gridworld_W), order='F')
    gw = gridworld.GridWorld(r_map, {},)  # instantiate
    # gridworld environment.  {} indicates that there are no terminal states
    P_a = gw.get_transition_mat()


    gw = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)
    gw.P = []
    # creating a transition prob matrix consistent with the other repo
    # P_a = np.zeros((gw.n_states, gw.n_states, gw.n_actions))
    for a in range(gw.n_actions):
        gw.P.append(P_a[:,:,a])


    assert is_markovian(P_a), "The transition probability matrix P_a is not Markovian for each action."
   

    gen_time_varying_weights = pickle.load(open(GEN_DIR_NAME + 
                                    "/generative_parameters.pickle", 'rb'))['time_varying_weights']

    

    true_reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    for t in range(gw.horizon):
        true_reward[t, HOME_STATE, :] = gen_time_varying_weights[t][0]
        true_reward[t, WATER_STATE, :] = gen_time_varying_weights[t][1]

    V, Q, pi = soft_bellman_operation(gw, true_reward)

    time_varying_policy = pi

  
    trajs_all_experts = []
    for expert in range(N_experts):
        traj = generate_expert_trajectories(gridworld_H, gridworld_W, rewards, time_varying_policy, T, GAMMA)
        trajs_all_experts.append(traj)

    resampled_trajs_all_experts = []
    for expert in range(20):
        traj = generate_expert_trajectories(gridworld_H, gridworld_W, rewards, time_varying_policy, T, GAMMA)
        resampled_trajs_all_experts.append(traj)




    # print(trajs_all_experts)
    save_dir  = os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'dynamic_irl',
        'data',
        'simulated_gridworld_data',
        'exclude_explore_share_weights_1',
    )
   
    file_name = save_dir+'/expert_trajectories.pickle'

    if os.path.exists(file_name):
        os.remove(file_name)

    with open(file_name,'wb') as handle:
        pickle.dump(trajs_all_experts, handle, protocol=pickle.HIGHEST_PROTOCOL)



    rec_dir_name = os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'dynamic_irl',
        'recovered_parameters',
        'gridworld_recovered_params',
        'exclude_explore_1',
        'maps_2_lr_0.001_0.05'
    )
    



    rec_goal_maps = np.load(rec_dir_name + "\goal_maps_trajs_200_seed_1_iters_100.npy")[-1]


    # load recovered parameters for this seed
    rec_weights = np.load(rec_dir_name + "\weights_trajs_200_seed_1_iters_100.npy")[-1] 

    rec_rewards = np.repeat((rec_weights.T @ rec_goal_maps)[:, :, np.newaxis], gw.n_actions, axis=2)
    rec_rewards = np.concatenate((rec_rewards, rec_rewards[-1:, :, :]), axis=0)

    path_to_expert_trajectories = os.path.join(
        os.path.expanduser('~'),
        'Desktop',
        'dynamic_irl',
        'data',
        'simulated_gridworld_data',
        'exclude_explore_share_weights_1',
        'expert_trajectories.pickle'
    )  

    def compute_and_print_log_likelihood(gw, reward, trajectories):
        V, Q, pi = soft_bellman_operation(gw, reward)
        # concatenate expert trajectories
        assert(len(trajectories) > 0), "no expert trajectories found!"
        state_action_pairs = []
        for num, traj in enumerate(trajectories):
            states = np.array(traj['states'])[:, np.newaxis]
            actions = np.array(traj['actions'])[:, np.newaxis]
            if len(states) == len(actions) + 1:
                states = np.array(traj['states'][:-1])[:, np.newaxis]
            assert len(states) == len(actions), "states and action sequences don't have the same length"
            T = len(states)
            state_action_pairs_this_traj = np.concatenate((states, actions), axis=1)
            assert state_action_pairs_this_traj.shape[0] == len(states), "error in concatenation of s,a,s' tuples"
            assert state_action_pairs_this_traj.shape[1] == 2, "states and actions are not integers?"
            state_action_pairs.append(state_action_pairs_this_traj)

        # compute the log likelihood for all trajectories
        num_trajectories = len(state_action_pairs)
        log_likelihood = 0
        for i in range(num_trajectories):
            states, actions = state_action_pairs[i][:, 0], state_action_pairs[i][:, 1]
            log_likelihood += sum(np.log(pi[range(T), states, actions]))

        print("The log likelihood is", log_likelihood)


        # Choose which methods to run
    methods_to_run = ["L1", "L2", "Linf"]
    results = run_methods(gw, pi, methods_to_run)

    reward_L1 = results["L1"][0]
    reward_L2 = results["L2"][0]
    reward_Linf = results["Linf"][0]

    # compute the log likelihood for the recovered rewards
    compute_and_print_log_likelihood(gw, rec_rewards, resampled_trajs_all_experts)
    # compute the log likelihood for the true rewards
    compute_and_print_log_likelihood(gw, true_reward, resampled_trajs_all_experts)
    compute_and_print_log_likelihood(gw, reward_L1, resampled_trajs_all_experts)
    compute_and_print_log_likelihood(gw, reward_L2, resampled_trajs_all_experts)
    compute_and_print_log_likelihood(gw, reward_Linf, resampled_trajs_all_experts)

    plot_results(gw, true_reward, results)
 


    """
    TODO:
    1. Visualize the two policies, see if they are doing the right thing
    2. Split into a train and test set. 
    """
 
