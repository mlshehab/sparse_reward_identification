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
GEN_DIR_NAME = 'data'

from main import run_methods, plot_results
from solvers import solve_PROBLEM_2, solve_PROBLEM_3, solve_PROBLEM_3_RNNM, solve_PROBLEM_3_RTH, solve_PROBLEM_3_regularized, feasible_reward
from scipy.optimize import minimize

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
    n_features = 3
    
    # choose noise covariance for the random walk priors over weights corresponding to these maps
    sigmas = [2**-(3.5)]*n_features

    # print(sigmas)
    weights0 = np.zeros(n_features) #initial weights at t=0
    weights0[1] = 1



    gw = BasicGridWorld(grid_size, wind, GAMMA, horizon, 0)

    # print("gw.horizon", gw.horizon)
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

    # Desired action
    desired_action = 4 # stay
    U[HOME_STATE + desired_action*gw.n_states, 2] = 1.0
    U[WATER_STATE + desired_action*gw.n_states, 2] = 1.0

    time_varying_weights  = pickle.load(open(GEN_DIR_NAME + 
                                    "/output_data.pickle", 'rb'))['gen_time_varying_weights']   
    
    time_varying_weights = time_varying_weights.T

    # Append a third column to time_varying_weights
    third_column = np.zeros((time_varying_weights.shape[0], 1))
    third_column[-1, 0] = 1
    time_varying_weights = np.hstack((time_varying_weights, third_column))

    # Generate time-varying weights for all map

    true_reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    # true_reward_2 = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    print("true_reward", true_reward.shape, "time_varying_weights", time_varying_weights.shape)

    for t in range(gw.horizon):
        for s in range(gw.n_states):
             for a in range(gw.n_actions):
                    true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                            + U[s + a * gw.n_states, 1] * time_varying_weights[t,1] \
                            + + U[s + a * gw.n_states, 2] * time_varying_weights[t,2]
                    # if t == gw.horizon - 1:
                    #     true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                    #         + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]
                    # if t == np.floor(gw.horizon/2):
                    #     true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                    #         + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]
                    # if t == 35:
                    #     true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                    #         + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]
                    # if t == 10:
                    #     true_reward[t, s, a] = U[s + a * gw.n_states,0] * time_varying_weights[t,0] \
                    #         + U[s + a * gw.n_states, 1] * time_varying_weights[t,1]
                        
    # for s in range(gw.n_states):
    #      for a in range(gw.n_actions):
    #           true_reward_2[gw.horizon - 2, s, a] = true_reward[gw.horizon - 1, s, a]

    # true_reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    # # print("true_reward", true_reward.shape, "time_varying_weights", time_varying_weights.shape)
    # for t in range(gw.horizon):
    #     for s in range(gw.n_states):
    #          for a in range(gw.n_actions):
    #                 if t == gw.horizon -2:
    #                     true_reward[t, s, a] = 1.0
    #                     if a == 0:
    #                         true_reward[t, s, a] += 10.0 
    #                 if t == gw.horizon -3:
    #                     true_reward[t, s, a] = 100.0
    
    V, Q, pi = soft_bellman_operation(gw, true_reward)

    r, nu = feasible_reward(gw, U, pi)

    singular_values = np.linalg.svd(r, compute_uv=False)
    print("Singular values of r:", np.round(singular_values, 4))

    # # Solve the problem with the new reward f
    # r_nuc, nu_nuc = solve_PROBLEM_3(gw, U, sigmas, pi)
    # # print("min r[:, 0] = ", np.min(r[:, 0]))


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

    # # Compute row space basis for both r_recovered_RTH and r_recovered_simple
    # # basis_RTH = row_space_basis(r_recovered_RTH)
    # basis_simple = row_space_basis(r_nuc, top_k=n_features)

    # def change_of_basis_matrix(B, B_prime):
    #     """
    #     Compute the change of basis matrix P such that B P ≈ B'.
    #     """
    #     P = np.linalg.pinv(B) @ B_prime
    #     return P

    # # Compute change of basis matrices for both RTH and simple
    # # P_RTH = change_of_basis_matrix(basis_RTH, U.T)
    # P_simple = change_of_basis_matrix(basis_simple, U.T)

    # # # Compute coordinates with respect to the new basis for both RTH and simple
    # # # coordinates_RTH = get_coordinates_wrt_row_basis(r_recovered_RTH, np.dot(basis_RTH, P_RTH))
    # coordinates_simple = get_coordinates_wrt_row_basis(r_nuc, np.dot(basis_simple, P_simple))
    # # print("coordinates_simple", coordinates_simple.shape)
    # # # Plot coordinates_simple
    # # plt.figure(figsize=(10, 6))
    # # for i in range(coordinates_simple.shape[1]):
    # #     plt.plot(coordinates_simple[:, i], label=f'Coordinate {i}')

    # # plt.axvline(x=np.floor(gw.horizon / 2), color='r', linestyle='--', label='Midpoint 1')
    # # plt.axvline(x=np.floor(gw.horizon / 2)-1, color='g', linestyle='--', label='Midpoint 2') 
    # # plt.xlabel('Time step')
    # # plt.ylabel('Coordinate value')
    # # plt.title('Coordinates with respect to the new basis (simple)')
    # # plt.legend()
    # # plt.show()




    # # Standardize coordinates_simple
    # coordinates_simple_standardized = (coordinates_simple - np.mean(coordinates_simple, axis=0)) / np.std(coordinates_simple, axis=0)

    # diff1 = r[-2,0] - r[0,0]
    # print("diff1 = ", diff1)
    # diff2 = coordinates_simple[-2,0] - coordinates_simple[0,0]
    # print("diff2 = ", diff2)
    # ratio = diff1 / diff2
    # print("ratio = ", ratio)
   

    # # Plot standardized coordinates_simple
    # plt.figure(figsize=(10, 6))

    # # Plot standardized coordinates_simple
    # plt.figure(figsize=(10, 6))
    # # for i in range(coordinates_simple_standardized.shape[1]):
    # plt.plot(-ratio*coordinates_simple[:, 0] + r[0,0]-coordinates_simple[0,0], label=f'Coordinate {0}')

    # # Standardize r and true_reward
    # r_standardized = (r[:, 0] - np.mean(r[:, 0])) / np.std(r[:, 0])
    # true_reward_standardized = (true_reward[:, 0, 0] - np.mean(true_reward[:, 0, 0])) / np.std(true_reward[:, 0, 0])

    # plt.plot(-r[:, 0], label='r[:, 0]')
    # plt.plot(true_reward[:, 0, 0], label='true_reward[:, 0, 0]')
    # plt.axvline(x=13, color='r', linestyle='--', label='Midpoint 1')
    # plt.axvline(x=14, color='g', linestyle='--', label='Midpoint 2') 
    # plt.axvline(x=30, color='r', linestyle='--', label='Midpoint 3')
    # plt.axvline(x=31, color='g', linestyle='--', label='Midpoint 4') 
    # plt.axvline(x=41, color='b', linestyle='--', label='Midpoint 5')
    # plt.axvline(x=42, color='y', linestyle='--', label='Midpoint 6')
    # plt.legend()
    # plt.xlabel('Time step')
    # plt.ylabel('Value')
    # plt.title('Comparison of r[:, 0] and true_reward[:, 0, 0]')
    # plt.show()
    # plt.show()






    # # print("nu = ", nu)

    # true_reward_2 = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    # # print("true_reward", true_reward.shape, "time_varying_weights", time_varying_weights.shape)
    # for t in range(gw.horizon):
    #     for s in range(gw.n_states):
    #          for a in range(gw.n_actions):
    #                 if t == 0:
    #                     true_reward_2[t, s, a] = 1.0
    #                 if t == 5:
    #                     true_reward_2[t, s, a] = 100.0

    # V_2, Q_2, pi_2 = soft_bellman_operation(gw, true_reward_2)

    # for t in range(gw.horizon):
    #     norm_diff = np.linalg.norm(pi[t] - pi_2[t])
    #     print(f"Time step {t}: Norm difference between pi and pi_2 = {norm_diff}")
    