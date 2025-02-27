from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
from solvers import solve_milp, solve_L_1, solve_L_inf
import numpy as np
import pickle
GEN_DIR_NAME = 'data'
import time
import matplotlib.pyplot as plt

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
    return results

def plot_results(gw, reward, results):
    plt.figure(figsize=(10, 6))
    true_reward_avg = np.mean(reward[:, 0, :], axis=1)
    plt.plot(range(gw.horizon), true_reward_avg, label='True Reward', linestyle='--')

    for method, (r, _) in results.items():
        reward_avg = np.mean(r[:, 0, :], axis=1)
        plt.plot(range(gw.horizon), reward_avg, label=method)

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

    HOME_STATE = 0
    WATER_STATE = 14

    reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
    for t in range(gw.horizon):
        reward[t, HOME_STATE, :] = gen_time_varying_weights[t][0]
        reward[t, WATER_STATE, :] = gen_time_varying_weights[t][1]

    V, Q, pi = soft_bellman_operation(gw, reward)

    # Choose which methods to run
    methods_to_run = ["L1", "Linf"]
    results = run_methods(gw, pi, methods_to_run)

    # Plot the results
    plot_results(gw, reward, results)

