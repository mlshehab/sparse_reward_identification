import numpy as np
from numba import set_num_threads

import os
import time 

import gurobipy as gp
from gurobipy import GRB

from solvers import solve_greedy_backward_bisection
from noisy_solvers import solve_milp_noisy, solve_greedy_backward_bisection_noisy, solve_greedy_backward_bisection_smaller_noisy#, solve_greedy_backward_alpha
from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation

from utils.sample import estimate_pi_and_visits_numba, compute_likelihood

NUMBER_OF_EXPERIMENTS = 1
# NUMBER_OF_FEATURES = 7


def check_feasibility(gw, pi, r, nu, b):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    epsilon = 1e-6
    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                if not (r[t, s, a] - epsilon <= np.log(pi[t, s, a]) + b[t, s, a] + nu[t,s] 
                               - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
                    return False
                elif not (r[t, s, a] + epsilon >= np.log(pi[t, s, a]) - b[t, s, a] + nu[t,s] 
                                  - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
                    return False
        print(f"Time {t} is feasible")
    
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
            if not (r[T-1, s, a] - epsilon <= np.log(pi[T-1, s, a]) + b[T-1, s, a] + nu[T-1, s]):
                return False
            elif not (r[T-1, s, a] + epsilon >= np.log(pi[T-1, s, a]) - b[T-1, s, a] + nu[T-1, s]):
                return False
    print(f"Time {T-1} is feasible")
    return True     




def compute_error_bound(gw, delta, visit_counts, pi_hat):
        log_term = np.log(2 / (1 - delta))

        b = np.full((gw.horizon, gw.n_states, gw.n_actions), 1e3)  # default to inf for unvisited

        # Where visits > 0, compute epsilon and b
        visited_mask = visit_counts > 0
        epsilons = np.zeros_like(visit_counts, dtype=np.float64)
        epsilons[visited_mask] = np.sqrt(1 / (2 * visit_counts[visited_mask]) * log_term)

        # Compute alpha[t, s, a] = pi_hat[t, s, a] - epsilons[t, s]
        epsilon_broadcast = np.broadcast_to(epsilons[:, :, np.newaxis], pi_hat.shape)  # shape (H, S, A)
        alpha = pi_hat - epsilon_broadcast

        # Assert all alpha values are positive where visited
        visited_mask_3d = np.broadcast_to(visited_mask[:, :, np.newaxis], alpha.shape)
        print(alpha.min())
        assert np.all(alpha[visited_mask_3d] > 0), "Alpha contains non-positive values!"

        # Avoid division by zero or negative alpha values
        safe_mask = visited_mask_3d & (alpha > 0)

        # b = np.full_like(pi_hat, 1e3)
        b[safe_mask] = epsilon_broadcast[safe_mask] / alpha[safe_mask]
        print(b.shape)

        # Check b for irregularities and print them
        irregular_mask = (b < 0) | ~np.isfinite(b) | np.isnan(b)
        irregular_indices = np.argwhere(irregular_mask)
        for t, s, a in irregular_indices:
            print(f"Irregular b at (t={t}, s={s}, a={a}): b={b[t, s, a]}, epsilon={epsilons[t, s]}, alpha={alpha[t, s, a]}")

        # # Compute fraction of (t, s, a) entries where deviation exceeds epsilon
        # violation_fraction = np.sum(np.abs(pi - pi_hat) > epsilon_broadcast) / (gw.horizon * gw.n_states * gw.n_actions)

        # print("Violation fraction: ", violation_fraction)
        print(f"{epsilons.max()=}, {epsilons.min()=}")
        return b


def compare_likelihood_experiment(num_trajectories, seed):
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    number_of_switches = 5
    # output_path = "results/problem1_new_rewards/"
    print(f"{num_trajectories=}")
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    np.random.seed(seed)
    reward_path = f"data/rewards/reward_{seed}_{number_of_switches}.npy"

    gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
    # now obtain time-varying reward maps

    if os.path.exists(reward_path):
        reward = np.load(reward_path)
    else:
        print(reward_path)
        generate_and_save_rewards(seed)
        reward = np.load(reward_path)

    V, Q, pi = soft_bellman_operation(gw, reward)

    delta = 1-1e-4

    P = np.asarray(gw.P, dtype=np.float64)     # shape (A, S, S)
    pi = np.asarray(pi, dtype=np.float64)   # shape (H, S, A)
    action_counts, visit_counts = estimate_pi_and_visits_numba(P, pi, gw.horizon, num_trajectories)


    action_counts_val, visit_counts_val = estimate_pi_and_visits_numba(P, pi, gw.horizon, num_trajectories)

    if (visit_counts == 0).any():
        print("Still there are t,s pairs not visited")

    # Normalize to get pi_hat
    pi_hat = np.zeros_like(pi)/gw.n_actions
    with np.errstate(divide='ignore', invalid='ignore'):
        pi_hat = np.divide(action_counts, visit_counts[:, :, None], where=visit_counts[:, :, None] != 0)

    b = compute_error_bound(gw, delta, visit_counts, pi_hat)

    # r_greedy, nu_greedy, switch_times = solve_greedy_backward_bisection(gw, pi)


    r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_smaller_noisy(gw, pi_hat, b)
    assert check_feasibility(gw, pi_hat, r_greedy, nu_greedy, b)
    V, Q, pi_hat_hat = soft_bellman_operation(gw, r_greedy)
    pi_hat_hat_likelihood = compute_likelihood(pi_hat_hat, visit_counts_val, action_counts_val)
    pi_hat_likelihood = compute_likelihood(pi_hat, visit_counts_val, action_counts_val)
    pi_likelihood = compute_likelihood(pi, visit_counts_val, action_counts_val)

    print("Log likelihoods")
    print("True Policy: ", pi_likelihood)
    print("Estimate Policy: ", pi_hat_likelihood)
    print("Reconstructed Policy: ", pi_hat_hat_likelihood)

    return (pi_likelihood, pi_hat_likelihood, pi_hat_hat_likelihood)





# if __name__ == "__main__":

#     # compare_likelihood_experiment(1, 500_000)

#     from multiprocessing import Process
#     from itertools import product

#     # Define your input lists
#     seed_list = [4,5,6,7,8,9,10]
#     num_traj_list = [300_000, 800_000]
#     # num_traj_list = [12_000_000, 16_000_000, 24_000_000]#, 8_000_000]
#     # num_traj_list = [120_000_000, 240_000_000]#[36_000_000, 48_000_000, 60_000_000]#, 8_000_000]

#     # for seed in seed_list:
#     #     generate_and_save_rewards(seed)

#     # Create Cartesian product of parameters
#     arg_list = list(product(num_traj_list, seed_list))

#     # Launch each in its own process
#     set_num_threads(60//len(arg_list))
#     processes = []
#     for x, y in arg_list:
#         p = Process(target=compare_likelihood_experiment, args=(x, y))
#         p.daemon = True
#         p.start()
#         processes.append(p)
    
#     for p in processes:
#         p.join()



from itertools import product
from multiprocessing import Process, Queue
import numpy as np


def wrapper(x, y, queue):
    # Call your function
    result = compare_likelihood_experiment(x, y)  # returns (pi, pi_hat, pi_hat_hat)
    queue.put((x, result))  # Include x so we can group later

if __name__ == "__main__":
    # Define your input lists
    seed_list = [4, 5, 6, 7, 8, 9, 10]
    num_traj_list = [300_000, 800_000]

    # Create Cartesian product of parameters
    arg_list = list(product(num_traj_list, seed_list))


    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)

    set_num_threads(60 // len(arg_list))
    queue = Queue()
    processes = []

    for x, y in arg_list:
        p = Process(target=wrapper, args=(x, y, queue))
        p.daemon = True
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Gather results
    results_by_traj = {}
    while not queue.empty():
        num_traj, result = queue.get()
        results_by_traj.setdefault(num_traj, []).append(result)

    # Compute means
    for num_traj, results in results_by_traj.items():
        array = np.array(results)  # shape: (num_seeds, 3)
        means = array.mean(axis=0)
        print(f"Results for num_traj = {num_traj}:")
        print(f"  Mean pi_likelihood       = {means[0]:.4f}")
        print(f"  Mean pi_hat_likelihood   = {means[1]:.4f}")
        print(f"  Mean pi_hat_hat_likelihood = {means[2]:.4f}")
