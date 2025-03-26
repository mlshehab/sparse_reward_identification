import numpy as np
from numba import njit, prange


import time 

import gurobipy as gp
from gurobipy import GRB

from noisy_solvers import solve_milp_noisy, solve_greedy_backward_bisection_noisy#, solve_greedy_backward_bisection_smaller, solve_greedy_backward_alpha
from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation

@njit(parallel=True)
def sample_categorical_parallel(prob_matrix, rand_vals):
    N, K = prob_matrix.shape
    samples = np.empty(N, dtype=np.int32)
    for i in prange(N):
        cum_sum = 0.0
        for k in range(K):
            cum_sum += prob_matrix[i, k]
            if rand_vals[i] < cum_sum:
                samples[i] = k
                break
    return samples

@njit(parallel=True)
def estimate_pi_and_visits_numba(P, pi, H, NUM_TRAJECTORIES):
    A, S, _ = P.shape

    visit_counts = np.zeros((H, S), dtype=np.int32)
    action_counts = np.zeros((H, S, A), dtype=np.int32)

    states = np.empty((NUM_TRAJECTORIES, H + 1), dtype=np.int32)
    actions = np.empty((NUM_TRAJECTORIES, H), dtype=np.int32)

    # Sample initial states from uniform distribution
    rand_init = np.random.rand(NUM_TRAJECTORIES)
    for i in prange(NUM_TRAJECTORIES):
        states[i, 0] = int(rand_init[i] * S)

    for t in range(H):
        s_t = states[:, t]
        action_probs = np.empty((NUM_TRAJECTORIES, A))
        for i in prange(NUM_TRAJECTORIES):
            action_probs[i, :] = pi[t, s_t[i], :]

        rand_actions = np.random.rand(NUM_TRAJECTORIES)
        a_t = sample_categorical_parallel(action_probs, rand_actions)
        actions[:, t] = a_t

        # Use serial loop for safe accumulation (parallelism handled per timestep)
        for i in range(NUM_TRAJECTORIES):
            s = s_t[i]
            a = a_t[i]
            visit_counts[t, s] += 1
            action_counts[t, s, a] += 1

        next_state_probs = np.empty((NUM_TRAJECTORIES, S))
        for i in prange(NUM_TRAJECTORIES):
            next_state_probs[i, :] = P[a_t[i], s_t[i], :]

        rand_next = np.random.rand(NUM_TRAJECTORIES)
        s_next = sample_categorical_parallel(next_state_probs, rand_next)
        states[:, t + 1] = s_next

    return action_counts, visit_counts

NUMBER_OF_EXPERIMENTS = 1
NUM_TRAJECTORIES = 3_000_000
# NUMBER_OF_FEATURES = 7

def check_feasibility(gw, pi, r, nu, b):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    epsilon = 1e-7
    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                if not (r[t, s, a] - epsilon <= np.log(pi[t, s, a]) + b[t, s] + nu[t,s] 
                               - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
                    return False
                elif not (r[t, s, a] + epsilon >= np.log(pi[t, s, a]) - b[t, s] + nu[t,s] 
                                  - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
                    return False
        print(f"Time {t} is feasible")
    
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
            if not (r[T-1, s, a] <= np.log(pi[T-1, s, a]) + b[T-1, s] + nu[T-1, s]):
                return False
            elif not (r[T-1, s, a] >= np.log(pi[T-1, s, a]) - b[T-1, s] + nu[T-1, s]):
                return False
    print(f"Time {T-1} is feasible")
    return True     

def run_problem_1():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    print(f"{NUM_TRAJECTORIES=}")
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 80
    reward = 1
    np.random.seed(1)
    for number_of_switches in [15]:
        for _ in range(NUMBER_OF_EXPERIMENTS):
            gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
            # now obtain time-varying reward maps
            rewards = np.zeros((gw.horizon, gw.n_states)) #array of size Txnum_states
            rewards = rewards.T


            reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches, replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
            print("True reward switch times: ", reward_switch_times)
            reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]
            reward_functions = [np.random.uniform(0,1,(gw.n_states,gw.n_actions)) for _ in range(number_of_switches + 1)]

            reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
            for k in range(number_of_switches + 1):
                for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                    reward[t,:,:] = reward_functions[k]

            V, Q, pi = soft_bellman_operation(gw, reward)

            delta = 1-1e-4

            P = np.asarray(gw.P, dtype=np.float64)     # shape (A, S, S)
            pi = np.asarray(pi, dtype=np.float64)   # shape (H, S, A)
            action_counts, visit_counts = estimate_pi_and_visits_numba(P, pi, gw.horizon, NUM_TRAJECTORIES)

            if (visit_counts == 0).any():
                print("Still there are t,s pairs not visited")

            # Normalize to get pi_hat
            pi_hat = np.zeros_like(pi)/gw.n_actions
            with np.errstate(divide='ignore', invalid='ignore'):
                pi_hat = np.divide(action_counts, visit_counts[:, :, None], where=visit_counts[:, :, None] != 0)

            log_term = np.log(2 / (1 - delta))

            b = np.full((gw.horizon, gw.n_states), 1e3)  # default to inf for unvisited

            # Where visits > 0, compute epsilon and b
            visited_mask = visit_counts > 0
            epsilons = np.zeros_like(visit_counts, dtype=np.float64)
            epsilons[visited_mask] = np.sqrt(1 / (2 * visit_counts[visited_mask]) * log_term)

            # Compute b where valid

            sum_array = pi_hat - epsilons[:, :, np.newaxis]
            # Find the minimal value
            alpha = min(np.min(sum_array), pi_hat.min())

            print(f"{alpha=}, {pi.min()=}, {pi_hat.min()=}")

            b[visited_mask] = epsilons[visited_mask] / alpha


            # epsilons has shape (H, S)
            # We need to compare (H, S, A) arrays, so expand epsilons
            epsilon_broadcast = epsilons[:, :, np.newaxis]  # shape (H, S, 1)

            # Compute fraction of (t, s, a) entries where deviation exceeds epsilon
            violation_fraction = np.sum(np.abs(pi - pi_hat) > epsilon_broadcast) / (gw.horizon * gw.n_states * gw.n_actions)

            print("Violation fraction: ", violation_fraction)

            # start_time = time.time()
            # r_milp, nu_milp, z = solve_milp_noisy(gw, pi_hat, b)
            # print(r_milp.shape, nu_milp.shape)
            # print(f"MILP done in {time.time() - start_time:.2f} seconds")

            # if check_feasibility(gw, pi_hat, r_milp, nu_milp, b):
            #     print("MILP solution is feasible")
            # else:
            #     print("MILP solution is infeasible")


            # print("MILP:", [index for index, value in enumerate(z) if value == 1])

            start_time = time.time()
            r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_noisy(gw, pi_hat, b)

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            if check_feasibility(gw, pi_hat, r_greedy, nu_greedy, b):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")


            print("Optimal switch times found:")
            # print("MILP:", [index for index, value in enumerate(z) if value == 1])
            print("Greedy:", switch_times)


if __name__ == "__main__":
    run_problem_1()
    exit()