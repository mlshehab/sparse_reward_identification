import numpy as np
from numba import njit, prange, set_num_threads

import os
import time 

import gurobipy as gp
from gurobipy import GRB

from solvers import solve_greedy_backward_bisection
from noisy_solvers import solve_milp_noisy, solve_greedy_backward_bisection_noisy, solve_greedy_backward_bisection_smaller_noisy#, solve_greedy_backward_alpha
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
# NUMBER_OF_FEATURES = 7

def check_feasibility_reward(gw, pi, r, b):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    model = gp.Model("MILP")
    model.setParam('OutputFlag', False)
    # Decision variables
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")

    # Objective: Minimize sum of z_t
    model.setObjective(0, GRB.MINIMIZE)

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                # model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
                model.addConstr(nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)) <= r[t, s, a]-np.log(pi[t, s, a]) + b[t,s], name=f"r_def_{t}_{s}_{a}")
                model.addConstr(nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)) >= r[t, s, a]-np.log(pi[t, s, a]) - b[t,s], name=f"r_def_{t}_{s}_{a}")
    
    # Solve the model
    model.optimize()

    # Return results
    return model.status == GRB.OPTIMAL

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

def run_problem_1(num_trajectories, seed):
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    output_path = "results/problem1_new_rewards/"
    print(f"{num_trajectories=}")
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    np.random.seed(1)
    for number_of_switches in [5]:
        for _ in range(NUMBER_OF_EXPERIMENTS):
            reward_path = f"data/rewards/reward_{seed}_{number_of_switches}.npy"

            gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
            # now obtain time-varying reward maps
            if os.path.exists(reward_path):
                reward = np.load(reward_path)
            else:
                generate_and_save_rewards(seed)
                reward = np.load(reward_path)

            V, Q, pi = soft_bellman_operation(gw, reward)

            delta = 1-1e-4

            P = np.asarray(gw.P, dtype=np.float64)     # shape (A, S, S)
            pi = np.asarray(pi, dtype=np.float64)   # shape (H, S, A)
            action_counts, visit_counts = estimate_pi_and_visits_numba(P, pi, gw.horizon, num_trajectories)

            if (visit_counts == 0).any():
                print("Still there are t,s pairs not visited")

            # Normalize to get pi_hat
            pi_hat = np.zeros_like(pi)/gw.n_actions
            with np.errstate(divide='ignore', invalid='ignore'):
                pi_hat = np.divide(action_counts, visit_counts[:, :, None], where=visit_counts[:, :, None] != 0)

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

            # Compute fraction of (t, s, a) entries where deviation exceeds epsilon
            violation_fraction = np.sum(np.abs(pi - pi_hat) > epsilon_broadcast) / (gw.horizon * gw.n_states * gw.n_actions)

            print("Violation fraction: ", violation_fraction)
            print(f"{epsilons.max()=}, {epsilons.min()=}")


            start_time = time.time()
            if num_trajectories == 1:
                r_greedy, nu_greedy, switch_times = solve_greedy_backward_bisection(gw, pi)
            else:
                r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_smaller_noisy(gw, pi_hat, b)

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")
            # print("True switch times: ", )
            print("Greedy:", switch_times)

            if num_trajectories == 1:
                assert check_feasibility(gw, pi, r_greedy, nu_greedy, np.zeros_like(b))
                    # print("Greedy solution is feasible")
                # else:
                    # print("Solution is infeasible")
            else:
                assert check_feasibility(gw, pi_hat, r_greedy, nu_greedy, b)
                #     print("Greedy solution is feasible")
                # else:
                #     print("Greedy solution is infeasible")
                #     print("Checking feasibility of rewards only")
                #     if check_feasibility_reward(gw, pi_hat, r_greedy, b):
                #         print("Greedy solution is REWARD feasible")
                #     else:
                #         print("Greedy solution is not even REWARD feasible")



            np.save(output_path + f"{seed=}, {num_trajectories=}_switches.npy", np.array(switch_times))
            np.save(output_path + f"{seed=}, {num_trajectories=}_rewards.npy", np.array(r_greedy))
            np.save(output_path + f"{seed=}, {num_trajectories=}_values.npy", np.array(nu_greedy))


def generate_and_save_rewards(seed):
        import random
        grid_size = 5
        wind = 0.1
        discount = 0.9
        horizon = 50
        reward = 1

        number_of_switches = 5
        min_switch_mag = 0.1
        max_switch_mag = 0.4
        magnitude_by_switch = (max_switch_mag-min_switch_mag)/number_of_switches


        np.random.seed(seed)
        gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
        # now obtain time-varying reward maps

        reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches, replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
        print("True reward switch times: ", reward_switch_times)
        reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]
        reward_functions = [np.random.uniform(0,1,(gw.n_states,gw.n_actions))]
        switch_magnitudes = [min_switch_mag + i*magnitude_by_switch for i in range(number_of_switches)]

        switch_magnitudes = random.sample(switch_magnitudes, len(switch_magnitudes))

        for i in range(number_of_switches):
            reward_functions += [reward_functions[i] + np.random.uniform(0,switch_magnitudes[i],(gw.n_states,gw.n_actions))]

        reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
        for k in range(number_of_switches + 1):
            for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                reward[t,:,:] = reward_functions[k]

        np.save(f"data/rewards/reward_{seed}_{number_of_switches}.npy", reward)
        np.save(f"data/rewards/switch_{seed}_{number_of_switches}.npy", np.array(reward_switch_times))
            


if __name__ == "__main__":

    from multiprocessing import Process
    from itertools import product

    # Define your input lists
    seed_list = [4,5,6,7,8,9,10]
    num_traj_list = [200_000, 400_000, 800_000, 1_000_000, 2_000_000, 4_000_000, 8_000_000]
    # num_traj_list = [12_000_000, 16_000_000, 24_000_000]#, 8_000_000]
    # num_traj_list = [120_000_000, 240_000_000]#[36_000_000, 48_000_000, 60_000_000]#, 8_000_000]

    for seed in seed_list:
        generate_and_save_rewards(seed)

    # Create Cartesian product of parameters
    arg_list = list(product(num_traj_list, seed_list))

    # Launch each in its own process
    set_num_threads(60//len(arg_list))
    processes = []
    for x, y in arg_list:
        p = Process(target=run_problem_1, args=(x, y))
        p.daemon = True
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

    # run_problem_1(2_000_000, 2)