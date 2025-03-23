import time 

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from noisy_solvers import solve_milp_noisy, solve_greedy_backward_bisection_noisy#, solve_greedy_backward_bisection_smaller, solve_greedy_backward_alpha
from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation


NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_FEATURES = 7

def test_greedy():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 4
    wind = 0.1
    discount = 0.9
    horizon = 30
    reward = 1
    np.random.seed(1)
    for number_of_switches in [2]:
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


            ## Sample from pi to obtain demonstration
            pi_hat = np.zeros_like(pi)
            alpha = pi.min()
            print(f"{alpha=}")
            print(f"{gw.n_actions=}")
            delta = 1-1e-3
            NUM_SAMPLES = 10000
            for t in range(gw.horizon):
                for s in range(gw.n_states):
                    # Sample NUM_SAMPLES actions from the distribution pi[t, s, :]
                    actions = np.random.choice(gw.n_actions, size=NUM_SAMPLES, p=pi[t, s, :])
                    
                    # Count occurrences of each action
                    counts = np.bincount(actions, minlength=gw.n_actions)
                    
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



            start_time = time.time()
            r_milp, nu_milp, z = solve_milp_noisy(gw, pi_hat, b)
            print(r_milp.shape, nu_milp.shape)
            print(f"MILP done in {time.time() - start_time:.2f} seconds")

            if check_feasibility(gw, pi_hat, r_milp, nu_milp, b):
                print("MILP solution is feasible")
            else:
                print("MILP solution is infeasible")


            print("MILP:", [index for index, value in enumerate(z) if value == 1])

            start_time = time.time()
            # r_greedy, nu_greedy, switch_times  = solve_greedy_backward(gw,pi)
            r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_noisy(gw, pi_hat, b)

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            if check_feasibility(gw, pi_hat, r_greedy, nu_greedy, b):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")
                print("Checking feasibility of rewards only")
                if check_feasibility_reward(gw, pi_hat, r_greedy, b):
                    print("Greedy solution is REWARD feasible")
                else:
                    print("Greedy solution is not even REWARD feasible")



            print("Optimal switch times found:")
            # print("MILP:", [index for index, value in enumerate(z) if value == 1])
            print("Greedy:", switch_times)

def test_greedy_alpha():
    # np.random.seed(1)
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 20
    reward = 1
    start_state = 10

    for number_of_switches in [2,5]:
        for _ in range(NUMBER_OF_EXPERIMENTS):
            gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
            # now obtain time-varying reward maps
            rewards = np.zeros((gw.horizon, gw.n_states)) #array of size Txnum_states
            rewards = rewards.T
            

            reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches,  replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
            print("True reward switch times: ", reward_switch_times)
            reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]

            U = []
            for i in range(NUMBER_OF_FEATURES):
                U.append(np.random.uniform(0,1,(gw.n_states,gw.n_actions)))
            alpha_t = [np.random.uniform(0,1,NUMBER_OF_FEATURES) for _ in range(number_of_switches + 1)]
            # print(alpha_t)
            # print(U)

            reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
            for k in range(number_of_switches + 1):
                for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                    reward[t,:,:] = sum([alpha_t[k][j]*U[j] for j in range(NUMBER_OF_FEATURES)])
            
            # print(reward)

            V, Q, pi = soft_bellman_operation(gw, reward)

            start_time = time.time()
            r_greedy, alpha_greedy, nu_greedy, switch_times  = solve_greedy_backward_alpha(gw,pi,U)
            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")

            if check_feasibility(gw, pi, r_greedy, nu_greedy):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")

            print("Optimal switch times found:")
            print("Greedy:", switch_times)

            print("True alpha values :", alpha_t)
            print("Computed alpha values:", [alpha_greedy[i] for i in [0]+switch_times])


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
                
if __name__ == "__main__":
    test_greedy()

    # test_greedy_alpha()
