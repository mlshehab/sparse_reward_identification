import time 

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solvers import * #solve_milp, solve_greedy_backward, solve_greedy_backward_bisection, solve_greedy_backward_bisection_smaller, solve_greedy_backward_alpha
from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation

# from /dynamic_irl.src.envs  import  gridworld

NUMBER_OF_EXPERIMENTS = 1
NUMBER_OF_FEATURES = 7

def test_greedy():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 4
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    start_state = 10

    for number_of_switches in [2,5]:
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

            start_time = time.time()
            r_milp, nu_milp, z = solve_milp(gw, pi)
            print(r_milp.shape, nu_milp.shape)
            print(f"MILP done in {time.time() - start_time:.2f} seconds")

            if check_feasibility(gw, pi, r_milp, nu_milp):
                print("MILP solution is feasible")
            else:
                print("MILP solution is infeasible")


            print("MILP:", [index for index, value in enumerate(z) if value == 1])

            start_time = time.time()
            # r_greedy, nu_greedy, switch_times  = solve_greedy_backward(gw,pi)
            r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_smaller(gw,pi)

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            if check_feasibility(gw, pi, r_greedy, nu_greedy):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")
                print("Checking feasibility of rewards only")
                if check_feasibility_reward(gw, pi, r_greedy):
                    print("Greedy solution is REWARD feasible")
                else:
                    print("Greedy solution is not even REWARD feasible")



            print("Optimal switch times found:")
            print("MILP:", [index for index, value in enumerate(z) if value == 1])
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


def test_equivalent_switch():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 3
    wind = 0.1
    discount = 0.9
    horizon = 10
    reward = 1

    for number_of_switches in [2,5]:
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
                    reward[t,:,:] = reward_functions[0] + k


            print(reward[:, 0, 0])
            print(reward)

            V, Q, pi = soft_bellman_operation(gw, reward)

            start_time = time.time()
            r_milp, nu_milp, z = solve_milp(gw, pi)
            print(r_milp.shape, nu_milp.shape)
            print(f"MILP done in {time.time() - start_time:.2f} seconds")

            if check_feasibility(gw, pi, r_milp, nu_milp):
                print("MILP solution is feasible")
            else:
                print("MILP solution is infeasible")


            print("MILP:", [index for index, value in enumerate(z) if value == 1])

            start_time = time.time()
            # r_greedy, nu_greedy, switch_times  = solve_greedy_backward(gw,pi)
            r_greedy, nu_greedy, switch_times  = solve_greedy_backward_bisection_smaller(gw,pi)

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            if check_feasibility(gw, pi, r_greedy, nu_greedy):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")
                print("Checking feasibility of rewards only")
                if check_feasibility_reward(gw, pi, r_greedy):
                    print("Greedy solution is REWARD feasible")
                else:
                    print("Greedy solution is not even REWARD feasible")



            print("Optimal switch times found:")
            print("MILP:", [index for index, value in enumerate(z) if value == 1])
            print("Greedy:", switch_times)

def compare_intra_interval_optimization():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 4
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1

    for number_of_switches in [4]:
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

            # start_time = time.time()
            # r_milp, nu_milp, z = solve_milp(gw, pi)
            # print(r_milp.shape, nu_milp.shape)
            # print(f"MILP done in {time.time() - start_time:.2f} seconds")

            # if check_feasibility(gw, pi, r_milp, nu_milp):
            #     print("MILP solution is feasible")
            # else:
            #     print("MILP solution is infeasible")


            # print("MILP:", [index for index, value in enumerate(z) if value == 1])

            start_time = time.time()
            # r_greedy, nu_greedy, switch_times  = solve_greedy_backward(gw,pi)
            r_greedy_by_interval, nu_greedy_by_interval, switch_times  = solve_greedy_backward_bisection_L1_by_interval(gw,pi)
            
            print(r_greedy_by_interval.shape)

            r_greedy_overall, nu_greedy_overall, switch_times  = solve_greedy_backward_bisection_L1_overall(gw,pi)

            print(f"By interval L1 norm {np.linalg.norm(r_greedy_by_interval.flatten(), ord=1)}")
            print(f"Overall L1 norm {np.linalg.norm(r_greedy_overall.flatten(), ord=1)}")
            # print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")

            if all(np.isclose(r_greedy_by_interval.flatten(), r_greedy_overall.flatten())):
                print("Rewards are very close")
            else:
                print("Rewards are not close enough")

            if check_feasibility(gw, pi, r_greedy_by_interval, nu_greedy_by_interval):
                print("Greedy solution by interval is feasible")
            else:
                print("Greedy solution by interval is infeasible")

            if check_feasibility(gw, pi, r_greedy_overall, nu_greedy_overall):
                print("Greedy solution overall is feasible")

            else:
                print("Greedy solution overall is infeasible")

            # else:
            #     print("Greedy solution is infeasible")
            #     print("Checking feasibility of rewards only")
            #     if check_feasibility_reward(gw, pi, r_greedy):
            #         print("Greedy solution is REWARD feasible")
            #     else:
            #         print("Greedy solution is not even REWARD feasible")



            # print("Optimal switch times found:")
            # print("MILP:", [index for index, value in enumerate(z) if value == 1])
            # print("Greedy:", switch_times)

def check_feasibility_reward(gw, pi, r):
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
                model.addConstr(nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)) == r[t, s, a]-np.log(pi[t, s, a]), name=f"r_def_{t}_{s}_{a}")
    
    # Solve the model
    model.optimize()

    # Return results
    return model.status == GRB.OPTIMAL


def check_feasibility(gw, pi, r, nu):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                if not np.isclose(r[t, s, a],np.log(pi[t, s, a]) + nu[t,s] 
                                  - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
                    return False
        print(f"Time {t} is feasible")
    
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
            if not np.isclose(r[T-1, s, a], np.log(pi[T-1, s, a]) + nu[T-1, s]):
                return False
    print(f"Time {T-1} is feasible")
    return True     
                
if __name__ == "__main__":
    # test_greedy()
    compare_intra_interval_optimization()
    # test_equivalent_switch()
    # test_greedy_alpha()
