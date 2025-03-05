import time 

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solvers import solve_milp, solve_greedy_backward, solve_greedy_linear_cvxpy
from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation

# from /dynamic_irl.src.envs  import  gridworld

NUMBER_OF_EXPERIMENTS = 1

def test_greedy_linear():
    '''
    This function compares the solutions found by Greedy-Linear to MILP over some randomly generated MDPs
    '''
    grid_size = 3#5
    wind = 0.1
    discount = 0.9
    horizon = 10#50
    reward = 1
    start_state = 10

    for number_of_switches in [2]:
        for _ in range(NUMBER_OF_EXPERIMENTS):
            gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
            # now obtain time-varying reward maps
            rewards = np.zeros((gw.horizon, gw.n_states)) #array of size Txnum_states
            rewards = rewards.T


            reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches) + 1) ### Ensures the switches do not occur at the last and first steps
            print("True reward switch times: ", reward_switch_times)
            reward_switch_intervals = [0] + reward_switch_times + [gw.horizon-1]
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
            r_greedy, nu_greedy, switch_times  = solve_greedy_backward(gw,pi)
            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            if check_feasibility(gw, pi, r_greedy, nu_greedy):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")


            print("Optimal switch times found:")
            print("MILP:", [index for index, value in enumerate(z) if value == 1])
            print("Greedy:", switch_times)

            print("Comparing reward values")

            # print(r_greedy)
            # print(r_milp)
        

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
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
                # model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
    
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
    test_greedy_linear()
