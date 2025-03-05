import time 

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from solvers import solve_milp, solve_greedy_linear, solve_greedy_linear_cvxpy
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

            # r_map = np.reshape(np.array(rewards[:, 0]), (gw.grid_size, gw.grid_size), order='F')
            # r_map = 0
            # env = gridworld.GridWorld(r_map, {},)  # instantiate
            # gridworld environment.  {} indicates that there are no terminal states
            # P_a = env.get_transition_mat()
            # P = [P_a[:,:,0], P_a[:,:,1], P_a[:,:,2], P_a[:,:,3], P_a[:,:,4]]

            # gw.P = P

            reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches) + 1) ### Ensures the switches do not occur at the last and first steps
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

            start_time = time.time()
            r_greedy, nu_greedy, switch_times  = solve_greedy_linear(gw,pi)

            # print(r_milp[:,0,:])
            # print(r_greedy[:,0,:])
            # print(nu_greedy)

            if check_feasibility(gw, pi, r_greedy):
                print("Greedy solution is feasible")
            else:
                print("Greedy solution is infeasible")




            # r = np.zeros((gw.horizon, gw.n_states, gw.n_states))
            # nu = np.zeros((gw.horizon, gw.n_states))
            switch_times_aux = [0] + switch_times
            # print(switch_times)
            # print(len(switch_times), len(rewards_nu_list))

            # for k,  r_nu_tuple in enumerate(rewards_nu_list):
                # r_greedy, nu_greedy = r_nu_tuple
                # print(r_greedy.shape)
                # print(f"Nu values of interval {k}")
                # print(nu_greedy)
            #     r[switch_times_aux[k]:switch_times_aux[k+1],:,:] = r_greedy
            #     nu[switch_times_aux[k]:switch_times_aux[k+1],:] = nu_greedy
            # feasible = check_feasibility(gw, pi, r, nu)
            # print(f"Greedy solution is feasible: {feasible}")

            print(f"Greedy-Linear done in {time.time() - start_time:.2f} seconds")


            print("Optimal switch times found:")
            print("MILP:", [index for index, value in enumerate(z) if value == 1])
            print("Greedy:", switch_times)

            print("Comparing reward values")
            switch_times = [0] + switch_times
            # for k,  r_nu_tuple in enumerate(rewards_nu_list):
                # r_greedy, nu_greedy = r_nu_tuple
                # print(f"MILP Reward for interval {k}")
                # print(r_milp[switch_times[k]:switch_times[k+1]])
                # print(f"Greedy Reward for interval {k}")
                # print(r_greedy)

                # if np.isclose(r_milp[switch_times[k]:switch_times[k+1]], r_greedy):
                #     print(f"Rewards close for interval {k}")
                # else:
                #     print(f"Rewards not close for interval {k}")

        

def check_feasibility(gw, pi, r):
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



    # # Constraints
    # for t in range(T-1):
    #     for s in range(n_states):
    #         for a in range(n_actions):
    #             if not np.isclose(r[t, s, a],np.log(pi[t, s, a]) + nu[t,s] 
    #                               - gamma * np.sum([P[a][s, j] * nu[t+1, j] for j in range(n_states)])):
    #                 return False
    #     print(f"Time {t} is feasible")
    # return True     
                
if __name__ == "__main__":
    test_greedy_linear()
