from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
# from solvers import solve_milp
from utils.bellman import state_only_soft_bellman_operation, soft_bellman_operation, time_varying_value_iteration
import numpy as np

if __name__ == "__main__":
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    start_state = 10

    gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
    

    P_a = np.array(gw.P).reshape(gw.n_states, gw.n_states, gw.n_actions)

    # print(P.shape)

    


    reward_state_only = np.zeros(shape = (gw.horizon, gw.n_states))
    for t in range(int(gw.horizon/2)):
        reward_state_only[t, 14] = 10.0 # 14 is the bottom middle of the grid
    for t in range(int(gw.horizon/2), gw.horizon):
        reward_state_only[t, 0] = 10.0

    V_state_only, Q_state_only, pi_state_only = state_only_soft_bellman_operation(gw, reward_state_only)

    valuem, policy = time_varying_value_iteration(P_a, reward_state_only, discount, error=0.01, return_log_policy=False)


    # Calculate the norm difference between the two policies at each time step
    norm_diff = np.zeros(gw.horizon)
    for t in range(gw.horizon):
        norm_diff[t] = np.linalg.norm(pi_state_only[t] - policy[t])
    
    print("Norm difference between the two policies at each time step:")
    print(norm_diff)




    # reward = np.zeros(shape = (gw.horizon, gw.n_states, gw.n_actions))

    # for t in range(int(gw.horizon/2)):
    #     reward[t, 14, :] = 10.0 # 14 is the bottom middle of the grid
    # for t in range(int(gw.horizon/2), gw.horizon):
    #     reward[t, 0, :] = 10.0
    
    # V,Q,pi = soft_bellman_operation(gw, reward)

    # norm_diff = np.zeros(gw.horizon)
    # for t in range(gw.horizon):
    #     norm_diff[t] = np.linalg.norm(pi[t] - pi_state_only[t])
    
    # print("Norm difference between the two policies at each time step:")
    # print(norm_diff)