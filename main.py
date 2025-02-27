from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation
from solvers import solve_milp, solve_L_1, solve_L_inf
import numpy as np

if __name__ == "__main__":
    grid_size = 5
    wind = 0.1
    discount = 0.9
    horizon = 50
    reward = 1
    start_state = 10

    gw = BasicGridWorld(grid_size, wind, discount, horizon, reward)
    
    reward = np.zeros(shape = (gw.horizon, gw.n_states, gw.n_actions))
    for t in range(int(gw.horizon/2)):
        reward[t, 14, :] = 10.0 # 14 is the bottom middle of the grid
    for t in range(int(gw.horizon/2), gw.horizon):
        reward[t, 0, :] = 10.0

    V,Q,pi = soft_bellman_operation(gw, reward)

    import time

    start_time = time.time()
    r_MILP, nu_MILP, z_MILP = solve_milp(gw,pi)
    print(f"MILP done in {time.time() - start_time:.2f} seconds")


    start_time = time.time()
    r_L_1, nu_L_1 = solve_L_1(gw,pi)
    print(f"L1 done in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    r_L_inf, nu_L_inf = solve_L_inf(gw,pi)
    print(f"Linf done in {time.time() - start_time:.2f} seconds")
    
    import matplotlib.pyplot as plt

    reward_MILP = r_MILP[:, 0, :]
    reward_L_1 = r_L_1[:, 0, :]
    reward_L_inf = r_L_inf[:, 0, :]

    # Calculate the average rewards over actions for state 0
    reward_MILP_avg = np.mean(reward_MILP, axis=1)
    reward_L_1_avg = np.mean(reward_L_1, axis=1)
    reward_L_inf_avg = np.mean(reward_L_inf, axis=1)

    # Plot the average reward over time
    plt.figure(figsize=(10, 6))
    plt.plot(range(gw.horizon), reward_MILP_avg, label='MILP')
    plt.plot(range(gw.horizon), reward_L_1_avg, label='L1')
    plt.plot(range(gw.horizon), reward_L_inf_avg, label='Linf')
    plt.xlabel('Time')
    plt.ylabel('Average Reward at State 0')
    plt.title('Average Reward at State 0 over Time for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.show()