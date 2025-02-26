from dynamics import BasicGridWorld
from utils.bellman import soft_bellman_operation

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

    gw.reset(start_state)
    
    trajectory = gw.simulate_trajectory(start_state,pi)
    gw.visualize_trajectory(trajectory)