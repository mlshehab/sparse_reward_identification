import numpy as np
from dynamics import BasicGridWorld

def generate_and_save_rewards_incremental(seed, horizon, number_of_switches):
        import random
        grid_size = 5
        wind = 0.1
        discount = 0.9
        # horizon = 50

        # number_of_switches = 5
        min_switch_mag = 0.1
        max_switch_mag = 0.4
        magnitude_by_switch = (max_switch_mag-min_switch_mag)/number_of_switches


        np.random.seed(seed)
        gw = BasicGridWorld(grid_size, wind, discount, horizon, None)
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


def generate_and_save_rewards_uniform(seed, horizon, number_of_switches):
        import random
        grid_size = 5
        wind = 0.1
        discount = 0.9


        np.random.seed(seed)
        gw = BasicGridWorld(grid_size, wind, discount, horizon, None)
        # now obtain time-varying reward maps

        reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches, replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
        print("True reward switch times: ", reward_switch_times)
        reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]
        reward_functions = [np.random.uniform(0,1,(gw.n_states,gw.n_actions)) for _ in range(number_of_switches + 1)]

        reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
        for k in range(number_of_switches + 1):
            for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                reward[t,:,:] = reward_functions[k]

        np.save(f"data/rewards/reward_{seed}_{number_of_switches}.npy", reward)
        np.save(f"data/rewards/switch_{seed}_{number_of_switches}.npy", np.array(reward_switch_times))


def generate_and_save_rewards_skewed(seed, horizon, number_of_switches):
        grid_size = 5
        wind = 0.1
        discount = 0.9


        np.random.seed(seed)
        gw = BasicGridWorld(grid_size, wind, discount, horizon, None)
        # now obtain time-varying reward maps

        reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches, replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
        print("True reward switch times: ", reward_switch_times)
        reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]
        reward_functions = [sparse_skewed_uniform_matrix((gw.n_states,gw.n_actions), ) for _ in range(number_of_switches + 1)]

        reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
        for k in range(number_of_switches + 1):
            for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                reward[t,:,:] = reward_functions[k]

        np.save(f"data/rewards/reward_skewed_{seed}_{number_of_switches}.npy", reward)
        np.save(f"data/rewards/switch_skewed_{seed}_{number_of_switches}.npy", np.array(reward_switch_times))


def sparse_skewed_uniform_matrix(shape, sparsity=0.9, high_fraction=0.1):
    """
    Generate a sparse matrix with mostly small values and a few large ones.

    Parameters:
        shape: tuple (rows, cols)
        sparsity: fraction of total entries that are zero
        high_fraction: fraction of non-zero entries that should be high

    Returns:
        NumPy array of shape `shape` with the desired sparsity and value pattern
    """
    total_entries = shape[0] * shape[1]
    num_non_zero = int((1 - sparsity) * total_entries)
    num_high = int(high_fraction * num_non_zero)
    num_low = num_non_zero - num_high

    # Create an array of zeros
    matrix = np.zeros(total_entries)

    # Assign small values (e.g., U(0, 0.1))
    matrix[:num_low] = np.random.uniform(0.0, 0.1, num_low)

    # Assign high values (e.g., U(0.8, 1.0))
    matrix[num_low:num_low + num_high] = np.random.uniform(2., 2.5, num_high)

    # Shuffle non-zero values across the matrix
    np.random.shuffle(matrix)

    # Reshape into matrix
    matrix = matrix.reshape(shape)

    return matrix


def generate_and_save_rewards_problem3_like(seed, horizon, number_of_switches):
        HOME_STATE = 0
        WATER_STATE = 14
        # select number of maps
        n_features = 2 

        grid_size = 5
        wind = 0.1
        discount = 0.9
        gw = BasicGridWorld(grid_size, wind, discount, horizon, None)

        U = np.zeros(shape=(gw.n_states, gw.n_actions, n_features))

        REWARD_SCALE = 2.
        U[HOME_STATE, :,0] = 1.0* REWARD_SCALE
        U[WATER_STATE, :,  1] = 1.0* REWARD_SCALE



        np.random.seed(seed)

        # now obtain time-varying reward maps

        reward_switch_times = sorted(np.random.choice(gw.horizon-3, number_of_switches, replace=False) + 1) ### Ensures the switches do not occur at the last and first steps
        print("True reward switch times: ", reward_switch_times)
        reward_switch_intervals = [0] + reward_switch_times + [gw.horizon]
        weights_1 = [np.random.uniform(0,1) for _ in range(number_of_switches + 1)]
        # reward_functions = [sparse_skewed_uniform_matrix((gw.n_states,gw.n_actions), ) for _ in range(number_of_switches + 1)]

        reward = np.zeros(shape=(gw.horizon, gw.n_states, gw.n_actions))
        for k in range(number_of_switches + 1):
            for t in range(reward_switch_intervals[k], reward_switch_intervals[k+1]):
                reward[t,:,:] = weights_1[k]*U[:, :, 0] + (1-weights_1[k])*U[:, :, 1]

        np.save(f"data/rewards/reward_lowrank_{seed}_{number_of_switches}.npy", reward)
        np.save(f"data/rewards/switch_lowrank_{seed}_{number_of_switches}.npy", np.array(reward_switch_times))