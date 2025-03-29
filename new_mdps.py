import math
import numpy as np
import scipy.special
import cvxpy as cp
import matplotlib.pyplot as plt

inf  = float("inf")


class SlipperyGridWorld(object):
    """
    Gridworld MDP.
    """

    def __init__(self, grid_size, wind, discount,horizon, reward, slippery):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Gridworld
        """
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))  # Added (0, 0) for stay action
        self.action_dict = {"down": 0, "right": 1, "up": 2, "left": 3, "stay": 4}

        self.n_actions = len(self.actions)
        self.n_states = grid_size**2
        self.grid_size = grid_size
        self.wind = wind
        self.wind_buffer = wind
        
        self.slippery = slippery

        self.discount = discount
        self.horizon = horizon
    
        self.reward = reward

        # # Preconstruct the transition probability array.
        # self.transition_probability = np.array(
        #     [[[self._transition_probability(i, j, k)
        #        for k in range(self.n_states)]
        #       for j in range(self.n_actions)]
        #      for i in range(self.n_states)])
        


        # self.normalize_transition_matrices()
        
        self.P = []
        for a in range(self.n_actions):
            matrix = np.random.rand(self.n_states, self.n_states)  # Random values in [0, 1)
            matrix /= matrix.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
            Pa = matrix
            self.P.append(Pa)

        self.reward = reward

  
    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        """
        Get whether two points neighbour each other. Also returns true if they
        are the same point.

        i: (x, y) int tuple.
        k: (x, y) int tuple.
        -> bool.
        """

        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1
    
    def reward(self, state_int):
        """
        Reward for being in state state_int.

        state_int: State integer. int.
        -> Reward.
        """

        if state_int == self.n_states - 1:
            return 1
        return 0
        
    def _transition_probability(self, i, j, k):
        """
        Get the probability of transitioning from state i to state k given
        action j.

        i: State int.
        j: Action int.
        k: State int.
        -> p(s_k | s_i, a_j)
        """

        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
            

        if not self.neighbouring((xi, yi), (xk, yk)):
            return 0.0
        
        # Is i center point and we are in slippery MDP? Then, all transitions have 1/5 probability
        if self.slippery and yi == self.grid_size//2:
            # Is k the intended state to move to?
            if (xi + xj, yi + yj) == (xk, yk):
                return 0
            else:
                return 1/(self.n_actions - 1)

        # Is k the intended state to move to?
        if (xi + xj, yi + yj) == (xk, yk):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if (xi not in {0, self.grid_size-1} and
                yi not in {0, self.grid_size-1}):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions
            
    def normalize_transition_matrices(self):
        for a in range(self.n_actions):
            P = self.transition_probability[:,a,:]
            sum_P = P.sum(axis = 1)
            normalized_P = P/sum_P[:,None]
            self.transition_probability[:,a,:] = normalized_P
            
    
    def make_state_determinstic(self, s):
        
        for s_prime in range(self.n_states):
            for a in range(self.n_actions):
                self.transition_probability[s,a,s_prime] = 0.0
                
        for a in range(self.n_actions):
            self.transition_probability[s,a,s-1] = 1.0
         
            
    def construct_Xi(self, pi):
        horizon, n_states, n_actions = pi.shape
    
        Xi = np.zeros((horizon*n_states*n_actions,1))
        
        for t in range(horizon):
            curr_pi = pi[t].flatten('F')[:,None]
    #         print(curr_pi.shape)
            # next_pi = pi[t+1].flatten('F')[:,None]
            Xi[t*n_states*n_actions:(t+1)*n_states*n_actions] = np.log(curr_pi) 
        return Xi    
    
    def prune_Gamma_and_Xi(self, pi):
        Reach, UnReach = self.get_reachable_tube()
        n_unreach_states = []
        total_unreach_states = set(UnReach[str(0)])

        Gamma = self.construct_Gamma()
        # print(Gamma.shape)
        Xi = self.construct_Xi(pi)

        remove_indices = []

        for t in range(self.horizon):
            UnReach_t = UnReach[str(t)]
            if t > 1:
                total_unreach_states = total_unreach_states.intersection(set(UnReach_t))
                n_unreach_states.append(len(total_unreach_states))
            # print(UnReach_t)
            for unreachable_state in UnReach_t:
                for a in range(self.n_actions):
                    remove_indices.append(  t*self.n_states*self.n_actions + (self.n_states*a + unreachable_state)) 

        # print(remove_indices)
        # print("len:", len(remove_indices))
        Gamma = np.delete(Gamma,obj = remove_indices, axis = 0)
        Xi = np.delete(Xi, obj = remove_indices, axis = 0 )                
        return Gamma, Xi , n_unreach_states


    def compute_projected_kernel(self,Gamma):
        K = scipy.linalg.null_space(Gamma)
        projected_K = K[:self.n_states*self.n_actions,:]
        # print("The shape of projeted matrix is: ", projected_K.shape)
        # print("The rank of projected K is: ", np.linalg.matrix_rank(projected_K))    
        return np.linalg.matrix_rank(projected_K)

    def reset(self,start_state):
        self.state = start_state

    def step(self, action):
        """
        Take a step in the environment.
        action: Action int.
        -> Next state int.
        """
        state = self.state
        probabilities = self.transition_probability[state, action]
        next_state = np.random.choice(self.n_states, p=probabilities)
        return next_state
    

    def simulate_trajectory(self, start_state, policy):
        """
        Simulate a trajectory in the environment.

        start_state: Starting state int.
        policy: Policy to follow (array of shape [horizon, n_states, n_actions]).
        -> List of states visited.
        """
        state = start_state
        trajectory = [state]
        action_dict_reverse = {v: k for k, v in self.action_dict.items()}
        for t in range(self.horizon):
            action = np.random.choice(self.n_actions, p=policy[t, state])
            next_state = self.step(action)
            print(f"Time: {t}, State: {state}, Action: {action_dict_reverse[action]}, Next State: {next_state}")
            state = next_state
            trajectory.append(state)
            # if state == self.n_states - 1:  # Goal state
            #     break
            self.state = state
        return trajectory

    def visualize_trajectory(self, trajectory):
        """
        Visualize a trajectory on the gridworld.

        trajectory: List of states visited.
        """
        grid = np.zeros((self.grid_size, self.grid_size))
        path_x, path_y = [], []
        for state in trajectory:
            x, y = self.int_to_point(state)
            path_x.append(y)  # Note the swap: x is now y
            path_y.append(x)  # Note the swap: y is now x

        # Create a custom colormap that goes from red to orange to blue to purple
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = ['red', 'orange', 'blue', 'purple']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors_list, N=256)  # Use 256 for smoother gradient
        norm = plt.Normalize(0, len(trajectory) - 1)
        colors = cmap(norm(np.linspace(0, len(trajectory) - 1, len(trajectory))))

        for i in range(len(path_x) - 1):
            plt.plot(path_x[i:i+2], path_y[i:i+2], marker='o', color=colors[i])

        plt.xlim(-0.5, self.grid_size - 0.5)
        plt.ylim(-0.5, self.grid_size - 0.5)
        plt.gca().invert_yaxis()
        plt.grid(True)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Time Step')
        plt.show()
