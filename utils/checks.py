import numpy as np

def is_markovian(P_a):
        """
        Check if the transition probability matrix P_a is Markovian for each action.
        
        P_a: N_STATES x N_STATES x N_ACTIONS transition probabilities matrix
        
        Returns:
        markovian: Boolean indicating if P_a is Markovian for each action
        """
        N_STATES, _, N_ACTIONS = P_a.shape
        for a in range(N_ACTIONS):
            for s in range(N_STATES):
                if not np.isclose(np.sum(P_a[s, :, a]), 1.0):
                    return False
        return True