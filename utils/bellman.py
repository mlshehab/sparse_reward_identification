
import math
import numpy as np
import scipy.special
from scipy.special import logsumexp

def state_only_soft_bellman_operation(env, reward):
    
    # Input:    env    :  environment object
    #           reward :  TxS dimensional vector 
    #           horizon:  finite horizon
        
    discount = env.discount
    horizon = env.horizon
    
    if horizon is None or math.isinf(horizon):
        raise ValueError("Only finite horizon environments are supported")
    
    n_states  = env.n_states
    n_actions = env.n_actions
    
    V = np.zeros(shape = (horizon, n_states))
    Q = np.zeros(shape = (horizon, n_states, n_actions))
        
    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = reward[horizon - 1, :, None]
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
        for a in range(n_actions):
            Ta = env.transition_probability[:,a,:]
            next_values_s_a = Ta @ V[t + 1, :]
            Q[t, :, a] = reward[t, :] + discount * next_values_s_a
            
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def soft_bellman_operation(env, reward):
    
    # Input:    env    :  environment object
    #           reward :  SxA dimensional vector 
    #           horizon:  finite horizon
        
    discount = env.discount
    horizon = env.horizon
    
    if horizon is None or math.isinf(horizon):
        raise ValueError("Only finite horizon environments are supported")
    
    n_states  = env.n_states
    n_actions = env.n_actions
    
#     T = env.transition_matrix
    
    V = np.zeros(shape = (horizon, n_states))
    Q = np.zeros(shape = (horizon, n_states,n_actions))
        
    

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = reward[horizon - 1, :, :]
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
#         next_values_s_a = T @ V[t + 1, :]
#         next_values_s_a = next_values_s_a.reshape(n_states,n_actions)
        for a in range(n_actions):
            # Ta = env.transition_probability[:,a,:]
            Ta = env.P[a]
            next_values_s_a = Ta@V[t + 1, :]
            Q[t, :, a] = reward[t, :, a] + discount * next_values_s_a
            
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi



def time_varying_value_iteration(P_a, rewards, gamma, error=0.01, return_log_policy=False):
  """
  time-varying soft value iteration function (to ensure that the policy is differentiable)
  
  inputs:
    P_a         N_STATESxN_STATESxN_ACTIONS transition probabilities matrix - 
                              P_a[s0, s1, a] is the transition prob of 
                              landing at state s1 when taking action 
                              a at state s0
    rewards     T X N_STATES matrix - rewards for all the states
    gamma       float - RL discount
    error       float - threshold for a stop

  returns:
    values    T X N_STATES matrix - estimated values
    policy    T X N_STATES x N_ACTIONS matrix - policy
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)
  # no. of time steps
  T = rewards.shape[0]

  values = np.zeros([T, N_STATES])
  q_values = np.zeros([T, N_STATES, N_ACTIONS])

  # estimate values and q-values iteratively
  while True:
    values_tmp = values.copy()
    q_values = np.stack([rewards + sum([gamma* (np.outer(P_a[:, s1, a], values_tmp[:,s1])).T
                                                for s1 in range(N_STATES)]) for a in range(N_ACTIONS)])
    q_values = np.transpose(q_values, (1, 2, 0))
    assert q_values.shape[0]==T and q_values.shape[1]==N_STATES, "q-values don't have the appropriate dimensions"
    assert q_values.shape[2]==N_ACTIONS, "q-values don't have the appropriate dimensions"
    values = logsumexp(q_values, axis=2)
    if max([abs(values[t,s] - values_tmp[t,s]) for s in range(N_STATES) for t in range(T)]) < error:
      break

  # generate policy
  log_policy = q_values - values[:,:,np.newaxis]
  policy = np.exp(log_policy)

  if return_log_policy:
    return values, policy, log_policy
  else:
    return values, policy