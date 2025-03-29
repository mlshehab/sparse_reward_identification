import numpy as np
from numba import njit, prange, set_num_threads


@njit(parallel=True)
def sample_categorical_parallel(prob_matrix, rand_vals):
    N, K = prob_matrix.shape
    samples = np.empty(N, dtype=np.int32)
    for i in prange(N):
        cum_sum = 0.0
        for k in range(K):
            cum_sum += prob_matrix[i, k]
            if rand_vals[i] < cum_sum:
                samples[i] = k
                break   
    return samples

@njit(parallel=True)
def estimate_pi_and_visits_numba(P, pi, H, NUM_TRAJECTORIES):
    A, S, _ = P.shape

    visit_counts = np.zeros((H, S), dtype=np.int32)
    action_counts = np.zeros((H, S, A), dtype=np.int32)

    states = np.empty((NUM_TRAJECTORIES, H + 1), dtype=np.int32)
    actions = np.empty((NUM_TRAJECTORIES, H), dtype=np.int32)

    # Sample initial states from uniform distribution
    rand_init = np.random.rand(NUM_TRAJECTORIES)
    for i in prange(NUM_TRAJECTORIES):
        states[i, 0] = int(rand_init[i] * S)

    for t in range(H):
        s_t = states[:, t]
        action_probs = np.empty((NUM_TRAJECTORIES, A))
        for i in prange(NUM_TRAJECTORIES):
            action_probs[i, :] = pi[t, s_t[i], :]

        rand_actions = np.random.rand(NUM_TRAJECTORIES)
        a_t = sample_categorical_parallel(action_probs, rand_actions)
        actions[:, t] = a_t

        # Use serial loop for safe accumulation (parallelism handled per timestep)
        for i in range(NUM_TRAJECTORIES):
            s = s_t[i]
            a = a_t[i]
            visit_counts[t, s] += 1
            action_counts[t, s, a] += 1

        next_state_probs = np.empty((NUM_TRAJECTORIES, S))
        for i in prange(NUM_TRAJECTORIES):
            next_state_probs[i, :] = P[a_t[i], s_t[i], :]

        rand_next = np.random.rand(NUM_TRAJECTORIES)
        s_next = sample_categorical_parallel(next_state_probs, rand_next)
        states[:, t + 1] = s_next

    return action_counts, visit_counts

"""Compute the likelihood of observed data under a time-variant policy.

Given:
  - policy[t, s, a]: Probability of taking action `a` in state `s` at time `t`.
  - visit_counts[t, s]: Number of times state `s` was visited at time `t`.
  - action_counts[t, s, a]: Number of times action `a` was chosen in state `s` at time `t`.

The log-likelihood is:
  L = sum_{t, s, a} ( action_counts[t, s, a] * log( policy[t, s, a] ) ).

The likelihood is exp(L). Note that sum_{a} action_counts[t, s, a] should match visit_counts[t, s].
"""


def compute_likelihood(policy: np.ndarray, visit_counts: np.ndarray, action_counts: np.ndarray) -> float:
    """Computes the likelihood of the observed data (visit_counts, action_counts) under the given policy."""
    # policy is shaped (T, S, A)
    # visit_counts is shaped (T, S)
    # action_counts is shaped (T, S, A)

    # Compute log-likelihood
    log_likelihood = 0.0
    T, S, A = policy.shape
    for t in range(T):
        for s in range(S):
            for a in range(A):
                count = action_counts[t, s, a]
                p = policy[t, s, a]
                if p == 0.0:
                    if count > 0:
                        # Probability zero but count is nonzero -> impossible event
                        return 0.0  # or float('-inf') if we prefer log-likelihood form
                    else:
                        continue
                log_likelihood += count * np.log(p)

    # Exponentiate the log-likelihood to get the likelihood
    # likelihood = np.exp(log_likelihood)
    assert action_counts.sum() == visit_counts.sum()
    return log_likelihood/visit_counts.sum()