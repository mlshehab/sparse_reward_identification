import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from dynamics import BasicGridWorld


class ExpertPolicyGenerator:
    def __init__(self, env):
        self.env = env
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.horizon = env.horizon
        self.discount = env.discount
        
    def compute_soft_optimal_policy(self, reward):
        """Compute the soft optimal policy using soft value iteration"""
        Q = np.zeros((self.horizon, self.n_states, self.n_actions))
        V = np.zeros((self.horizon+1, self.n_states))
        policy = np.zeros((self.horizon, self.n_states, self.n_actions))

        for h in reversed(range(self.horizon)):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.env.transition_probability[s, a, :]
                    Q[h,s,a] = reward[s] + self.discount * np.dot(next_probs, V[h+1])
                
                V[h,s] = logsumexp(Q[h,s])
                policy[h,s] = np.exp(Q[h,s] - V[h,s])
                policy[h,s] /= policy[h,s].sum()
        
        return policy

class MaxEntIRL:
    def __init__(self, env, expert_trajs, reward_states, n_iter=100, lr=0.1):
        self.env = env
        self.expert_trajs = expert_trajs
        self.reward_states = reward_states  # List of states with rewards
        self.n_iter = n_iter
        self.lr = lr
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.horizon = env.horizon
        self.discount = env.discount
        
        # Initialize weights only for reward_states
        self.weights = np.zeros(self.n_states)
        self.weights[self.reward_states] = np.random.randn(len(self.reward_states))
        
        # Compute expert feature expectations (only for reward_states)
        self.expert_features = self._compute_expert_features()

    def _compute_expert_features(self):
        features = np.zeros(len(self.reward_states))
        for traj in self.expert_trajs:
            for state, _ in traj:
                if state in self.reward_states:
                    idx = self.reward_states.index(state)
                    features[idx] += 1
        return features / len(self.expert_trajs)

    def soft_value_iteration(self):
        Q = np.zeros((self.horizon, self.n_states, self.n_actions))
        V = np.zeros((self.horizon+1, self.n_states))
        policy = np.zeros((self.horizon, self.n_states, self.n_actions))

        for h in reversed(range(self.horizon)):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.env.transition_probability[s, a, :]
                    # Only reward_states contribute to reward
                    reward = self.weights[s] if s in self.reward_states else 0
                    Q[h,s,a] = reward + self.discount * np.dot(next_probs, V[h+1])
                
                V[h,s] = logsumexp(Q[h,s])
                policy[h,s] = np.exp(Q[h,s] - V[h,s])
                policy[h,s] /= policy[h,s].sum()
        
        return policy, Q, V

    def compute_expected_features(self, policy):
        # Forward calculation of state visitation frequencies
        D = np.zeros((self.horizon+1, self.n_states))
        D[0, 0] = 1.0  # Start at state 0
        
        for h in range(self.horizon):
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_probs = self.env.transition_probability[s, a, :]
                    D[h+1] += D[h,s] * policy[h,s,a] * next_probs
        
        # Return features only for reward_states
        return np.array([D.sum(axis=0)[s] for s in self.reward_states])

    def train(self):
        for _ in range(self.n_iter):
            # Compute optimal policy with current weights
            policy, _, _ = self.soft_value_iteration()
            
            # Compute expected features under policy
            policy_features = self.compute_expected_features(policy)
            
            # Update only the weights for reward_states
            gradient = policy_features - self.expert_features
            self.weights[self.reward_states] -= self.lr * gradient
            
        return self.weights

# Create environment
grid_size = 5
horizon = 49
gt_reward = np.zeros(grid_size**2)
reward_states = [0, 14]  # States with rewards
gt_reward[reward_states[0]] = -1   # Small negative reward at start
gt_reward[reward_states[1]] = 5   # Intermediate reward state (2,4) in 0-indexed grid
# gt_reward[reward_states[2]] = 10  # Goal state at (4,4)

env = BasicGridWorld(
    grid_size=grid_size,
    wind=0.1,   # Small wind probability
    discount=0.9,
    horizon=horizon,
    reward=gt_reward
)

# Generate expert trajectories using soft optimal policy
expert_gen = ExpertPolicyGenerator(env)
# soft_optimal_policy = expert_gen.compute_soft_optimal_policy(gt_reward)
soft_optimal_policy = np.load('pi_expert.npy')
# print(soft_optimal_policy.shape)
expert_trajs = []
for _ in range(50):
    traj = []
    state = 0  # Start at (0,0)
    for h in range(horizon):
        action = np.random.choice(env.n_actions, p=soft_optimal_policy[h, state])
        traj.append((state, action))
        state = env.step(state, action)
    expert_trajs.append(traj)

# Train MaxEnt IRL with constrained rewards
irl = MaxEntIRL(env, expert_trajs, reward_states=reward_states, n_iter=100, lr=0.1)
learned_weights = irl.train()
# Reshape learned weights to horizon x n_states
learned_weights_matrix = np.zeros((horizon, env.n_states))

for t in range(horizon):
    learned_weights_matrix[t, :] = learned_weights
np.save('data/static_MaxEntIRL_reward.npy', learned_weights_matrix)


# print("Learned weights matrix shape:", learned_weights_matrix.shape)
# print(learned_weights_matrix)




# Convert weights to grid format
gt_grid = gt_reward.reshape((grid_size, grid_size)).T
learned_grid = learned_weights.reshape((grid_size, grid_size)).T

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(gt_grid, cmap='viridis')
ax1.set_title("Ground Truth Reward")
ax1.set_xticks([])
ax1.set_yticks([])
for (j,i), label in np.ndenumerate(gt_grid):
    ax1.text(i,j,f"{label:.1f}",ha='center',va='center')

im = ax2.imshow(learned_grid, cmap='viridis')
ax2.set_title(f"Learned Reward (States {reward_states} only)")
ax2.set_xticks([])
ax2.set_yticks([])
for (j,i), label in np.ndenumerate(learned_grid):
    ax2.text(i,j,f"{label:.1f}",ha='center',va='center')

plt.colorbar(im, ax=ax2)
plt.tight_layout()
plt.show()

# Print comparison
print("\nReward comparison:")
print(f"{'State':<10}{'Ground Truth':<15}{'Learned':<10}")
for s in reward_states:
    print(f"{s:<10}{gt_reward[s]:<15.2f}{learned_weights[s]:<10.2f}")
