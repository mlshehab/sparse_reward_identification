import gurobipy as gp
from gurobipy import GRB
import numpy as np

# Define problem parameters
T = 10  # Time horizon
n_states = 5  # Number of states
n_actions = 3  # Number of actions
M = 100  # Large constant for big-M constraints
gamma = 0.9  # Discount factor

# Example values for pi_t, E, and P
pi_t = np.random.rand(T, n_states, n_actions)
E = np.random.rand(n_states, n_actions, n_states)
P = np.random.rand(n_states, n_actions, n_states)

# Create Gurobi model
model = gp.Model("MILP")

# Decision variables
r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
z = model.addVars(T, vtype=GRB.BINARY, name="z")

# Objective: Minimize sum of z_t
model.setObjective(gp.quicksum(z[t] for t in range(T)), GRB.MINIMIZE)

# Constraints
for t in range(T-1):
    for s in range(n_states):
        for a in range(n_actions):
            model.addConstr(r[t, s, a] == np.log(pi_t[t, s, a]) + gp.quicksum(E[s, a, j] * nu[t, j] for j in range(n_states)) - gamma * gp.quicksum(P[s, a, j] * nu[t+1, j] for j in range(n_states)),
                            name=f"r_def_{t}_{s}_{a}")
    model.addConstr(-M * z[t] <= gp.quicksum(r[t, s, a] - r[t-1, s, a] for s in range(n_states) for a in range(n_actions)), name=f"bigM_lower_{t}")
    model.addConstr(gp.quicksum(r[t, s, a] - r[t-1, s, a] for s in range(n_states) for a in range(n_actions)) <= M * z[t], name=f"bigM_upper_{t}")

# Solve the model
model.optimize()

# Print results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for t in range(T):
        for s in range(n_states):
            for a in range(n_actions):
                print(f"r[{t},{s},{a}] = {r[t, s, a].x}")
        for j in range(n_states):
            print(f"nu[{t},{j}] = {nu[t, j].x}")
        print(f"z[{t}] = {z[t].x}")
else:
    print("No optimal solution found.")
