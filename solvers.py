import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_milp(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    # Create Gurobi model
    M = 100
    model = gp.Model("MILP")
    model.setParam('OutputFlag', False)
    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
    z = model.addVars(T, vtype=GRB.BINARY, name="z")

    # Objective: Minimize sum of z_t
    model.setObjective(gp.quicksum(z[t] for t in range(T)), GRB.MINIMIZE)

    # Auxiliary variables for the infinity norm
    norm_vars = model.addVars(T, vtype=GRB.CONTINUOUS, name="norm")

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)),
                                name=f"r_def_{t}_{s}_{a}")
    
    # Calculate the infinity norm using auxiliary variables
    for t in range(1, T):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(norm_vars[t] >= r[t, s, a] - r[t-1, s, a], name=f"norm_pos_{t}_{s}_{a}")
                model.addConstr(norm_vars[t] >= -(r[t, s, a] - r[t-1, s, a]), name=f"norm_neg_{t}_{s}_{a}")

    # Big-M constraints using auxiliary variables
    for t in range(1, T):
        model.addConstr(-M * z[t] <= norm_vars[t], name=f"bigM_lower_{t}")
        model.addConstr(norm_vars[t] <= M * z[t], name=f"bigM_upper_{t}")

    # Solve the model
    model.optimize()

    # Print results
    if model.status == GRB.OPTIMAL:
        r_values = np.zeros((T, n_states, n_actions))
        nu_values = np.zeros((T, n_states))
        z_values = np.zeros(T)
        
        for t in range(T):
            for s in range(n_states):
                for a in range(n_actions):
                    r_values[t, s, a] = r[t, s, a].x
            for j in range(n_states):
                nu_values[t, j] = nu[t, j].x
            z_values[t] = z[t].x
        
        return r_values, nu_values, z_values
    else:
        print("No optimal solution found.")


def solve_L_1(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    model = gp.Model("MILP_L1")
    model.setParam('OutputFlag', False)
    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
    diff = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="diff")

    # Objective: Minimize sum of absolute differences (L1 norm)
    model.setObjective(gp.quicksum(diff[t, s, a] for t in range(T-1) for s in range(n_states) for a in range(n_actions)), GRB.MINIMIZE)

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
               

    for t in range(1,T):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(diff[t, s, a] >= r[t, s, a] - r[t-1, s, a])
                model.addConstr(diff[t, s, a] >= -(r[t, s, a] - r[t-1, s, a]))

    model.optimize()
    return extract_solution(model, r, nu, T, n_states, n_actions)


def solve_L_inf(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    model = gp.Model("MILP_Linf")
    model.setParam('OutputFlag', False)
    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
    norm_vars = model.addVars(T, vtype=GRB.CONTINUOUS, name="norm")

    # Objective: Minimize sum of infinity norms (L∞ norm)
    model.setObjective(gp.quicksum(norm_vars[t] for t in range(T-1)), GRB.MINIMIZE)

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
            
    # Constraints
    for t in range(1,T):
        for s in range(n_states):
            for a in range(n_actions):
                
                model.addConstr(norm_vars[t] >= r[t, s, a] - r[t-1, s, a])
                model.addConstr(norm_vars[t] >= -(r[t, s, a] - r[t-1, s, a]))


    model.optimize()
    return extract_solution(model, r, nu, T, n_states, n_actions)


def extract_solution(model, r, nu, T, n_states, n_actions):
    if model.status == GRB.OPTIMAL:
        r_values = np.zeros((T, n_states, n_actions))
        nu_values = np.zeros((T, n_states))
        
        for t in range(T):
            for s in range(n_states):
                for a in range(n_actions):  
                    r_values[t, s, a] = r[t, s, a].x
            for j in range(n_states):
                nu_values[t, j] = nu[t, j].x
        
        return r_values, nu_values
    else:
        print("No optimal solution found.")
        return None, None
