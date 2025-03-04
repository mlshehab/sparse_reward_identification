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

def solve_greedy_linear(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = 0
    switch_times = []
    rewards_nu_list = []
    for i in range(T-1):
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam('OutputFlag', False)
        model.setObjective(0, GRB.MINIMIZE)
        r = model.addVars(n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(i+1-tau + 1, n_states, vtype=GRB.CONTINUOUS, name="nu")

        ### Reward constraints
        for t in range(tau,i+1):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[s, a] == np.log(pi[t, s, a]) + nu[t-tau,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1-tau, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
        ### Nu consistency constraints
        # if len(rewards_nu_list) > 0:
        #     for s in range(n_states):
        #         model.addConstr(nu[0,s] == rewards_nu_list[-1][1][-1,s], name=f"nu_consistency_{tau}_{switch_times[-1]}")

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            r_values = np.zeros((n_states, n_actions))
            nu_values = np.zeros((T, n_states))
            for t in range(tau,i+1):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[s, a] = r[s, a].x
                for j in range(n_states):
                    nu_values[t-tau, j] = nu[t-tau, j].x
        else:
            switch_times += [i]
            tau = i
            rewards_nu_list.append((r_values,nu_values))

    if tau < T-2:
        switch_times += [T-1]
        rewards_nu_list.append((r_values,nu_values))    

    return switch_times, rewards_nu_list

def solve_L_1(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    model = gp.Model("MILP_L1")
    model.setParam('OutputFlag', False)
    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
    diff = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="diff")

    # Objective: Minimize sum of absolute differences (L1 norm)
    model.setObjective(gp.quicksum(diff[t, s, a] for t in range(T) for s in range(n_states) for a in range(n_actions)), GRB.MINIMIZE)

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))

    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
             model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

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
    model.setObjective(gp.quicksum(norm_vars[t] for t in range(T)), GRB.MINIMIZE)

    # Constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
    
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
             model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

    # Constraints
    for t in range(1,T):
        for s in range(n_states):
            for a in range(n_actions):
                
                model.addConstr(norm_vars[t] >= r[t, s, a] - r[t-1, s, a])
                model.addConstr(norm_vars[t] >= -(r[t, s, a] - r[t-1, s, a]))


    model.optimize()
    return extract_solution(model, r, nu, T, n_states, n_actions)

def solve_L2(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    # Create Gurobi model
    model = gp.Model("MILP")
    model.setParam('OutputFlag', 0)  # Silence optimizer output

    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")

    # Auxiliary variables for L2 norm calculation
    norm_vars = model.addVars(T, vtype=GRB.CONTINUOUS, name="norm")

    # Objective: Minimize sum of L2 norms
    model.setObjective(gp.quicksum(norm_vars[t] for t in range(1, T)), GRB.MINIMIZE)

    # Constraints defining the reward function
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(
                    r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)),
                    name=f"r_def_{t}_{s}_{a}"
                )
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
             model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

    # Quadratic constraints for L2 norm
    for t in range(1, T):
        model.addQConstr(
            norm_vars[t] * norm_vars[t] >= gp.quicksum((r[t, s, a] - r[t-1, s, a]) * (r[t, s, a] - r[t-1, s, a]) 
                                                       for s in range(n_states) for a in range(n_actions)),
            name=f"l2_norm_{t}"
        )

    # Solve the model
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
