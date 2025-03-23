import gurobipy as gp
import cvxpy as cp
from gurobipy import GRB
import numpy as np

def solve_milp_noisy(gw, pi, b):
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
                model.addConstr(r[t, s, a] >= np.log(pi[t, s, a]) - b[t,s] + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)),
                                name=f"r_def_{t}_{s}_{a}")
                model.addConstr(r[t, s, a] <= np.log(pi[t, s, a]) + b[t,s] + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)),
                                name=f"r_def_{t}_{s}_{a}")

    # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[T-1, s, a] >= np.log(pi[T-1, s, a]) - b[T-1,s] + nu[T-1, s]  )
                model.addConstr(r[T-1, s, a] <= np.log(pi[T-1, s, a]) + b[T-1,s] + nu[T-1, s]  )


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

def solve_greedy_backward_bisection_noisy(gw, pi, b):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = T
    switch_times = []

    i = T//2
    inf_i = -1
    sup_i = T

    r_values = np.zeros((T, n_states, n_actions))
    nu_values = np.zeros((T, n_states))

    n_iterations = 0
    while i >= 0:
        n_iterations += 1
        print(f"Testing from {i} to {T-1}. {tau=}. {inf_i=}, {sup_i=}")
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam("NumericFocus", 1)
        model.setParam('OutputFlag', False)
        model.setObjective(0, GRB.MINIMIZE)
        r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")

        ### Reward constraints
        for t in range(i,T-1):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[t, s, a] >= np.log(pi[t, s, a]) - b[t,s] + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
                    model.addConstr(r[t, s, a] <= np.log(pi[t, s, a]) + b[t,s] + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

        # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[T-1, s, a] >= np.log(pi[T-1, s, a]) - b[T-1,s] + nu[T-1, s]  )
                model.addConstr(r[T-1, s, a] <= np.log(pi[T-1, s, a]) + b[T-1,s] + nu[T-1, s]  )

        ### Reward Consistency constraints
        for t in range(i,tau-1):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[t,s, a] == r[tau-1,s,a], name=f"r_def_{t}_{s}_{a}")        
 
        ### Reward Other Intervals Consistency constraints
        for t in range(tau,T):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[t,s, a] == r_values[t,s,a], name=f"r_def_{t}_{s}_{a}")               
            model.addConstr(nu[t,s] == nu_values[t,s], name=f"nu_def_{t}_{s}_{a}")               


        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(T):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[t, s, a] = r[t, s, a].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t, j].x
            

            sup_i = i
            if inf_i+1 == i and i>0:
                switch_times += [i]
                tau = i
                inf_i = -1

        else:
            print(f"Infeasibility found. New tau={tau}")
            if inf_i == i or sup_i == i + 1:
                switch_times += [i+1]
                tau = i+1
                inf_i = -1
                sup_i = i+1
            else:
                inf_i = i
        i = (sup_i + inf_i)//2

    print(f"It took {n_iterations} iteration to solve")
    return r_values, nu_values, switch_times[::-1]

def solve_PROBLEM_2_noisy(gw, U, sigmas, pi, b):

    n_features = U.shape[1]
    
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
   
    model = gp.Model("Problem 2")

    # Decision variables
    r = model.addVars(T, n_states, n_actions, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, lb=float("-inf"), vtype=GRB.CONTINUOUS, name="nu")
    alpha = model.addVars(T, n_features, lb=float("-inf"),  vtype=GRB.CONTINUOUS, name="alpha")
    
    # Objective: Minimize sum of infinity norms (L∞ norm)
    model.setObjective(gp.quicksum(( (alpha[t, i] - alpha[t - 1, i]) ** 2) / (2 * sigmas[i] ** 2) for i in range(n_features) for t in range(1, T)), GRB.MINIMIZE)
    # model.setObjective(0, GRB.MINIMIZE)
    
    # Constraints
    # model.addConstr(alpha[0, 0] == 0)
    # model.addConstr(alpha[0, 1] == 1)

    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[t, s, a] <= np.log(pi[t, s, a]) + b[t,s] + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
                model.addConstr(r[t, s, a] >= np.log(pi[t, s, a]) - b[t,s] + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
                model.addConstr(r[t, s, a] == gp.quicksum(U[s + a * n_states, i] * alpha[t, i] for i in range(n_features)))
                
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
            model.addConstr(r[T-1, s, a] <= np.log(pi[T-1, s, a]) + b[T-1, s] + nu[T-1, s]  )
            model.addConstr(r[T-1, s, a] >= np.log(pi[T-1, s, a]) - b[T-1, s] + nu[T-1, s]  )

            model.addConstr(r[T-1, s, a] == gp.quicksum(U[s + a * n_states, i] * alpha[T-1, i] for i in range(n_features)))

    
    model.setParam('OutputFlag', 1)  # Enable detailed output
    model.setParam("IterationLimit", 1e8)
    model.setParam("BarIterLimit", 1000 )  # Increase to a larger number
    model.optimize()
    # model.computeIIS()
    # model.write("model.ilp")
    print("Status: ", model.status)
   
    # print("Iterations:", model.IterCount)  # Number of iterations

    alpha_values = np.zeros((T, n_features))
    for t in range(T):
        for i in range(n_features):
            alpha_values[t, i] = alpha[t, i].x
    # print(alpha_values)

    return alpha_values, extract_solution(model, r, nu, alpha, T, n_states, n_actions, n_features)

def extract_solution(model, r, nu, alpha, T, n_states, n_actions, n_features):
    print(model.status)
    if model.status == GRB.OPTIMAL:
        r_values = np.zeros((T, n_states, n_actions))
        nu_values = np.zeros((T, n_states))
        alpha_values = np.zeros((T, n_features))
        
        for t in range(T):
            for s in range(n_states):
                for a in range(n_actions):  
                    r_values[t, s, a] = r[t, s, a].x
            for j in range(n_states):
                nu_values[t, j] = nu[t, j].x
            for i in range(n_features):
                alpha_values[t, i] = alpha[t, i].x
            
        
        return r_values, nu_values, alpha_values
    else:
        print("No optimal solution found.")
        
        return None, None, None
