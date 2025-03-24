import gurobipy as gp
import cvxpy as cp
from gurobipy import GRB
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power
import scipy.linalg

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

    # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

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

def solve_greedy_backward_alpha(gw, pi, U):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    n_features = len(U)

    tau = T
    switch_times = []

    i = T-1

    alpha_values = np.zeros((T, n_features))
    nu_values = np.zeros((T, n_states))

    while i >= 0:
        print(f"Testing from {i} to {T-1}. {tau=}")
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam("NumericFocus", 1)
        model.setParam('OutputFlag', False)
        model.setObjective(0, GRB.MINIMIZE)
        alpha = model.addVars(T, n_features, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")

        ### Reward constraints
        for t in range(i,T-1):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(gp.quicksum(alpha[t,j] * U[j][s,a] for j in range(n_features)) == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

        # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(gp.quicksum(alpha[T-1,j] * U[j][s,a] for j in range(n_features)) == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

        ### Reward Consistency constraints
        for t in range(i,tau-1):
            for j in range(n_features):
                model.addConstr(alpha[t,j] == alpha[tau-1,j], name=f"r_def_{t}_{s}_{a}")        
 
        ### Reward Other Intervals Consistency constraints
        for t in range(tau,T):
            for j in range(n_features):
                model.addConstr(alpha[t,j] == alpha_values[t,j], name=f"r_def_{t}_{s}_{a}")               

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(T):
                for j in range(n_features):
                    alpha_values[t, j] = alpha[t,j].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t, j].x
            i -= 1
        else:
            print(model.status)
            switch_times += [i+1]
            tau = i+1
            print(f"Infeasibility found. New tau={tau}")

    r_values = np.zeros(shape=(T, n_states, n_actions))
    for t in range(T):
        for s in range(n_states):
            for a in range(n_actions):
                r_values[t,s,a] = np.sum([alpha_values[t,j]*U[j][s,a] for j in range(n_features)])
    return r_values, alpha_values, nu_values, switch_times[::-1]

def solve_greedy_backward(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = T
    switch_times = []

    i = T-1

    r_values = np.zeros((T, n_states, n_actions))
    nu_values = np.zeros((T, n_states))

    while i >= 0:
        print(f"Testing from {i} to {T-1}. {tau=}")
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
                    model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

        # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

        ### Reward Consistency constraints
        for t in range(i,tau-1):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[t,s, a] == r[tau-1,s,a], name=f"r_def_{t}_{s}_{a}")        
 
        ### Reward&Nu Other Intervals Consistency constraints
        for t in range(tau,T):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[t,s, a] == r_values[t,s,a], name=f"r_def_{t}_{s}_{a}")               
                model.addConstr(nu[t,s] == nu_values[t,s,], name=f"nu_def_{t}_{s}_{a}")               

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(T):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[t, s, a] = r[t, s, a].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t, j].x
            i -= 1
        else:
            print(model.status)
            switch_times += [i+1]
            tau = i+1
            print(f"Infeasibility found. New tau={tau}")

    return r_values, nu_values, switch_times

def solve_greedy_backward_bisection(gw, pi):
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
                    model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

        # Add constraints for the last time step
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

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

def solve_greedy_backward_bisection_smaller(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = T
    switch_times = []

    i = T//2
    inf_i = -1
    sup_i = T

    r_values = np.zeros((T, n_states, n_actions))
    nu_values = np.zeros((T+1, n_states))

    n_iterations = 0
    while i >= 0:
        n_iterations += 1
        print(f"Testing from {i} to {T-1}. {tau=}. {inf_i=}, {sup_i=}")
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam("NumericFocus", 1)
        model.setParam('OutputFlag', False)
        model.setObjective(0, GRB.MINIMIZE)
        r = model.addVars(n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(tau-i + 1, n_states, vtype=GRB.CONTINUOUS, name="nu")

        ### Reward constraints
        for t in range(i,tau):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[s, a] == np.log(pi[t, s, a]) + nu[t-i,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1-i, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

 
        for s in range(n_states):
            model.addConstr(nu[tau-i, s] == nu_values[tau,s], name=f"r_def_{t}_{s}_{a}")        


        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(i, tau):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[t, s, a] = r[s, a].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t-i, j].x
            
                # for j in range(n_states):
                #     nu_values[tau, j] = nu[tau-i, j].x           

            sup_i = i
            # if inf_i+1 == i and i>0:
            #     switch_times += [i]
            #     tau = i
            #     inf_i = -1

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

def solve_greedy_backward_bisection_L1_by_interval(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = T
    switch_times = []

    i = T//2
    inf_i = -1
    sup_i = T

    r_values = np.zeros((T, n_states, n_actions))
    nu_values = np.zeros((T+1, n_states))


    n_iterations = 0
    while i >= 0:
        n_iterations += 1
        print(f"Testing from {i} to {T-1}. {tau=}. {inf_i=}, {sup_i=}")
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam("NumericFocus", 1)
        model.setParam('OutputFlag', True)
        # model.setObjective(gp., GRB.MINIMIZE)
        r = model.addVars(n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(tau-i + 1, n_states, vtype=GRB.CONTINUOUS, name="nu")

        model.setObjective(gp.quicksum(r[i,j] for i in range(n_states) for j in range(n_actions)), GRB.MINIMIZE)

        ### Reward constraints
        for t in range(i,tau):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[s, a] == np.log(pi[t, s, a]) + nu[t-i,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1-i, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

        for s in range(n_states):
            model.addConstr(nu[tau-i, s] == nu_values[tau,s], name=f"r_def_{t}_{s}_{a}")        

        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(i, tau):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[t, s, a] = r[s, a].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t-i, j].x
            
                # for j in range(n_states):
                #     nu_values[tau, j] = nu[tau-i, j].x           

            sup_i = i
            # if inf_i+1 == i and i>0:
            #     switch_times += [i]
            #     tau = i
            #     inf_i = -1

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

def solve_greedy_backward_bisection_L1_overall(gw, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = T
    switch_times = []

    i = T//2
    inf_i = -1
    sup_i = T

    r_values = np.zeros((T, n_states, n_actions))
    nu_values = np.zeros((T+1, n_states))


    n_iterations = 0
    while i >= 0:
        n_iterations += 1
        print(f"Testing from {i} to {T-1}. {tau=}. {inf_i=}, {sup_i=}")
        ## Is feasible?
        model = gp.Model("Feasible")
        model.setParam("NumericFocus", 1)
        model.setParam('OutputFlag', False)
        model.setObjective(0, GRB.MINIMIZE)
        r = model.addVars(n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
        nu = model.addVars(tau-i + 1, n_states, vtype=GRB.CONTINUOUS, name="nu")

        ### Reward constraints
        for t in range(i,tau):
            for s in range(n_states):
                for a in range(n_actions):
                    model.addConstr(r[s, a] == np.log(pi[t, s, a]) + nu[t-i,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1-i, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")

 
        for s in range(n_states):
            model.addConstr(nu[tau-i, s] == nu_values[tau,s], name=f"r_def_{t}_{s}_{a}")        


        model.optimize()
        if model.Status == GRB.OPTIMAL:
            for t in range(i, tau):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[t, s, a] = r[s, a].x
                for j in range(n_states):
                    nu_values[t, j] = nu[t-i, j].x       

            sup_i = i
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

    n_switches = len(switch_times[::-1])

    def find_reward_interval(t):
        i = 0 
        while t >= (switch_times[::-1] + [T])[i]:
            i += 1
        return i

    print(f"{n_switches=}")
        ## Is feasible?
    model = gp.Model("Feasible")
    model.setParam("NumericFocus", 1)
    model.setParam('OutputFlag', False)
    r = model.addVars(n_switches + 1, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")

    model.setObjective(gp.quicksum(r[k, i,j] for i in range(n_states) for j in range(n_actions) for k in range(n_switches + 1)), GRB.MINIMIZE)

    ### Reward constraints
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                model.addConstr(r[find_reward_interval(t), s, a] == np.log(pi[t, s, a]) + nu[t,s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)), name=f"r_def_{t}_{s}_{a}")
        print(f"Reward interval of time {t}: {find_reward_interval(t)}")
    
    # interval_bounds = [0] + switch_times[::-1]
    # for ind_st in range(len(interval_bounds)):
    #     for t in range(interval_bounds[ind_st], interval_bounds[ind_st+1]):
    #     for s in range(n_states):
    #         for a in range(n_actions):
    #         r[t]

    for s in range(n_states):
        for a in range(n_actions):
            model.addConstr(r[n_switches, s, a] == np.log(pi[T-1, s, a]) + nu[T-1,s], name=f"r_def_{t}_{s}_{a}")
  
    model.optimize()
    if model.status == GRB.OPTIMAL:
        r_values = np.zeros((T, n_states, n_actions))
        nu_values = np.zeros((T, n_states))
        
        for t in range(T):
            for s in range(n_states):
                for a in range(n_actions):
                    r_values[t, s, a] = r[find_reward_interval(t), s, a].x
            for j in range(n_states):
                nu_values[t, j] = nu[t, j].x
        
    else:
        print("No optimal solution found.")


    print(f"It took {n_iterations} iteration to solve")
    return r_values, nu_values, switch_times[::-1]

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

def solve_PROBLEM_2(gw, U, sigmas, pi):

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
                model.addConstr(r[t, s, a] == np.log(pi[t, s, a]) + nu[t, s] - gamma * gp.quicksum(P[a][s, j] * nu[t+1, j] for j in range(n_states)))
                model.addConstr(r[t, s, a] == gp.quicksum(U[s + a * n_states, i] * alpha[t, i] for i in range(n_features)))
                
    # Add constraints for the last time step
    for s in range(n_states):
        for a in range(n_actions):
            model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )
             
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


def solve_PROBLEM_2_cvxpy(gw, U, sigmas, pi):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    n_features = U.shape[1]

    # Decision variables
    r = cp.Variable((T, n_states * n_actions))  # Flattened r
    nu = cp.Variable((T, n_states))
    alpha = cp.Variable((T, n_features))

    constraints = []

    # Constraints for t in 0 to T-2
    for t in range(T - 1):
        for s in range(n_states):
            for a in range(n_actions):
                idx = s + a * n_states
                trans_term = gamma * cp.sum(cp.multiply(P[a][s, :], nu[t + 1, :]))

                constraints.append(r[t, idx] == np.log(pi[t, s, a]) + nu[t, s] - trans_term)
                constraints.append(r[t, idx] == U[idx, :] @ alpha[t, :])

    # Constraints for the last timestep
    for s in range(n_states):
        for a in range(n_actions):
            idx = s + a * n_states
            constraints.append(r[T - 1, idx] == np.log(pi[T - 1, s, a]) + nu[T - 1, s])
            constraints.append(r[T - 1, idx] == U[idx, :] @ alpha[T - 1, :])

    # Objective: Minimize sum of squared differences weighted by sigma
    objective = cp.Minimize(cp.sum(
        [(alpha[t, i] - alpha[t - 1, i]) ** 2 / (2 * sigmas[i] ** 2)
         for i in range(n_features) for t in range(1, T)]
    ))

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)

    if problem.status == cp.OPTIMAL:
        print("Problem solved to optimality.")
    else:
        print("Solver did not reach optimality. Status:", problem.status)

    alpha_values = alpha.value
    r_reshaped = r.value.reshape(T, n_actions, n_states).transpose(0, 2, 1)  # Convert to (T, n_states, n_actions)

    return alpha_values, (r_reshaped, nu.value, alpha_values)  # Placeholder for extract_solution

def solve_PROBLEM_3(gw, U, sigmas, pi):
    import cvxpy as cp
    import numpy as np

    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    n_features = U.shape[1]

    # Decision variables
    r = cp.Variable((T, n_states * n_actions))  # Flattened reward matrix
    nu = cp.Variable((T, n_states))
    # alpha = cp.Variable((T, n_features))
    
    # Constraints
    constraints = []
    for t in range(T-1):
        for s in range(n_states):
            for a in range(n_actions):
                idx = s + a * n_states
                constraints.append(r[t, idx] == cp.log(pi[t, s, a]) + nu[t, s] - gamma * (P[a][s, :] @ nu[t+1, :]))
                # constraints.append(r[t, idx] == U[idx, :] @ alpha[t, :])
    
    for s in range(n_states):
        for a in range(n_actions):
            idx = s + a * n_states
            constraints.append(r[T-1, idx] == cp.log(pi[T-1, s, a]) + nu[T-1, s])
            # constraints.append(r[T-1, idx] == U[idx, :] @ alpha[T-1, :])
    
    # Objective: Minimize the nuclear norm of the reward matrix
    objective = cp.Minimize(cp.norm(r,"nuc"))
    
    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    
    print("Status:", problem.status)
    
    return r.value, nu.value


def solve_PROBLEM_3_RNNM(gw, U, sigmas, pi, max_iter=10, delta= 1e-4):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    n_features = U.shape[1]
    
    # Initialize weights
    W1 = np.eye(T)
    W2 = np.eye(n_states * n_actions)
    
    # Initialize variables to warm start the solver
    r_prev = np.zeros((T, n_states * n_actions))
    nu_prev = np.zeros((T, n_states))
    
    for _ in range(max_iter):
        # Reset decision variables in each iteration
        r = cp.Variable((T, n_states * n_actions), value=r_prev)
        nu = cp.Variable((T, n_states), value=nu_prev)
        
        # Constraints
        constraints = []
        for t in range(T - 1):
            for s in range(n_states):
                for a in range(n_actions):
                    idx = s + a * n_states
                    constraints.append(r[t, idx] == cp.log(pi[t, s, a]) + nu[t, s] - gamma * cp.sum(P[a][s, :] @ nu[t + 1, :]))
        
        for s in range(n_states):
            for a in range(n_actions):
                idx = s + a * n_states
                constraints.append(r[T - 1, idx] == cp.log(pi[T - 1, s, a]) + nu[T - 1, s])
        
        # Weighted nuclear norm minimization
        objective = cp.Minimize(cp.norm(W1 @ r @ W2, "nuc"))
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=True)  # Reduced verbosity
        
        # Update weights based on singular values
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            U, Sigma, Vt = np.linalg.svd(W1 @ r.value @ W2, full_matrices=False)
    
            Sigma = np.diag(Sigma)
        
            Y = np.linalg.pinv(W1) @ U @ Sigma @ U.T @ np.linalg.pinv(W1)
            Z = np.linalg.pinv(W2) @ Vt.T @ Sigma@ Vt @ np.linalg.pinv(W2)
           
            W1 = scipy.linalg.sqrtm(np.linalg.pinv(Y + delta * np.eye(T)))
            W2 = scipy.linalg.sqrtm(np.linalg.pinv(Z + delta * np.eye(n_states * n_actions)))
            # Warm start for next iteration
            r_prev = r.value
            nu_prev = nu.value
        else:
            print("Optimization failed.")
            break
    
    print("Final Status:", problem.status)
    return r.value, nu.value



def solve_PROBLEM_3_RTH(gw, U, sigmas, pi, max_iter=10, delta=1e-4):
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    n_features = U.shape[1]
    
    # Initialize variables to warm start the solver
    r_prev = np.zeros((T, n_states * n_actions))
    nu_prev = np.zeros((T, n_states))

    Y_k = np.eye(T)
    Z_k = np.eye(n_states * n_actions)
    
    for _ in range(max_iter):
        # Decision variables
        r = cp.Variable((T, n_states * n_actions), value=r_prev)
        nu = cp.Variable((T, n_states), value=nu_prev)
        Y = cp.Variable((T, T), PSD=True)
        Z = cp.Variable((n_states * n_actions, n_states * n_actions), PSD=True)
        
        # Constraints
        constraints = [
            cp.bmat([[Y, r], [r.T, Z]]) >> 0,  # Positive semi-definite constraint
        ]
        
        for t in range(T - 1):
            for s in range(n_states):
                for a in range(n_actions):
                    idx = s + a * n_states
                    constraints.append(r[t, idx] == cp.log(pi[t, s, a]) + nu[t, s] - gamma * cp.sum(P[a][s, :] @ nu[t + 1, :]))
        
        for s in range(n_states):
            for a in range(n_actions):
                idx = s + a * n_states
                constraints.append(r[T - 1, idx] == cp.log(pi[T - 1, s, a]) + nu[T - 1, s])
        
        # Reweighted Trace Heuristic Objective
        # Compute inverses using scipy.linalg.inv
        Y_k_inv = scipy.linalg.pinv(Y_k + delta * np.eye(T))
        Z_k_inv = scipy.linalg.pinv(Z_k + delta * np.eye(n_states * n_actions))
        
        # Reweighted Trace Heuristic Objective
        objective = cp.Minimize(cp.trace(Y_k_inv @ Y) + cp.trace(Z_k_inv @ Z))

        # Solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=True)
        
        # Update weights based on solution
        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            r_prev = r.value
            iteration = _
            singular_values = np.linalg.svd(r_prev, compute_uv=False)
            rounded_singular_values = np.round(singular_values, 4)
            rank_r_prev = np.linalg.matrix_rank(r_prev)
            
            print(f"Iteration: {iteration}")
            print(f"Singular values of r_prev (rounded to 4 decimal points): {rounded_singular_values}")
            print(f"Rank of r_prev: {rank_r_prev}")
            nu_prev = nu.value
            Y_k = Y.value
            Z_k = Z.value
        else:
            print("Optimization failed.")
            break
    
    print("Final Status:", problem.status)
    return r.value, nu.value



def extract_solution(model, r, nu, alpha, T, n_states, n_actions, n_features):
 
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
