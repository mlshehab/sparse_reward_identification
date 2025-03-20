import gurobipy as gp
import cvxpy as cp
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

        # # Add constraints for the last time step
        # for s in range(n_states):
        #     for a in range(n_actions):
        #         model.addConstr(r[T-1, s, a] == np.log(pi[T-1, s, a]) + nu[T-1, s]  )

        ### Reward Consistency constraints
        # for t in range(i,tau-1):
        #     for s in range(n_states):
        #         for a in range(n_actions):
        #             model.addConstr(r[t-i,s, a] == r[tau-i-1,s,a], name=f"r_def_{t}_{s}_{a}")        
 
        for s in range(n_states):
            model.addConstr(nu[tau-i, s] == nu_values[tau,s], name=f"r_def_{t}_{s}_{a}")        

             



        # ### Reward Other Intervals Consistency constraints
        # for t in range(tau,T):
        #     for s in range(n_states):
        #         for a in range(n_actions):
        #             model.addConstr(r[t,s, a] == r_values[t,s,a], name=f"r_def_{t}_{s}_{a}")               
        #     model.addConstr(nu[t,s] == nu_values[t,s], name=f"nu_def_{t}_{s}_{a}")               


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

def solve_greedy_linear_cvxpy(gw, pi): ### Does not work
    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P

    tau = 0
    switch_times = []
    rewards_nu_list = []
    for i in range(T-1):
        ## Is feasible?

        r = cp.Variable((n_states, n_actions), nonneg=True)
        nu = cp.Variable((i+1-tau + 1, n_states), nonneg=True, name="nu")

        ### Reward constraints
        constraints = []
        for t in range(tau,i+1):
            for s in range(n_states):
                for a in range(n_actions):
                    constraints.append(r[s, a] == (np.log(pi[t, s, a]) + nu[t-tau, s] -
                                                 gamma * cp.sum(cp.multiply(P[a][s, :], nu[t+1-tau,:]))))
        ### Nu consistency constraints
        # if len(rewards_nu_list) > 0:
        #     for s in range(n_states):
        #         model.addConstr(nu[0,s] == rewards_nu_list[-1][1][-1,s], name=f"nu_consistency_{tau}_{switch_times[-1]}")

        objective = cp.Minimize(0)
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)

        if problem.status in ["optimal", "feasible"]:
            r_values = np.zeros((n_states, n_actions))
            nu_values = np.zeros((i+1-tau, n_states))
            for t in range(tau,i+1):
                for s in range(n_states):
                    for a in range(n_actions):
                        r_values[s, a] = r[s, a].value
                for j in range(n_states):
                    nu_values[t-tau, j] = nu[t-tau, j].value
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


def solve_PROBLEM_2(gw, U, sigmas, pi):

    n_features = U.shape[1]
    print(n_features)

    T, n_states, n_actions, gamma, P = gw.horizon, gw.n_states, gw.n_actions, gw.discount, gw.P
    print(f"The number of actions is {n_actions}")
    model = gp.Model("Problem 2")

    # Decision variables
    r = model.addVars(T, n_states, n_actions, vtype=GRB.CONTINUOUS, name="r")
    nu = model.addVars(T, n_states, vtype=GRB.CONTINUOUS, name="nu")
    alpha = model.addVars(T, n_features, vtype=GRB.CONTINUOUS, name="alpha")
    
    # Objective: Minimize sum of infinity norms (L∞ norm)
    # model.setObjective(gp.quicksum(( (alpha[t, i] - alpha[t - 1, i]) ** 2) / (2 * sigmas[i] ** 2) for i in range(n_features) for t in range(1, T)), GRB.MINIMIZE)
    model.setObjective(0, GRB.MINIMIZE)
    # model.setParam(GRB.Param.NonConvex, 2)
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

    # model.setParam('IterationLimit', 2e6)  # Increase iteration limit
    model.setParam('OutputFlag', 1)  # Enable detailed output
    model.optimize()
    model.computeIIS()
    model.write("model.ilp")

    return extract_solution(model, r, nu, alpha, T, n_states, n_actions, n_features)




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
