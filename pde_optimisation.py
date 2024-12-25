from docplex.mp.model import Model
import numpy as np
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Parameters
M = 6  # Number of clients
N = 3  # Number of potential sites
np.random.seed(42)

# Randomly generated data
capacities = np.random.randint(50, 100, size=N)  # Capacity q_j
construction_costs = np.random.randint(500, 1000, size=N)  # Construction cost c_j
revenues = np.random.randint(10, 50, size=(M, N))  # Revenue r_ij
base_demands = np.random.randint(10, 30, size=M)  # Base demands d_i

# Define delta values and probabilities
delta_values = [5, 20, 40]
p_s = [0.2, 0.45, 0.35]  # Probabilities

# Function to generate scenarios
def generate_scenarios(delta):
    variations = np.array([1 - delta / 100, 1, 1 + delta / 100])
    scenarios = np.outer(base_demands, variations).T
    return scenarios

# 3. Forme déterministe équivalente (PDE)
def solve_pde(scenarios):
    from itertools import product
    scenario_demands = np.array(list(product(*scenarios.T)))  # All scenarios
    S = len(scenario_demands)
    prob_scenario = [np.prod([p_s[k] for k in idx]) for idx in product([0, 1, 2], repeat=M)]

    mdl = Model(name='PDE_Optimization')

    # Decision variables
    y = mdl.binary_var_list(N, name='y')
    z = mdl.continuous_var_cube(M, N, S, name='z')
    x = mdl.binary_var_cube(M, N, S, name='x')

    # Objective
    mdl.maximize(
        mdl.sum(prob_scenario[s] * 
                (mdl.sum(revenues[i][j] * z[i, j, s] for i in range(M) for j in range(N)) -
                 mdl.sum(construction_costs[j] * y[j] for j in range(N))) for s in range(S))
    )

    # Constraints
    for s in range(S):
        for j in range(N):
            mdl.add_constraint(mdl.sum(z[i, j, s] for i in range(M)) <= capacities[j] * y[j])
        for i in range(M):
            mdl.add_constraint(mdl.sum(z[i, j, s] for j in range(N)) >= scenario_demands[s, i])
            mdl.add_constraint(mdl.sum(x[i, j, s] for j in range(N)) == 1)
            for j in range(N):
                mdl.add_constraint(z[i, j, s] <= scenario_demands[s, i] * x[i, j, s])

    solution = mdl.solve(log_output=False)
    return solution.objective_value if solution else 0

# 4. Approches intuitives
def calculate_ws_eev(scenarios):
    ws_values = []
    eev_values = []

    # Wait & See (WS)
    for scenario in scenarios.T:
        mdl_ws = Model(name='WS_Model')
        z_ws = mdl_ws.continuous_var_matrix(M, N, name='z_ws')
        y_ws = mdl_ws.binary_var_list(N, name='y_ws')

        # Objective
        mdl_ws.maximize(
            mdl_ws.sum(revenues[i][j] * z_ws[i, j] for i in range(M) for j in range(N)) -
            mdl_ws.sum(construction_costs[j] * y_ws[j] for j in range(N))
        )

        # Constraints
        for j in range(N):
            mdl_ws.add_constraint(mdl_ws.sum(z_ws[i, j] for i in range(M)) <= capacities[j] * y_ws[j])
        for i in range(M):
            mdl_ws.add_constraint(mdl_ws.sum(z_ws[i, j] for j in range(N)) >= scenario[i])

        solution_ws = mdl_ws.solve(log_output=False)
        ws_values.append(solution_ws.objective_value if solution_ws else 0)

    # Estimated Expected Value (EEV)
    avg_demand = np.mean(scenarios, axis=0)
    mdl_eev = Model(name='EEV_Model')
    z_eev = mdl_eev.continuous_var_matrix(M, N, name='z_eev')
    y_eev = mdl_eev.binary_var_list(N, name='y_eev')

    # Objective
    mdl_eev.maximize(
        mdl_eev.sum(revenues[i][j] * z_eev[i, j] for i in range(M) for j in range(N)) -
        mdl_eev.sum(construction_costs[j] * y_eev[j] for j in range(N))
    )

    # Constraints
    for j in range(N):
        mdl_eev.add_constraint(mdl_eev.sum(z_eev[i, j] for i in range(M)) <= capacities[j] * y_eev[j])
    for i in range(M):
        mdl_eev.add_constraint(mdl_eev.sum(z_eev[i, j] for j in range(N)) >= avg_demand[i])

    solution_eev = mdl_eev.solve(log_output=False)
    eev_values.append(solution_eev.objective_value if solution_eev else 0)

    return ws_values, eev_values

# 5. Mesures d’efficacité
results = []
for delta in delta_values:
    scenarios = generate_scenarios(delta)

    # Solve PDE
    stochastic_value = solve_pde(scenarios)

    # Expected Perfect Information (EPI)
    epi_profits = []
    for scenario in scenarios.T:
        mdl_epi = Model(name='EPI_Model')
        z_epi = mdl_epi.continuous_var_matrix(M, N, name='z_epi')
        y_epi = mdl_epi.binary_var_list(N, name='y_epi')

        # Objective
        mdl_epi.maximize(
            mdl_epi.sum(revenues[i][j] * z_epi[i, j] for i in range(M) for j in range(N)) -
            mdl_epi.sum(construction_costs[j] * y_epi[j] for j in range(N))
        )

        # Constraints
        for j in range(N):
            mdl_epi.add_constraint(mdl_epi.sum(z_epi[i, j] for i in range(M)) <= capacities[j] * y_epi[j])
        for i in range(M):
            mdl_epi.add_constraint(mdl_epi.sum(z_epi[i, j] for j in range(N)) >= scenario[i])

        epi_solution = mdl_epi.solve(log_output=False)
        epi_profits.append(epi_solution.objective_value if epi_solution else 0)

    expected_epi = np.mean(epi_profits)

    # Approches intuitives
    ws_values, eev_values = calculate_ws_eev(scenarios)

    # Store results
    results.append((delta, stochastic_value, expected_epi, ws_values, eev_values))

# Display results
for result in results:
    delta, stochastic_value, expected_epi, ws_values, eev_values = result
    logging.info(f"Delta: {delta}%")
    logging.info(f"Stochastic Value: {stochastic_value}")
    logging.info(f"Expected Perfect Information (EPI): {expected_epi}")
    logging.info(f"Efficiency: {100 * (stochastic_value - expected_epi) / expected_epi:.2f}%\n")
    logging.info(f"Wait & See Values: {ws_values}")
    logging.info(f"EEV Values: {eev_values}\n")
