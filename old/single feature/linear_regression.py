import numpy as np
import neal

# Feature: [Weight]
X = np.array([10, 12, 9, 14, 11])  # Weight of each bicycle

# Prices of the bicycles
y = np.array([500, 700, 450, 800, 650])  # Corresponding prices

# Number of observations (bicycles)
n_samples = len(y)

# QUBO coefficients initialization
Q = np.zeros((2, 2))  # Include intercept and one feature

# Construct QUBO matrix for linear regression with one feature
for i in range(n_samples):
    # Diagonal element for the feature coefficient
    Q[0][0] += X[i] * X[i]
    Q[0][1] += -2 * X[i] * y[i]
    Q[1][0] = Q[0][1]  # Symmetric entry
    Q[1][1] += y[i] * y[i]

# Prepare the QUBO dictionary format
Q_dict = {(i, j): Q[i][j] for i in range(2) for j in range(2) if i != j}

# Use the Neal simulator to solve the QUBO problem
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(Q_dict, num_reads=1000)

# Extract the best solution
best_solution = sampleset.first.sample

# Extract coefficients (feature and intercept)
intercept = best_solution[1]
feature_coeff = best_solution[0]

print(f"Estimated intercept: {intercept}")
print(f"Estimated coefficient for weight: {feature_coeff}")