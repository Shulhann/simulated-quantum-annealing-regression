import numpy as np
import neal

# Features: [Weight, Gear Count, Brand Popularity]
X = np.array([
    [10, 18, 7],  # Bicycle 1
    [12, 21, 8],  # Bicycle 2
    [9, 15, 6],   # Bicycle 3
    [14, 20, 9],  # Bicycle 4
    [11, 22, 10]  # Bicycle 5
])

# Prices of the bicycles
y = np.array([500, 700, 450, 800, 650])  # Corresponding prices

# Number of observations (bicycles)
n_samples = len(y)
n_features = X.shape[1]  # Number of features

# QUBO coefficients initialization
Q = np.zeros((n_features + 1, n_features + 1))  # Include intercept term

# Construct QUBO matrix for linear regression
for i in range(n_samples):
    for j in range(n_features):
        if i == j:
            Q[j][j] += -2 * y[i] * X[i, j]  # diagonal elements for each feature
        elif i < j:
            Q[j][i] += X[i, j] * X[i, j]  # off-diagonal elements

# Add intercept term
for i in range(n_samples):
    Q[n_features][n_features] += -2 * y[i]  # Set intercept coefficient
    for j in range(n_features):
        Q[n_features][j] += -2 * y[i] * X[i, j]  # Link intercept to features
        Q[j][n_features] = Q[n_features][j]  # Symmetric entry

# Prepare the QUBO dictionary format
Q_dict = {(i, j): Q[i][j] for i in range(n_features + 1) for j in range(n_features + 1) if i != j}

# Use the Neal simulator to solve the QUBO problem
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(Q_dict, num_reads=1000)

# Extract the best solution
best_solution = sampleset.first.sample

# Extract coefficients (including intercept)
coefficients = [best_solution[i] for i in range(n_features + 1)]
print("Estimated coefficients:", coefficients)
