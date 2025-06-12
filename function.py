# Consisting functions required for performing regression using quantum annealing

# Import necessary library
import neal
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures
from collections import defaultdict

import networkx as nx
from collections import defaultdict
from sklearn.linear_model import LinearRegression

from amplify import VariableGenerator
from amplify import Poly
from amplify import FixstarsClient, solve

# To set a number to negative or positive value
def multiplier(x):
    if x <= 2:
        return 1
    else:
        return -1

# Function to convert data to polynomial form
def polynomialForm(X, d, dim, precision, degree):
    poly = PolynomialFeatures(degree=degree)
    X = poly.fit_transform(X)
    d = len(X[0])
    dim = d * (2 * precision)

    return X, d, dim


# These function below is called if you want to solve using neal simulated annealing
# Generate QUBO matrix    
def generateQuboMatrix_neal(XtX, XtY, precision, d):
    Q = defaultdict(int)
    # First term, same weights
    for i in range(d):
        xii = XtX[i, i]
        for k in range(2 * precision):
            d1 = i * 2 * precision + k
            Q[(d1, d1)] += xii / pow(2, 2 * (k % precision))
            for l in range(k + 1, 2 * precision):
                d2 = i * 2 * precision + l
                Q[(d1, d2)] += 2 * xii / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)

    # First term, different weights
    for i in range(d):
        for j in range(i + 1, d):
            xij = XtX[i, j]
            for k in range(2 * precision):
                for l in range(2 * precision):
                    d1 = i * 2 * precision + k
                    d2 = j * 2 * precision + l
                    Q[(d1, d2)] += 2 * xij / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)


    # Second Term
    for i in range(d):
        xyi = XtY[i]
        for k in range(2 * precision):
            d1 = i * 2 * precision + k
            Q[(d1, d1)] -= 2 * xyi / pow(2, k % precision) * multiplier(k)

    return Q

# Perform quantum annealing    
def sampling_neal(Q):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=20, chain_strength=10, run_time=5)
    return sampleset

# Extract the best solution
def solve_neal(sampleset, dim, precision, d):
    distributions = []

    for sample, energy in sampleset.data(['sample', 'energy']):
        distributions.append(sample)

    sol_no = 1
    for di in distributions:
        wts = np.array([0.0 for i in range(d)])
        for x in range(dim):
            i = x // (2 * precision)
            k = x % (2 * precision)
            wts[i] += di[x] / pow(2, k % precision) * multiplier(k)
    
    return distributions, wts


# These function below is called if you want to solve using fixstar simulated annealing
# Generate QUBO matrix
def generateQuboMatrix_fixstar(XtX, XtY, dim, precision, d):
    gen = VariableGenerator()
    x = gen.array("Binary", dim) 
    model = Poly(0.0)
    
    # First term, same weights
    for i in range(d):
        xii = XtX[i, i]
        for k in range(2 * precision):
            d1 = i * 2 * precision + k
            # Diagonal term
            model += xii / pow(2, 2 * (k % precision)) * x[d1]
            
            # Off-diagonal terms
            for l in range(k + 1, 2 * precision):
                d2 = i * 2 * precision + l
                coeff = 2 * xii / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)
                model += coeff * x[d1] * x[d2]

    # First term, different weights
    for i in range(d):
        for j in range(i + 1, d):
            xij = XtX[i, j]
            for k in range(2 * precision):
                for l in range(2 * precision):
                    d1 = i * 2 * precision + k
                    d2 = j * 2 * precision + l
                    coeff = 2 * xij / pow(2, k % precision + l % precision) * multiplier(k) * multiplier(l)
                    model += coeff * x[d1] * x[d2]

    # Second Term
    for i in range(d):
        xyi = XtY[i]
        for k in range(2 * precision):
            d1 = i * 2 * precision + k
            coeff = -2 * xyi / pow(2, k % precision) * multiplier(k)
            model += coeff * x[d1]

    return model

# Perform simulated annealing
def sampling_fixstar(model):
    client = FixstarsClient()
    client.token = "AE/8S2qgkdm29vrDGF5d1VfqKFK80lSSJkz"
    client.parameters.timeout = 5000

    result = solve(model, client)
    result_best_values = result.best.values  

    return result_best_values

# Extract the best solution
def solve_fixstar(result_best_values, dim, precision, d):
    wts = np.array([0.0 for i in range(d)])
    distributions = [0] * dim
    idx = 0
    for key, val in result_best_values.items():
        distributions[idx] = val
        idx+=1

    for x in range(dim):
        i = x // (2 * precision)  # which weight this bit belongs to
        k = x % (2 * precision)   # position within this weight
        wts[i] += distributions[x] / pow(2, k % precision) * multiplier(k)

    return distributions, wts


# Additional functions (for testing purposes)
def append_to_excel_enhanced(excel_filename, df_new, check_columns):
    """
    Parameters:
    - excel_filename: Path to Excel file
    - df_new: New DataFrame to append
    - check_columns: List of columns to check for duplicates
                    e.g., ['Dataset_Name', 'Precision', 'Degree', 'Model', 'Dataset']
    """
    try:
        existing_df = pd.read_excel(excel_filename, sheet_name="Evaluation Metrics")
    except FileNotFoundError:
        df_new.to_excel(excel_filename, index=False, sheet_name="Evaluation Metrics")
        print("New Excel file created with the data.")
        return
    
    # Create a temporary merged DataFrame for comparison
    temp_df = existing_df[check_columns].merge(
        df_new[check_columns], 
        how='inner',
        on=check_columns
    )
    
    if not temp_df.empty:
        print("Duplicate entries found based on columns:", check_columns)
        print("The following combinations already exist:")
        print(temp_df.drop_duplicates())
        print("Skipping addition of duplicate data.")
        return
    
    combined_df = pd.concat([existing_df, df_new], ignore_index=True)
    combined_df.to_excel(excel_filename, index=False, sheet_name="Evaluation Metrics")
    print("Data successfully appended to Excel file.")