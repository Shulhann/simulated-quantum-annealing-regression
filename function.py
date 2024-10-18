# Consisting functions required for performing regression using quantum annealing

# Import necessary library
import neal
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import PolynomialFeatures

import networkx as nx
from collections import defaultdict
from sklearn.linear_model import LinearRegression

# To set a number to negative or positive value
def multiplier(x):
    if x <= 2:
        return 1
    else:
        return -1

# Generate QUBO matrix    
def generateQuboMatrix(XtX, XtY, precision, Q, d):
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

# Perform quantum annealing    
def sampling(Q):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=20, chain_strength=10)
    return sampleset

# Extract the best solution
def solve(sampleset, dim, precision, X,Y, d):
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
        if sol_no == 1:
            Y_pred = np.matmul(X, wts)
            err = mse(Y, Y_pred)
            sol_no += 1
    
    return distributions, wts, err

# Function to convert data to polynomial form
def polynomialForm(X, d, dim, precision):
    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)
    d = len(X[0])
    dim = d * (2 * precision)

    return X, d, dim
    
