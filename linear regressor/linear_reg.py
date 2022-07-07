#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 11:16:08 2022

@author: katerina
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from scipy.stats import norm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression



def _comput_permutation(X, y, i, test_size=0.2, n_repeats=100):
    print(f"Processing trial:{i+1}")
    null_var = 20
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print("Fitting the model")
    regressor = LinearRegression()
#    rf = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    # z_statistic = np.empty((n_repeats, len(y_test), X_test.shape[1]))
    z_statistic = np.empty((n_repeats, len(y_test)))
    print("Computing the importance scores")
    for perm in range(n_repeats):
        # print(f"Processing: {perm}")
        # for p_col in range(X_test.shape[1]):
        current_X = X_test.copy()
        np.random.shuffle(current_X[:, null_var])
        z_statistic[perm, :] = (y_test - regressor.predict(current_X)) ** 2 - \
            (y_test - regressor.predict(X_test)) ** 2
    imp_sc = np.mean(np.mean(z_statistic, axis=0), axis=0)
    std_sc = np.std(np.mean(z_statistic, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
    z_stat = imp_sc/std_sc
    return z_stat


def _comput_conditional(X, y, i, test_size=0.2, n_repeats=100):
    print(f"Processing trial:{i+1}")
    null_col = 20

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print("Fitting the model")
    regressor = LinearRegression()
#    rf = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    # z_statistic = np.empty((n_repeats, len(y_test), X_test.shape[1]))
    z_statistic = np.empty((n_repeats, len(y_test)))
    X_test_minus_idx = np.delete(np.copy(X_test), null_col, 1)
    
    regr = RandomForestRegressor()
    regr.fit(X_test_minus_idx, X_test[:, null_col])
    X_col_pred = regr.predict(X_test_minus_idx)    
    Res_col = X_test[:, null_col] - X_col_pred

    print("Computing the importance scores")
    for perm in range(n_repeats):
        np.random.shuffle(Res_col)
        X_col_new = X_col_pred + Res_col
        current_X = X_test.copy()
        current_X[:, null_col] = X_col_new
        z_statistic[perm, :] = (y_test - regressor.predict(current_X)) ** 2 - \
            (y_test - regressor.predict(X_test)) ** 2
    imp_sc = np.mean(np.mean(z_statistic, axis=0), axis=0)
    std_sc = np.std(np.mean(z_statistic, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
    z_stat = imp_sc/std_sc
    return z_stat

# Common configuration
## Random generator
seed = 100
# Number of samples
n = 1000
# Number of features
p = 50
# Number of relevant features
n_signal = 20
# Signal-Noise Ratio
snr = 4
# Degree of correlation
rho = 0.8
# Number of repetitions
n_trials = 1000
# Size of the test set (computation of the importance scores)
test_size=0.2
# Number of permutations (Computation of z-statistic)
n_repeats=100
## Parallel options
n_jobs = -1
verbose = 0

# Create Correlation matrix
cov_matrix = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        if i != j:
            if (i < 21) and (j < 21):
                cov_matrix[i, j] = rho
            else:
                cov_matrix[i, j] = 0
        else:
            cov_matrix[i, j] = 1

# Generation of the predictors
## Fix the seed 
rng = np.random.RandomState(seed)
X = rng.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)

# Random choice of the coefficients for the signal
effectset = np.array([-0.5, -1, -2, -3, 0.5, 1, 2, 3])
betas = rng.choice(effectset, n_signal)
prod_signal = np.dot(X[:, :n_signal], betas) 

# The magnitude for the gaussian additive noise
noise_magnitude = np.linalg.norm(prod_signal, ord=2) / (np.sqrt(n) * snr)
y =  prod_signal + noise_magnitude * rng.normal(size=n)

start = time.time()
parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
z_stat_perm = parallel(delayed(_comput_permutation)(X, y, i,
    test_size=test_size, n_repeats=n_repeats)
    for i in range(n_trials)) 
z_stat_cond = parallel(delayed(_comput_conditional)(X, y, i,
    test_size=test_size, n_repeats=n_repeats)
    for i in range(n_trials))
duration = time.time() - start

fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle("Parameters:n="+str(n)+"/p="+str(p)+"/n_signal="+str(n_signal)
          +"\n n_trials="+str(n_trials)+"/snr="+str(snr)+"/rho="+str(rho)
          +"/n_repeats="+str(n_repeats)+"\n time="+"{:.2f}".format(duration), fontsize=14)

axs[0].hist(z_stat_perm)
mean_perm = np.mean(z_stat_perm)
std_perm = np.std(z_stat_perm)
axs[0].set_title('Permutation approach: Mean={:.2f}, Std={:.2f}'.format(mean_perm, std_perm))

axs[1].hist(z_stat_cond)
mean_cond = np.mean(z_stat_cond)
std_cond = np.std(z_stat_cond)
axs[1].set_title('Conditional approach: Mean={:.2f}, Std={:.2f}'.format(mean_cond, std_cond))

# Save the histogram
fig.savefig('hist_n{}_p{}_ntrial{}_rho{}_nrepeats{}.png'.format(n, p, n_trials, rho, n_repeats))

# Display the plot
plt.show()