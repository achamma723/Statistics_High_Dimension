import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time


list_z_stat = []
n = 1000
p = 50
n_signal = 20
snr = 4
rho = 0.8
n_trials = 100
# Parallel options
n_jobs = -1
verbose = 0

# Create Correlation matrix with the toeplitz design
elements = np.repeat(rho, p)
reduction_combs = np.arange(p)
vector = np.array([x ** y for (x, y) in zip(elements, reduction_combs)])
cov_matrix = toeplitz(vector)

# Generation of the predictors
rng = np.random.RandomState(100)
X = rng.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)

# Random choice of the relevant variables
list_var = np.random.choice(p, n_signal, replace=False)
reorder_var = np.array([i for i in range(p) if i not in list_var])

# Reorder data matrix so that first n_signal predictors are the signal predictors
X = X[:, np.concatenate([list_var, reorder_var], axis=0)]

# Random choice of the coefficients for the signal
effectset = np.array([-0.5, -1, -2, -3, 0.5, 1, 2, 3])
betas = np.random.choice(effectset, n_signal)
prod_signal = np.dot(X[:, :n_signal], betas) 

# The magnitude for the gaussian additive noise
noise_magnitude = np.linalg.norm(prod_signal, ord=2) / (np.sqrt(n) * snr)
y =  prod_signal + noise_magnitude * np.random.normal(size=n)

# for p_col in range(p):
#     current_X = X.copy()
#     current_X = np.delete(current_X, p_col, 1)
#     rf_p = RandomForestRegressor(max_depth=2)
#     rf_p.fit(current_X, X[:, p_col])
#     residuals = X[:, p_col] - rf_p.predict()

def comput_statistic(X, y, i, test_size=0.2, n_repeats=100):
    print(f"Processing trial:{i+1}")
    null_col = 20

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Fitting the model")
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    # z_statistic = np.empty((n_repeats, len(y_test), X_test.shape[1]))
    z_statistic = np.empty((n_repeats, len(y_test)))
    X_test_minus_idx = np.delete(np.copy(X_test), null_col, 1)
    regr = RandomForestRegressor(max_depth=2)
    regr.fit(X_test_minus_idx, X_test[:, null_col])
    X_col_pred = regr.predict(X_test_minus_idx)    
    Res_col = X_test[:, null_col] - X_col_pred

    print("Computing the importance scores")
    for perm in range(n_repeats):
        np.random.shuffle(Res_col)
        X_col_new = X_col_pred + Res_col
        current_X = X_test.copy()
        current_X[:, null_col] = X_col_new
        z_statistic[perm, :] = (y_test - rf.predict(current_X)) ** 2 - \
            (y_test - rf.predict(X_test)) ** 2
    imp_sc = np.mean(np.mean(z_statistic, axis=0), axis=0)
    std_sc = np.std(np.mean(z_statistic, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
    z_stat = imp_sc/std_sc
    return z_stat


start = time.time()
parallel = Parallel(n_jobs=n_jobs, verbose=verbose)
list_z_stat = parallel(delayed(comput_statistic)(X, y, i, test_size=0.2, n_repeats=100)
    for i in range(n_trials)) 
print(f"Time elapsed:{time.time() - start}")

# Plot the histogram
plt.hist(list_z_stat)

# Save the histogram
plt.savefig('hist_cond_100.png')

# Display the plot
plt.show()
