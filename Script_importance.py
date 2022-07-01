from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.linalg import toeplitz


n = 1000
p = 50
n_signal = 20
snr = 4
rho = 0.8

# Create Correlation matrix with the toeplitz design
elements = np.repeat(rho, p)
reduction_combs = np.arange(p)
vector = np.array([x ** y for (x, y) in zip(elements, reduction_combs)])
cov_matrix = toeplitz(vector)

# Generation of the predictors
X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov_matrix, size=n)

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

