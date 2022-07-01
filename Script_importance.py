import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


list_z_stat = []
n = 1000
p = 50
n_signal = 20
snr = 4
rho = 0.8
n_trials = 100

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

for i in range(n_trials):
    print(f"Processing trial:{i+1}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("Fitting the model")
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)

    print("Computing the importance scores")
    result = permutation_importance(
        rf, X_test, y_test, n_repeats=100, n_jobs=10
    )
    z_stat = result['importances_mean'] / result['importances_std']
    list_z_stat.append(z_stat[20])

# Plot the histogram
plt.hist(list_z_stat)

# Save the histogram
plt.savefig('hist.png')

# Display the plot
plt.show()