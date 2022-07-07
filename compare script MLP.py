#MLP

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
import time
from tqdm import tqdm
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MLP(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            #self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            #output = self.relu(output)
            return output

def train_MLP(X_train,y_train,n_epochs=20):
    model = MLP(X_train.shape[1], 20)
    X_trainTensor = torch.from_numpy(X_train).float()
    y_trainTensor = torch.from_numpy(y_train).float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
    
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(X_trainTensor)
        print(y_pred)
        # Compute Loss
        loss =  criterion(y_pred.squeeze(), y_trainTensor)
       
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()
    
    return model



def _comput_permutation(X, y, i, test_size=0.2, n_repeats=100):
    print(f"Processing trial:{i+1}")
    null_var = 20
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print("Fitting the model")
    model = train_MLP(X_train, y_train)
    model.eval()
    y_pred = model(torch.from_numpy(X_test).float()).squeeze().detach().numpy()

    # z_statistic = np.empty((n_repeats, len(y_test), X_test.shape[1]))
    z_statistic = np.empty((n_repeats, len(y_test)))
    print("Computing the importance scores")
    for perm in range(n_repeats):
        # print(f"Processing: {perm}")
        # for p_col in range(X_test.shape[1]):
        current_X = X_test.copy()
        np.random.shuffle(current_X[:, null_var])
        current_y_pred = model(torch.from_numpy(current_X).float()).squeeze().detach().numpy()
        print(current_y_pred.shape)
        z_statistic[perm, :] = (y_test - current_y_pred) ** 2 - \
            (y_test - y_pred) ** 2
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
    model = train_MLP(X_train, y_train)
    model.eval()
    y_pred = model(torch.from_numpy(X_test).float()).squeeze().detach().numpy()

    # z_statistic = np.empty((n_repeats, len(y_test), X_test.shape[1]))
    z_statistic = np.empty((n_repeats, len(y_test)))
    X_test_minus_idx = np.delete(np.copy(X_test), null_col, 1)
    regr = RandomForestRegressor(max_depth= 2)
    regr.fit(X_test_minus_idx, X_test[:, null_col])
    X_col_pred = regr.predict(X_test_minus_idx)    
    Res_col = X_test[:, null_col] - X_col_pred

    print("Computing the importance scores")
    for perm in range(n_repeats):
        np.random.shuffle(Res_col)
        X_col_new = X_col_pred + Res_col
        current_X = X_test.copy()
        current_X[:, null_col] = X_col_new
        current_y_pred = model(torch.from_numpy(current_X).float()).squeeze().detach().numpy()
        z_statistic[perm, :] = (y_test - current_y_pred) ** 2 - \
            (y_test - y_pred) ** 2
    imp_sc = np.mean(np.mean(z_statistic, axis=0), axis=0)
    std_sc = np.std(np.mean(z_statistic, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
    z_stat = imp_sc/std_sc
    return z_stat

########MAIN#################
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
z_stat_perm = parallel(delayed(_comput_permutation)(X, y, i,test_size=test_size, n_repeats=n_repeats) for i in (range(n_trials))) 
duration1 = time.time() - start

start = time.time()

z_stat_cond = parallel(delayed(_comput_conditional)(X, y, i,test_size=test_size, n_repeats=n_repeats) for i in (range(n_trials)))
duration2 = time.time() - start

fig, axs = plt.subplots(1, 2, constrained_layout=True)
fig.suptitle("Predictor =MLP \n Parameters:n="+str(n)+"/p="+str(p)+"/n_signal="+str(n_signal)
          +"\n n_trials="+str(n_trials)+"/snr="+str(snr)+"/rho="+str(rho)
          +"/n_repeats="+str(n_repeats), fontsize=14)

axs[0].hist(z_stat_perm)
mean_perm = np.mean(z_stat_perm)
std_perm = np.std(z_stat_perm)
axs[0].set_title('Permutation approach:\n Mean={:.2f}, Std={:.2f}'.format(mean_perm, std_perm)+"\n time="+"{:.2f}".format(duration1))

axs[1].hist(z_stat_cond)
mean_cond = np.mean(z_stat_cond)
std_cond = np.std(z_stat_cond)
axs[1].set_title('Conditional approach: \n Mean={:.2f}, Std={:.2f}'.format(mean_cond, std_cond)+"\n time="+"{:.2f}".format(duration2))

# Save the histogram
fig.savefig('hist_n{}_p{}_ntrial{}_rho{}_nrepeats{}.png'.format(n, p, n_trials, rho, n_repeats))

# Display the plot
plt.show()