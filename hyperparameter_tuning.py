# Code to perform hyperparameter tuning

import numpy as np
from collections import Counter

from training import run_trial, load_data, split_scale, split_scale_even

# Load datset
X,y = load_data('forest_cover.npz')
# Preprocess data and split into train and test sets
X_train, X_test, y_train, y_test = split_scale_even(X,y)

# Define hyperparameter grid
N_values = [32,64,128]
L_values = [1e-3, 1e-4, 1e-5]
F_values = ['sigmoid','relu']

results = []

# Grid search for hyperparameter tuning.
trial_id = 0  # ID for each trial
for N in N_values:
    for L in L_values:
        for F in F_values:
            config = {'N': N, 'L': L, 'F': F}
            result = run_trial(X_train, y_train, X_test, y_test, config, trial_id)
            results.append(result)
            # Results saved at each trial in case of interruption
            np.savez('tuning_results.npz', results=results)
            trial_id += 1