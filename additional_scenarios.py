# Code to train a model on the unbalanced dataset and on a reduced-sized balanced dataset

import numpy as np
from collections import Counter

from training import run_trial, load_data, split_scale, split_scale_even

## Train on unbalanced dataset ---

# Load datset
X,y = load_data('forest_cover.npz')
# Preprocess data and split into train and test sets
X_train, X_test, y_train, y_test = split_scale(X,y)

config = {'N': 32, 'L': 1e-3, 'F': 'sigmoid'}
trial_id = 18
results = run_trial(X_train, y_train, X_test, y_test, config, 18)
np.savez('training_result_unbalanced.npz', results=results)

## Train on reduced-sized balanced dataset ---

X,y = load_data('forest_cover.npz')
# Preprocess data and split into train and test sets
X_train, X_test, y_train, y_test = split_scale_even(X,y,42,0.6)

config = {'N': 32, 'L': 1e-3, 'F': 'sigmoid'}
trial_id = 19
results = run_trial(X_train, y_train, X_test, y_test, config, 19)
np.savez('training_result_reduced.npz', results=results)
