# Code to plot the confusion matrices for the two additional scenarios:
    # 1. The unbalanced dataset
    # 2. The reduced-sized balanced dataset
# Before running this code, the additional_scenarios.py file should be run.

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from training import split_scale, load_data, split_scale_even
from visualization import plot_confusion_matrix, plot_confusion_matrix_percentage

## For the unbalanced dataset --

# Load results from the unbalanced dataset
model_filename_unbalanced = "saved_models/model_trial_18.keras"
model_unbalanced = tf.keras.models.load_model(model_filename_unbalanced)

# Load the dataset
X, y = load_data('forest_cover.npz')
X_train, X_test, y_train, y_test = split_scale(X,y)

# Make sure the 'figs' directory exists
os.makedirs('figs', exist_ok=True)

# Plot confusion matrix 
filename = "figs/" + str(18) + "_unbalanced_confusion_matrix.png"
filename_percentage = "figs/" + str(18) + "_unbalanced_confusion_matrix_percentage.png"
plot_confusion_matrix(model_unbalanced, X_test, y_test, filename=filename)
plot_confusion_matrix_percentage(model_unbalanced, X_test, y_test, filename=filename_percentage)

## For the reduced dataset --

# Load results from the reduced dataset
model_filename_reduced = "saved_models/model_trial_19.keras"
model_reduced = tf.keras.models.load_model(model_filename_reduced)

# Load the dataset
X, y = load_data('forest_cover.npz')
X_train, X_test, y_train, y_test = split_scale_even(X,y,42,0.6)

# Make sure the 'figs' directory exists
os.makedirs('figs', exist_ok=True)

# Plot confusion matrix 
filename = "figs/" + str(18) + "_reduced_confusion_matrix.png"
filename_percentage = "figs/" + str(18) + "_reduced_confusion_matrix_percentage.png"
plot_confusion_matrix(model_reduced, X_test, y_test, filename=filename)
plot_confusion_matrix_percentage(model_reduced, X_test, y_test, filename=filename_percentage)

plt.show()