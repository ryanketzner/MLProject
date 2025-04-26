# Code to plot the confusion matrix for the best model from the hyperparameter tuning.
# Before running this code, the hyperparameter_tuning.py file should be run.

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from training import split_scale, load_data, split_scale_even
from visualization import plot_confusion_matrix, plot_confusion_matrix_percentage

# Load the results from the hyperparameter tuning and
# find the best model based on validation accuracy.
data = np.load('tuning_results.npz', allow_pickle=True)
results = data['results']
idx, best_model = max(enumerate(results), key=lambda x: x[1]['val_acc'])
best_model_filename = "saved_models/model_trial_" + str(idx) + ".keras"
model = tf.keras.models.load_model(best_model_filename)

# Load the dataset
X, y = load_data('forest_cover.npz')
X_train, X_test, y_train, y_test = split_scale_even(X,y)

# Make sure the 'figs' directory exists
os.makedirs('figs', exist_ok=True)

# Plot confusion matrix for the best model
filename = "figs/" + str(idx) + "_confusion_matrix.png"
filename_percentage = "figs/" + str(idx) + "_confusion_matrix_percentage.png"
plot_confusion_matrix(model, X_test, y_test, filename=filename)
plot_confusion_matrix_percentage(model, X_test, y_test, filename=filename_percentage)

plt.show()