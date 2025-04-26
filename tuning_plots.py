# Code to load saved hyperparameter tuning results and visualize them using bar charts.
# Before running this code, the hyperparameter_tuning.py file should be run.

import os
import numpy as np
import matplotlib.pyplot as plt
from visualization import tuning_bar_charts

# Make sure the 'figs' directory exists
os.makedirs('figs', exist_ok=True)

# Load the results from the hyperparameter tuning
data = np.load('tuning_results.npz', allow_pickle=True)
results = data['results']

# Plot result on barcharts
tuning_bar_charts(results,key='test_acc', filename='figs/test_acc.png')
tuning_bar_charts(results, key='train_acc', filename='figs/train_acc.png')
tuning_bar_charts(results,key='val_acc', filename='figs/val_acc.png')

plt.show()