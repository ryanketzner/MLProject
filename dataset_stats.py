# Function for displaying dataset statistics

import numpy as np
from collections import Counter

def compute_dataset_statistics(X, y):
    """
    Compute and display statistics about the dataset.

    Args:
        X (numpy.ndarray): Feature matrix of shape (num_samples, num_features).
        y (numpy.ndarray): Target vector of shape (num_samples,).

    Returns:
        None
    """
    num_samples, num_features = X.shape
    num_classes = len(np.unique(y))
    class_distribution = Counter(y)

    print(f"Samples: {num_samples}, Features: {num_features}, Classes: {num_classes}")
    for cls, count in class_distribution.items():
        print(f"Class {cls}: {count}")
