# Functions for visualizing the results of the model training and evaluation.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test, filename=None):
    """
    Plots the confusion matrix for the model using the input dataset.
    
    Will save the plot to a file if filename is provided.
    Must call plt.show() after calling this function to display the plots.

    Args:
        model (tf.keras.Model): The TensorFlow model to evaluate.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): True labels for the test dataset.
        filename (str, optional): Path to save the confusion matrix plot.
    """
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    if filename:
        plt.savefig(filename)

def plot_confusion_matrix_percentage(model, X_test, y_test, filename=None):
    """
    Plots the confusion matrix for the model using the test dataset, with percentages instead of counts.
    
    Will save the plot to a file if filename is provided.
    Must call plt.show() after calling this function to display the plots.

    Args:
        model (tf.keras.Model): The TensorFlow model to evaluate.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): True labels for the test dataset.
        filename (str, optional): Path to save the confusion matrix plot.
    """

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=np.unique(y_test))
    disp.plot()
    
    if filename:
        plt.savefig(filename)

def tuning_bar_charts(results, key='test_acc', filename=None):
    """
    Visualize the results of hyperparameter tuning as a horizontal bar chart.

    Will save the plot to a file if filename is provided.
    Must call plt.show() after calling this function to display the plots.

    Args:
        results (list): List of dictionaries containing configurations and their test accuracies.
        key (str, optional): The key in the dictionary to visualize. Default is 'test_acc'.
        filename (str, optional): The file path to save the image. If None, the image is not saved.

    Returns:
        None
    """

    # Sort results by the activation function 'F' for readability
    results = sorted(results, key=lambda x: x['config']['F'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = []
    accuracies = []

    for res in results:
        cfg = res['config']
        label = f"N={cfg['N']}, F={cfg['F']}, L={cfg['L']}"
        labels.append(label)
        accuracies.append(res[key])

    ax.barh(labels, accuracies, color='skyblue')
    ax.set_xlabel('Test Accuracy')
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
