# Functions for model training

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    """
    Load the dataset from a .npz file.

    Args:
        data_path (str): Path to the dataset file.

    Returns:
        tuple: Feature matrix and target vector (X, y).
    """

    data = np.load(data_path)
    X, y = data['X'], data['y']
    return X, y

def split_scale(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets, and scale the features.

    Args:
        data_path (str): Path to the dataset file.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to .2.
        random_state (int): Seed for reproducibility. Default to 42.

    Returns:
        tuple: Scaled training and testing features and labels (X_train, X_test, y_train, y_test).
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def split_scale_even(X, y, random_state=42, percent=0.8):
    """
    Balance the dataset by taking an equal number of samples from each class.

    This function computes the count for each class as a percent of the count of the least
    frequent class, then randomly samples that number of instances from each class.
    The function then splits the data into training and testing sets, and scales the features.
    The training set will contain the sampled instances, while the testing set will contain
    the remaining instances.

    Args:
        X (numpy.ndarray): Feature matrix.
        y (numpy.ndarray): Target vector.
        random_state (int, optional): Seed for reproducibility. Defaults to 42.
        percent (float, optional): Proportion of samples in least frequent class to take from each class.
            Defaults to 0.8.

    Returns:
        tuple: Balanced training and testing feature matrices and target vectors 
            (X_train, y_train, X_test, y_test).
    """
    np.random.seed(random_state)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    min_count = np.min(class_counts)
    count = int(percent * min_count)

    train_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, count, replace=False)
        train_indices.extend(sampled_indices)

    train_indices = np.array(train_indices)
    test_indices = np.setdiff1d(np.arange(len(y)), train_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def save_model_and_history(model, history, config, trial_id):
    """
    Save the trained model and its training history.

    Args:
        model (tensorflow.keras.Model): Trained model.
        history (tensorflow.keras.callbacks.History): Training history.
        config (dict): Configuration dictionary.
        trial_id (int): Unique identifier for the trial.

    Returns:
        None
    """
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/model_trial_{trial_id}.keras"
    history_path = f"saved_models/history_trial_{trial_id}.json"

    # Save the model
    model.save(model_path)

    # Save the training history
    with open(history_path, 'w') as f:
        json.dump({
            'config': config,
            'history': history.history
        }, f)

    print(f"Saved model to {model_path} and history to {history_path}")

def run_trial(X_train, y_train, X_test, y_test, config, trial_id):
    """
    Train and evaluate a neural network model with the given configuration.

    Uses 300 max epochs and early stopping with a patience of 3 epochs.
    
    Args:
        X_train (numpy.ndarray): Training feature matrix.
        y_train (numpy.ndarray): Training target vector.
        X_test (numpy.ndarray): Testing feature matrix.
        y_test (numpy.ndarray): Testing target vector.
        config (dict): Dictionary containing hyperparameters:
            - 'N': Number of neurons in the first layer.
            - 'F': Activation function.
            - 'L': Learning rate.
        trial_id (int): Unique identifier for the trial.

    Returns:
        dict: A dictionary containing the configuration, test loss, test accuracy,
            training loss, training accuracy, validation loss, and validation accuracy.
    """
    print(f"Running trial {trial_id}: {config}")

    # Start with N neurons in the first layer and halve until reaching the number of classes
    layers = [Dense(config['N'], activation=config['F'])]
    neurons = config['N'] // 2
    while neurons >= len(np.unique(y_train)):
        layers.append(Dense(neurons, activation=config['F']))
        neurons //= 2
    layers.append(Dense(len(np.unique(y_train)), activation='softmax'))

    model = Sequential([Input(shape=(X_train.shape[1],)), *layers])

    optimizer = Adam(learning_rate=config['L'])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=128,
        validation_split=0.20,
        callbacks=[early_stopping]
    )

    # Save the model and training history
    save_model_and_history(model, history, config, trial_id)

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    result = {
        'config': config,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'train_loss': history.history['loss'][-1],
        'train_acc': history.history['accuracy'][-1],
        'val_loss': history.history['val_loss'][-1],
        'val_acc': history.history['val_accuracy'][-1]
    }

    print(f"Finished trial {trial_id}: {config}, "
          f"Test Accuracy: {test_acc:.4f}, "
          f"Train Accuracy: {history.history['accuracy'][-1]:.4f}, "
          f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    return result