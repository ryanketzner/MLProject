# Forest Cover Classification Project

This project aims to classify forest cover types using a neural network. It includes data preprocessing, model training, hyperparameter tuning, and visualization of results.

## Project Structure

The project contains the following non-executable files with functions used for the project:

1. **`dataset_stats.py`**  
   - **Description**: Includes a function to compute and display statistics about the dataset, such as the number of samples, features, and class distribution.  

2. **`training.py`**  
   - **Description**: Contains functions for loading data, preprocessing, splitting, scaling, and training the neural network.

3. **`visualization.py`**  
    - **Description**: Contains utility functions for plotting confusion matrices and bar charts.  

And the following files to execute the project.

1. **`hyperparameter_tuning.py`**  
   - **Description**: Performs grid search for hyperparameter tuning.  
   - **Output**: Saves tuning results dictionary in `tuning_results.npz` and trained models in the `saved_models/` directory.  

2. **`tuning_plots.py`**  
   - **Description**: Visualizes the results of hyperparameter tuning using bar charts.  
   - **Output**: Saves bar charts in the `figs/` directory.  

3. **`additional_scenarios.py`**  
   - **Description**: Trains models on two additional scenarios:  
     1. An unbalanced dataset.  
     2. A reduced-sized balanced dataset.  
   - **Output**: Saves results in `training_result_unbalanced.npz` and `training_result_reduced.npz`, and models in the `saved_models/` directory.  

4. **`confusion_plots.py`**  
   - **Description**: Plots confusion matrices for the best model from hyperparameter tuning experiment.
   - **Output**: Saves confusion matrix plots in the `figs/` directory.  
   - **Run Order**: Must be run after `hyperparameter_tuning.py`.

5. **`confusion_plots_additional.py`**  
   - **Description**: Plots confusion matrices for the models trained on the additional scenarios.  
   - **Output**: Saves confusion matrix plots in the `figs/` directory.  
   - **Run Order**: Must be run after `additional_scenarios.py`.

## Dependencies

To run this project, the following dependencies must be installed:

- Python 3.8+
- NumPy
- Matplotlib
- TensorFlow
- scikit-learn

These can be installed using:

```bash
pip install numpy matplotlib tensorflow scikit-learn
```

Additionally, the CoverType dataset must be downloaded into the project directory before running this code. The following python script can be used to download.

```
import numpy as np
from sklearn.datasets import fetch_covtype

def main():
    covtype = fetch_covtype()
    X = covtype.data
    y = covtype.target - 1  # Adjust to zero-index

    np.savez('forest_cover.npz', X=X, y=y)
    print("Dataset downloaded and saved as 'forest_cover.npz'.")

if __name__ == "__main__":
    main()
