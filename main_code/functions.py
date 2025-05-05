
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def get_data_dir():
    try:
        # Assumes the script/notebook is in the 'main_code' directory
        script_dir = os.path.dirname(__file__)
    except NameError:
        # Fallback if __file__ is not defined (common in notebooks/interactive)
        # Assumes the current working directory is 'main_code' or 'OPT_hw'
        # If cwd is OPT_hw, remove the '..' in the join below.
        # If cwd is main_code, this structure works.
        script_dir = os.getcwd()
        # If your notebook/script is actually in OPT_hw, adjust the path construction:
        # data_dir = os.path.abspath(os.path.join(script_dir, 'data', 'raw', 'archive'))


    # Construct the path relative to the parent directory of script_dir
    # Goes up one level from script_dir (main_code) to OPT_hw, then into data/raw/archive
    data_dir_relative = os.path.join(script_dir, '..', 'data', 'raw')

    # Get the absolute, normalized path (resolves '..')
    data_dir = os.path.abspath(data_dir_relative)

    print(f"Data directory path: {data_dir}")
    return data_dir

# load training.csv from the data directory
def load_higgs_data(data_dir_path):
    """
    Loads the training data from a CSV file.

    Args:
        data_dir_path (str): Path to the directory containing the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    # Construct the full path to the CSV file
    csv_file_path = os.path.join(data_dir_path, 'training.csv')

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Display the first few rows of the DataFrame
    print(f"Loaded data from {csv_file_path}")
    print(df.head())

    return df


def generate_semi_supervised_data(n_samples=10000, noise=0.15, labeled_proportion=0.1, random_state=42):
    """
    Generates a semi-supervised dataset with two classes (-1 and 1).

    Args:
        n_samples (int): Total number of samples to generate.
        noise (float): Standard deviation of Gaussian noise added to the data.
        labeled_proportion (float): Proportion of the data to be labeled (between 0 and 1).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - X_labeled (np.ndarray): Features of the labeled data.
            - y_labeled (np.ndarray): Labels (-1 or 1) of the labeled data.
            - X_unlabeled (np.ndarray): Features of the unlabeled data.
    """
    # Generate a two-moon dataset
    #X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    X, y = make_blobs(n_samples=n_samples, centers=2, cluster_std=noise, random_state=random_state)
    # Convert labels from {0, 1} to {-1, 1}
    y = y * 2 - 1

    # Split into labeled and unlabeled sets
    n_labeled = int(n_samples * labeled_proportion)

    # Ensure at least one sample per class in the labeled set if possible
    # This simple split might not guarantee class balance in the small labeled set
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X, y, train_size=n_labeled, random_state=random_state, stratify=y
    )

    # For the unlabeled set, we typically only use the features (X_unlabeled)
    # The y_unlabeled is usually discarded or ignored in semi-supervised settings,
    # but we return it here for potential analysis or alternative approaches.
    # In many algorithms, you'd just pass X_unlabeled.

    print(f"Generated dataset with {n_samples} total samples.")
    print(f"Labeled samples: {len(X_labeled)}")
    print(f"Unlabeled samples: {len(X_unlabeled)}")
    print(f"Labeled class distribution: {np.unique(y_labeled, return_counts=True)}")

    return X_labeled, y_labeled, X_unlabeled

