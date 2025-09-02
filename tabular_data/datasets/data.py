"""
Main data handling file: Tabular data fetching and preparation
"""

from datasets.downscaled_mnist_pca_loading import get_dataset as get_mnist_pca
from datasets.mnist import generate_mnist
from datasets.hidden_manifold_loading import get_dataset as get_hm
from datasets.hidden_manifold import generate_hidden_manifold_model
from datasets.two_curves_loading import get_dataset as get_two_curves
from datasets.two_curves import generate_two_curves
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import os


def download_datasets():
    print("Downloading datasets...")
    qml.data.load('other', name='downscaled-mnist')
    qml.data.load("other", name="hidden-manifold")
    qml.data.load("other", name="two-curves")
    print("Done.")
    return


def get_data(data: str, loading="fetch", arg1=None, arg2=None, random_state=42):
    if data == "downscaled_mnist_pca":
        if loading == "fetch":
            x_train, x_test, y_train, y_test = get_mnist_pca(arg1)
            return x_train, x_test, y_train.astype(np.int64), y_test.astype(np.int64)
        elif loading == "build":
            x_train, x_test, y_train, y_test = generate_mnist(3, 5, "pca", arg1)
            return x_train, x_test, y_train.astype(np.int64), y_test.astype(np.int64)
        else:
            raise ValueError(f"Unknown loading {loading}")

    elif data == "hidden_manifold":
        if loading == "fetch":
            x_train, x_test, y_train, y_test = get_hm(arg1, arg2)
            return x_train, x_test, y_train.astype(np.int64), y_test.astype(np.int64)
        elif loading == "build":
            x, y = generate_hidden_manifold_model(300, arg1, arg2)
            return train_test_split(x, y, test_size=60, train_size=240, random_state=random_state)
        else:
            raise ValueError(f"Unknown loading {loading}")

    elif data == "two_curves":
        if loading == "fetch":
            x_train, x_test, y_train, y_test = get_two_curves(arg1, arg2)
            return x_train, x_test, y_train.astype(np.int64), y_test.astype(np.int64)
        elif loading == "build":
            x, y = generate_two_curves(300, arg1, arg2, 1/(2*arg2), 0.01)
            return train_test_split(x, y, test_size=60, train_size=240, random_state=random_state)
        else:
            raise ValueError(f"Unknown loading {loading}")
    else:
        raise ValueError(f"Unknown dataset: {data}")


def get_summary(x_train, x_test, y_train, y_test, path, dataset):
    """
    Generates a statistical summary of train and test datasets and saves it to a file.
    Args:
        x_train, x_test: numpy arrays or pandas DataFrames of features
        y_train, y_test: numpy arrays or pandas Series of labels
        path: folder path where 'dataset_summary.txt' will be saved
    """
    lines = []

    # Convert to DataFrame if not already
    if not isinstance(x_train, pd.DataFrame):
        x_train_df = pd.DataFrame(x_train)
    else:
        x_train_df = x_train

    if not isinstance(x_test, pd.DataFrame):
        x_test_df = pd.DataFrame(x_test)
    else:
        x_test_df = x_test

    # Basic shape info
    lines.append(f"x_train shape: {x_train_df.shape}")
    lines.append(f"x_test shape:  {x_test_df.shape}")
    lines.append(f"y_train shape: {y_train.shape}")
    lines.append(f"y_test shape:  {y_test.shape}\n")

    # Feature statistics
    lines.append("=== x_train statistics ===")
    lines.append(str(x_train_df.describe(include='all')) + "\n")

    lines.append("=== x_test statistics ===")
    lines.append(str(x_test_df.describe(include='all')) + "\n")

    # Label statistics
    lines.append("=== y_train statistics ===")
    if isinstance(y_train, np.ndarray):
        y_train_series = pd.Series(y_train)
    else:
        y_train_series = y_train
    lines.append(str(y_train_series.describe()) + "\n")
    lines.append("Value counts:\n" + str(y_train_series.value_counts()) + "\n")

    lines.append("=== y_test statistics ===")
    if isinstance(y_test, np.ndarray):
        y_test_series = pd.Series(y_test)
    else:
        y_test_series = y_test
    lines.append(str(y_test_series.describe()) + "\n")
    lines.append("Value counts:\n" + str(y_test_series.value_counts()) + "\n")

    # Save to file
    os.makedirs(path, exist_ok=True)
    summary_file = path + f"{dataset}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("\n".join(lines))

    print(f"Dataset summary saved to {summary_file}")
    return


def preprocess_data(x_train, x_test, scaling):
    """
    Preprocess the data by applying the specified scaling technique.

    Args:
        x_train (array-like): Feature matrix for the training set (n_samples x n_features).
        x_test (array-like): Feature matrix for the test set (n_samples x n_features).
        y_train (array-like): Label vector for the training set (n_samples,).
        y_test (array-like): Label vector for the test set (n_samples,).
        scaling (str): The scaling technique to apply. Options are:
            - "standardize": Standardize features (zero mean, unit variance).
            - "minmax": Min-max scale features (range [0, 1]).
            - "arctan": Apply arctangent scaling to features.

    Returns:
        tuple: Scaled feature matrices (x_train, x_test) and label vectors (y_train, y_test).
    """
    # Assert valid scaling method
    assert scaling in ["standardize", "minmax", "arctan", "none", None], f"Invalid scaling method: {scaling}"

    # Standardization (zero mean, unit variance)
    if scaling == "standardize":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Min-Max scaling (scales to the range [0, 1])
    elif scaling == "minmax":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    # Arctangent scaling (applies arctan to each feature value)
    elif scaling == "arctan":
        x_train = np.arctan(x_train)
        x_test = np.arctan(x_test)

    return x_train, x_test


def preprocess_labels(y_train, y_test, treatment='0-1'):
    if treatment == '0-1':
        # Only apply conversion if necessary
        if set(np.unique(y_train)) <= {-1, 1}:
            y_train = (y_train + 1) // 2
        if set(np.unique(y_test)) <= {-1, 1}:
            y_test = (y_test + 1) // 2
    elif treatment == 'none' or treatment is None or treatment == 'None':
        y_train = y_train
        y_test = y_test
    elif treatment == 'q_kernel':
        y_train = y_train
        y_test = y_test
    else:
        raise NotImplementedError(f'Invalid labels treatment: {treatment}')
    return y_train, y_test


def convert_array_to_tensor(x_train, x_test, y_train, y_test, dtype=torch.float32, labels_treatment='0-1'):
    """
    Converts train/test features and labels to PyTorch tensors.
    Args:
        x_train, x_test: numpy arrays or pandas DataFrames of features
        y_train, y_test: numpy arrays or pandas Series of labels
        dtype: torch dtype for features (default: float32)
    Returns:
        x_train_t, x_test_t, y_train_t, y_test_t: PyTorch tensors
    """
    # Convert to tensors
    x_train_t = torch.tensor(x_train, dtype=dtype)
    x_test_t = torch.tensor(x_test, dtype=dtype)

    # For labels, use float for regression, long for classification
    if np.issubdtype(y_train.dtype, np.floating) or labels_treatment == 'q_kernel':
        y_dtype = torch.float32
    else:
        y_dtype = torch.long

    y_train_t = torch.tensor(y_train, dtype=y_dtype)
    y_test_t = torch.tensor(y_test, dtype=y_dtype)

    return x_train_t, x_test_t, y_train_t, y_test_t


def convert_tensor_to_loader(x, y, batch_size=32, shuffle=True):
    """
    Converts feature and label tensors into a PyTorch DataLoader.

    Args:
        x: feature tensor (torch.Tensor)
        y: label tensor (torch.Tensor)
        batch_size: number of samples per batch
        shuffle: whether to shuffle the data each epoch

    Returns:
        DataLoader object
    """
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def subsample(x_train, x_test, y_train, y_test, num_train, num_test):
    # pick random indices
    train_idx = np.random.choice(len(x_train), num_train, replace=False)
    test_idx = np.random.choice(len(x_test), num_test, replace=False)

    # subset
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    return x_train, x_test, y_train, y_test