"""HDF5 loader for downscaled MNIST PCA dataset."""

import os
import h5py
import numpy as np


def get_dataset(d: int):
    """
    Load downscaled MNIST PCA dataset from disk.

    Args:
        d: PCA dimension (2..20).

    Returns:
        x_train, x_test, y_train, y_test arrays.
    """
    assert type(d) is int and d >= 2 and d <= 20, f"Invalid parameter d: {d}"

    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # path of the current .py file
    file_path = os.path.join(base_path, "downscaled-mnist", "downscaled-mnist.h5")
    with h5py.File(file_path, "r") as f:
        inputs_train_group = f["train"][str(d)]["inputs"]
        inputs_test_group = f["test"][str(d)]["inputs"]
        labels_train_group = f["train"][str(d)]["labels"]
        labels_test_group = f["test"][str(d)]["labels"]

        def load_data(inputs_group, labels_group):
            x_data = []
            y_data = []
            for i in range(len(inputs_group)):
                sample = [float(inputs_group[str(i)][str(dim)][()]) for dim in range(d)]
                x_data.append(sample)
                y_data.append(labels_group[str(i)][()])
            return np.array(x_data), np.array(y_data)

        x_train, y_train = load_data(inputs_train_group, labels_train_group)
        x_test, y_test = load_data(inputs_test_group, labels_test_group)

    return x_train, x_test, y_train, y_test
