"""HDF5 loader for hidden manifold dataset."""

import os
import h5py
import numpy as np


def get_dataset(d: int, m: int):
    """
    Load hidden manifold dataset from disk.

    Args:
        d: Input dimension (2..20).
        m: Manifold dimension (2..20).

    Returns:
        x_train, x_test, y_train, y_test arrays.
    """
    assert type(d) is int and d >= 2 and d <= 20, f"Invalid parameter d: {d}"
    assert type(m) is int and m >= 2 and m <= 20, f"Invalid parameter m: {m}"

    base_path = os.path.dirname(
        os.path.abspath(__file__)
    )  # path of the current .py file
    file_path = os.path.join(base_path, "hidden-manifold", "hidden-manifold.h5")
    with h5py.File(file_path, "r") as f:
        if m == 6:
            inputs_train_group = f["diff_train"][str(d)]["inputs"]
            inputs_test_group = f["diff_test"][str(d)]["inputs"]
            labels_train_group = f["diff_train"][str(d)]["labels"]
            labels_test_group = f["diff_test"][str(d)]["labels"]
        elif d == 10:
            inputs_train_group = f["train"][str(m)]["inputs"]
            inputs_test_group = f["test"][str(m)]["inputs"]
            labels_train_group = f["train"][str(m)]["labels"]
            labels_test_group = f["test"][str(m)]["labels"]
        else:
            raise Exception(
                f"Invalid combination of parameters d,m: {d},{m}\nEither m=6 or d=10 is needed"
            )

        def load_data(inputs_group, labels_group):
            x_data = []
            y_data = []
            for i in range(len(inputs_group)):
                sample_keys = sorted([int(k) for k in inputs_group[str(i)].keys()])
                sample = [
                    float(inputs_group[str(i)][str(dim)][()]) for dim in sample_keys
                ]
                x_data.append(sample)
                y_data.append(labels_group[str(i)][()])
            return np.array(x_data), np.array(y_data)

        x_train, y_train = load_data(inputs_train_group, labels_train_group)
        x_test, y_test = load_data(inputs_test_group, labels_test_group)

    return x_train, x_test, y_train, y_test
