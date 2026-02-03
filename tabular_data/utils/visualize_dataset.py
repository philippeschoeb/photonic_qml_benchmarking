import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.fetch_data import fetch_data
from datasets.data import get_data, preprocess_data, preprocess_labels
from registry import DATASET_BASE_NAMES

DEFAULT_RANDOM_STATE = 42
DEFAULT_PREPROCESSING = {
    "scaling": "minmax",
    "labels_treatment": "0_1",
}
DEFAULT_SAVE_DIR = os.path.join("results", "datasets")


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def visualize_dataset(
    dataset,
    pre_processed=False,
    random_state=DEFAULT_RANDOM_STATE,
    save_dir=DEFAULT_SAVE_DIR,
    show=False,
    **preprocess_overrides,
):
    """
    Visualize a 2D projection of a dataset.

    Args:
        dataset: Dataset name with optional args (e.g., "two_curves_2_5").
        pre_processed: If True, apply scaling/label treatment before plotting.
        random_state: Optional seed for subsampling in fetch_data.
        save_dir: Output directory for the PNG.
        show: If True, display the plot interactively.
        **preprocess_overrides: Override DEFAULT_PREPROCESSING values.
    """
    if pre_processed:
        params = dict(DEFAULT_PREPROCESSING)
        params.update(preprocess_overrides)
        _, _, x_train, x_test, y_train, y_test = fetch_data(
            dataset, random_state, **params
        )
        x_train = _to_numpy(x_train)
        x_test = _to_numpy(x_test)
        y_train = _to_numpy(y_train)
        y_test = _to_numpy(y_test)
    else:
        dataset_name = next(
            name for name in DATASET_BASE_NAMES if dataset.startswith(name)
        )
        args_part = dataset[len(dataset_name) + 1 :]
        args = args_part.split("_") if args_part else []
        arg1 = int(args[0]) if len(args) >= 1 else None
        arg2 = int(args[1]) if len(args) >= 2 else None
        x_train, x_test, y_train, y_test = get_data(
            dataset_name, arg1=arg1, arg2=arg2
        )
        x_train, x_test = preprocess_data(
            x_train, x_test, preprocess_overrides.get("scaling", "none")
        )
        y_train, y_test = preprocess_labels(
            y_train, y_test, preprocess_overrides.get("labels_treatment", "none")
        )
        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

    if x_train.shape[1] < 2 or x_test.shape[1] < 2:
        raise ValueError(
            f"Dataset must have at least 2 features, got train={x_train.shape[1]} test={x_test.shape[1]}"
        )

    x_train_2 = x_train[:, :2]
    x_test_2 = x_test[:, :2]
    title_suffix = "preprocessed" if pre_processed else "raw"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{dataset}_{title_suffix}.png")

    plt.figure(figsize=(8, 6))
    blue = "#1f77b4"
    red = "#d62728"
    labels = np.unique(np.concatenate([y_train, y_test]))
    colors = [blue, red]
    for label, color in zip(labels, colors):
        mask_train = y_train == label
        mask_test = y_test == label
        if np.any(mask_train):
            plt.scatter(
                x_train_2[mask_train, 0],
                x_train_2[mask_train, 1],
                c=color,
                marker="o",
                edgecolor="k",
                s=30,
                label=f"train label {label}",
            )
        if np.any(mask_test):
            plt.scatter(
                x_test_2[mask_test, 0],
                x_test_2[mask_test, 1],
                c=color,
                marker="^",
                edgecolor="k",
                s=30,
                label=f"test label {label}",
            )
    plt.title(f"{dataset} ({title_suffix})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize a tabular dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., two_curves_2_5)",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Apply preprocessing before plotting",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save the plot",
    )
    args = parser.parse_args()

    out_path = visualize_dataset(
        args.dataset,
        pre_processed=args.preprocess,
        save_dir=args.save_dir,
        show=args.show,
    )
    print(out_path)
