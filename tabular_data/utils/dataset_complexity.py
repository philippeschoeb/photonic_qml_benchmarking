import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from registry import DATASET_BASE_NAMES

SUBSAMPLED_SUFFIX = "_subsampled"
SUBSAMPLE_DATASET_TOKEN = "downscaled_mnist_pca"
SUBSAMPLE_TRAIN_SIZE = 240
SUBSAMPLE_TEST_SIZE = 60
COMBINE_SUFFIX = "_combine"

downscaled_mnist_pca_values = [2, 4, 6, 8]
hidden_manifold_values = [(2, 6), (8, 6)]
hidden_manifold_diff_values = [(10, 2), (10, 10)]
two_curves_values = [(2, 5), (8, 5)]
two_curves_diff_values = [(10, 2), (10, 10)]

COMBINE_DATASET_CONFIGS: dict[str, list] = {
    "downscaled_mnist_pca": downscaled_mnist_pca_values,
    "hidden_manifold": hidden_manifold_values,
    "hidden_manifold_diff": hidden_manifold_diff_values,
    "two_curves": two_curves_values,
    "two_curves_diff": two_curves_diff_values,
}


def _parse_dataset_name(dataset: str) -> tuple[str, int | None, int | None]:
    """Parse dataset identifier into base name and optional integer args."""
    dataset_name = next(
        (name for name in DATASET_BASE_NAMES if dataset.startswith(name)),
        None,
    )
    if dataset_name is None:
        supported = ", ".join(DATASET_BASE_NAMES)
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Expected one of: {supported}"
        )

    args_part = dataset[len(dataset_name) + 1 :]
    args = args_part.split("_") if args_part else []

    arg1 = int(args[0]) if len(args) >= 1 and args[0] != "" else None
    arg2 = int(args[1]) if len(args) >= 2 and args[1] != "" else None
    return dataset_name, arg1, arg2


def _to_arff_value(value) -> str:
    """Convert a scalar to an ARFF-safe string."""
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (str, np.str_)):
        escaped = str(value).replace("'", "\\'")
        return f"'{escaped}'"
    return str(value)


def _write_arff(x: np.ndarray, y: np.ndarray, output_file: Path, relation: str) -> Path:
    """Write a feature matrix and labels to a single ARFF file."""
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {x.shape}")
    if len(x) != len(y):
        raise ValueError(
            f"Mismatched number of samples: len(x)={len(x)} vs len(y)={len(y)}"
        )

    class_values = sorted(np.unique(y).tolist())
    class_values_str = ",".join(_to_arff_value(v) for v in class_values)

    lines = [f"@RELATION {relation}", ""]
    for idx in range(x.shape[1]):
        lines.append(f"@ATTRIBUTE feature_{idx} NUMERIC")
    lines.append(f"@ATTRIBUTE class {{{class_values_str}}}")
    lines.extend(["", "@DATA"])

    for row, label in zip(x, y):
        row_values = [_to_arff_value(v) for v in row]
        row_values.append(_to_arff_value(label))
        lines.append(",".join(row_values))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n")
    return output_file


def _to_metric_stats(metric_value, metric_name: str, split_name: str) -> tuple[float, float]:
    """Convert pycol output to (mean, std), supporting scalar or vector metrics."""
    arr = np.asarray(metric_value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{metric_name} returned an empty value for {split_name}.")
    return float(np.mean(arr)), float(np.std(arr))


def _validate_subsample_args(dataset: str, subsample: bool) -> None:
    if subsample and SUBSAMPLE_DATASET_TOKEN not in dataset:
        raise ValueError(
            f"--subsample only supports datasets containing "
            f"'{SUBSAMPLE_DATASET_TOKEN}', got '{dataset}'."
        )


def _validate_combine_args(dataset: str, pycol_combine_visu: bool) -> None:
    if pycol_combine_visu and dataset not in COMBINE_DATASET_CONFIGS:
        allowed = ", ".join(COMBINE_DATASET_CONFIGS.keys())
        raise ValueError(
            f"--pycol_combine_visu only supports dataset in {{{allowed}}}, got '{dataset}'."
        )
    if not pycol_combine_visu and dataset in COMBINE_DATASET_CONFIGS:
        raise ValueError(
            f"Dataset '{dataset}' is reserved for --pycol_combine_visu mode. "
            f"Add --pycol_combine_visu."
        )


def _subsample_split(
    x: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    split_name: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < n_samples:
        raise ValueError(
            f"Cannot subsample {n_samples} samples from {split_name} split with "
            f"{len(x)} samples."
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(x), size=n_samples, replace=False)
    return x[indices], y[indices]


def _get_tqdm():
    try:
        from tqdm import tqdm as _tqdm

        return _tqdm
    except ImportError:
        return None


def _make_progress(total: int, desc: str):
    tqdm_cls = _get_tqdm()
    if tqdm_cls is None:
        return None
    progress = tqdm_cls(
        total=total,
        desc=desc,
        unit="step",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0,
    )
    progress.refresh()
    return progress


def _load_dataset_splits(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from datasets.data import download_datasets, get_data

    dataset_name, arg1, arg2 = _parse_dataset_name(dataset)
    try:
        x_train, x_test, y_train, y_test = get_data(dataset_name, arg1=arg1, arg2=arg2)
    except FileNotFoundError:
        download_datasets()
        x_train, x_test, y_train, y_test = get_data(dataset_name, arg1=arg1, arg2=arg2)
    return (
        np.asarray(x_train),
        np.asarray(x_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


def download_dataset_as_arff(
    dataset: str,
    output_root: Path | None = None,
    subsample: bool = False,
) -> Path:
    """Fetch dataset and store train/test and merged ARFF files."""
    _validate_subsample_args(dataset, subsample)
    x_train, x_test, y_train, y_test = _load_dataset_splits(dataset)

    dataset_key = dataset
    if subsample:
        x_train, y_train = _subsample_split(
            x_train, y_train, SUBSAMPLE_TRAIN_SIZE, "train", seed=0
        )
        x_test, y_test = _subsample_split(
            x_test, y_test, SUBSAMPLE_TEST_SIZE, "test", seed=1
        )
        dataset_key = f"{dataset}{SUBSAMPLED_SUFFIX}"

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    if output_root is None:
        output_root = Path(ROOT_DIR) / "datasets" / "downloaded"

    dataset_dir = output_root / dataset_key
    train_file = dataset_dir / f"{dataset_key}_train.arff"
    test_file = dataset_dir / f"{dataset_key}_test.arff"
    merged_file = dataset_dir / f"{dataset_key}.arff"

    _write_arff(x_train, y_train, train_file, relation=f"{dataset_key}_train")
    _write_arff(x_test, y_test, test_file, relation=f"{dataset_key}_test")
    return _write_arff(x_all, y_all, merged_file, relation=dataset_key)


def _combine_concrete_dataset(dataset: str, value) -> tuple[str, str]:
    if dataset == "downscaled_mnist_pca":
        concrete = f"{dataset}_{value}"
        label = f"d={value}"
        return concrete, label
    if dataset in {"hidden_manifold", "hidden_manifold_diff", "two_curves", "two_curves_diff"}:
        base = "hidden_manifold" if dataset.startswith("hidden_manifold") else "two_curves"
        d, second = value
        concrete = f"{base}_{d}_{second}"
        if base == "hidden_manifold":
            label = f"d={d}, m={second}"
        else:
            label = f"d={d}, D={second}"
        return concrete, label
    raise ValueError(f"Unsupported combine dataset '{dataset}'.")


def download_combined_dataset_arffs(
    dataset: str,
    output_root: Path | None = None,
    subsample: bool = False,
) -> list[tuple[str, str, Path]]:
    if output_root is None:
        output_root = Path(ROOT_DIR) / "datasets" / "downloaded"

    dataset_dir = output_root / f"{dataset}{COMBINE_SUFFIX}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, str, Path]] = []
    for value in COMBINE_DATASET_CONFIGS[dataset]:
        concrete_dataset, label = _combine_concrete_dataset(dataset, value)
        filename_stem = concrete_dataset
        if subsample:
            filename_stem = f"{filename_stem}{SUBSAMPLED_SUFFIX}"
        arff_file = dataset_dir / f"{filename_stem}.arff"
        if not arff_file.exists():
            x_train, x_test, y_train, y_test = _load_dataset_splits(concrete_dataset)
            if subsample:
                x_train, y_train = _subsample_split(
                    x_train, y_train, SUBSAMPLE_TRAIN_SIZE, "train", seed=0
                )
                x_test, y_test = _subsample_split(
                    x_test, y_test, SUBSAMPLE_TEST_SIZE, "test", seed=1
                )
            x_all = np.concatenate([x_train, x_test], axis=0)
            y_all = np.concatenate([y_train, y_test], axis=0)
            _write_arff(x_all, y_all, arff_file, relation=filename_stem)
        outputs.append((concrete_dataset, label, arff_file))
    return outputs


def pycol_combined_figure(
    dataset: str,
    dataset_dir: Path,
    arff_configs: list[tuple[str, str, Path]],
    subsample: bool = False,
) -> Path:
    from pycol_complexity.complexity import Complexity
    import matplotlib.pyplot as plt

    metric_specs = [
        ("Feature Overlap (F1)", "F1"),
        ("Instance Overlap (R-value)", "R_value"),
        ("Structural Overlap (N1)", "N1"),
        ("Multiresolution Overlap (MRCA)", "MRCA"),
    ]

    labels = []
    metric_values = {metric_label: [] for metric_label, _ in metric_specs}
    metric_stds = {metric_label: [] for metric_label, _ in metric_specs}
    for _, config_label, arff_file in arff_configs:
        complexity = Complexity(arff_file, distance_func="default", file_type="arff")
        labels.append(config_label)
        for metric_label, method_name in metric_specs:
            raw_value = getattr(complexity, method_name)()
            mean_value, std_value = _to_metric_stats(raw_value, metric_label, "combined")
            metric_values[metric_label].append(mean_value)
            metric_stds[metric_label].append(std_value)

    x = np.arange(len(labels))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(labels))]

    fig, axes = plt.subplots(2, 2, figsize=(10, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, (metric_label, _) in enumerate(metric_specs):
        ax = axes[idx]
        values = metric_values[metric_label]
        stds = metric_stds[metric_label]
        ax.bar(
            x,
            values,
            yerr=stds,
            color=colors,
            ecolor="black",
            capsize=4,
            error_kw={"elinewidth": 1},
        )
        ax.set_title(metric_label, fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Metric Value", fontsize=14)
        ax.set_xlabel("Dataset Configuration", fontsize=14)
        ax.xaxis.set_label_coords(0.5, -0.24)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(labels))
        ax.tick_params(axis="y", labelsize=12)
        for i, value in enumerate(values):
            label_y = value
            label_va = "bottom"
            if value > 0.5:
                label_y = value - 0.03
                label_va = "top"
            ax.text(
                i - 0.2,
                label_y,
                f"{value:.2f}",
                ha="center",
                va=label_va,
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                i,
                -0.12,
                labels[i],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11,
            )

    title_dataset = dataset
    if subsample:
        title_dataset = f"{title_dataset} (subsampled)"
    fig.suptitle(
        f"Pycol-Complexity Metrics for {title_dataset}\n(train + test)", fontsize=20
    )
    fig.tight_layout(rect=[0, 0.10, 1, 0.92])

    output_stem = f"{dataset}"
    if subsample:
        output_stem = f"{output_stem}{SUBSAMPLED_SUFFIX}"
    output_file = dataset_dir / f"pycol_combined_{output_stem}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def sklearn_accuracy_figure(dataset: str, subsample: bool = False) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    x_train, x_test, y_train, y_test = _load_dataset_splits(dataset)
    dataset_key = dataset
    if subsample:
        x_train, y_train = _subsample_split(
            x_train, y_train, SUBSAMPLE_TRAIN_SIZE, "train", seed=0
        )
        x_test, y_test = _subsample_split(
            x_test, y_test, SUBSAMPLE_TEST_SIZE, "test", seed=1
        )
        dataset_key = f"{dataset_key} (subsampled)"

    models = [
        ("LogisticRegression", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=5)),
        ("HistGradientBoostingClassifier", HistGradientBoostingClassifier()),
    ]
    progress = _make_progress(
        total=len(models) * 4,
        desc=f"sklearn_visu {dataset_key}",
    )

    model_names = []
    train_scores = []
    test_scores = []
    try:
        for model_name, model in models:
            if progress is not None:
                progress.set_postfix_str(f"train {model_name}")
                progress.update(1)
                progress.refresh()
            model.fit(x_train, y_train)
            train_pred = model.predict(x_train)
            if progress is not None:
                progress.update(1)
                progress.refresh()
                progress.set_postfix_str(f"test {model_name}")
                progress.update(1)
                progress.refresh()
            test_pred = model.predict(x_test)
            if progress is not None:
                progress.update(1)
                progress.refresh()
            model_names.append(model_name)
            train_scores.append(float(accuracy_score(y_train, train_pred)))
            test_scores.append(float(accuracy_score(y_test, test_pred)))
    finally:
        if progress is not None:
            progress.close()

    x = np.arange(len(model_names))
    width = 0.36
    plt.figure(figsize=(10, 6))
    bars_train = plt.bar(
        x - width / 2, train_scores, width=width, label="Train", color="#1f77b4"
    )
    bars_test = plt.bar(
        x + width / 2, test_scores, width=width, label="Test", color="#ff7f0e"
    )
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(f"Sklearn Models Accuracy for {dataset_key}")
    plt.legend()
    for bars, vals in ((bars_train, train_scores), (bars_test, test_scores)):
        for bar, val in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()-0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    plt.tight_layout()

    dataset_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{dataset}{COMBINE_SUFFIX}"
    if subsample:
        output_stem = f"{output_stem}{SUBSAMPLED_SUFFIX}"
    output_file = dataset_dir / f"sklearn_{output_stem}.png"
    plt.savefig(output_file, dpi=200)
    plt.close()
    return output_file


def sklearn_combined_accuracy_figure(dataset: str, subsample: bool = False) -> Path:
    import matplotlib.pyplot as plt
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    configs = COMBINE_DATASET_CONFIGS[dataset]
    config_labels = []
    model_scores: dict[str, list[float]] = {
        "LogisticRegression": [],
        "KNeighborsClassifier": [],
        "HistGradientBoostingClassifier": [],
    }
    model_builders = [
        ("LogisticRegression", lambda: LogisticRegression(max_iter=2000, solver="lbfgs")),
        ("KNeighborsClassifier", lambda: KNeighborsClassifier(n_neighbors=5)),
        ("HistGradientBoostingClassifier", lambda: HistGradientBoostingClassifier()),
    ]
    progress = _make_progress(
        total=len(configs) * len(model_builders),
        desc=f"sklearn_combine_visu {dataset}",
    )

    try:
        for value in configs:
            concrete_dataset, config_label = _combine_concrete_dataset(dataset, value)
            x_train, x_test, y_train, y_test = _load_dataset_splits(concrete_dataset)
            if subsample:
                x_train, y_train = _subsample_split(
                    x_train, y_train, SUBSAMPLE_TRAIN_SIZE, "train", seed=0
                )
                x_test, y_test = _subsample_split(
                    x_test, y_test, SUBSAMPLE_TEST_SIZE, "test", seed=1
                )
            config_labels.append(config_label)
            for model_name, model_builder in model_builders:
                model = model_builder()
                if progress is not None:
                    progress.set_postfix_str(f"{config_label} train {model_name}")
                    progress.refresh()
                model.fit(x_train, y_train)
                if progress is not None:
                    progress.set_postfix_str(f"{config_label} test {model_name}")
                    progress.refresh()
                test_pred = model.predict(x_test)
                if progress is not None:
                    progress.update(1)
                    progress.refresh()
                model_scores[model_name].append(float(accuracy_score(y_test, test_pred)))
    finally:
        if progress is not None:
            progress.close()

    x = np.arange(len(config_labels))
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, sharey=True)
    model_order = [
        "LogisticRegression",
        "KNeighborsClassifier",
        "HistGradientBoostingClassifier",
    ]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    for idx, model_name in enumerate(model_order):
        ax = axes[idx]
        vals = model_scores[model_name]
        ax.bar(x, vals, color=colors[idx], width=0.6)
        ax.set_title(model_name, fontsize=16)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Test Accuracy", fontsize=14)
        ax.set_xlabel("Dataset Configuration", fontsize=14)
        ax.xaxis.set_label_coords(0.5, -0.20)
        ax.set_xticks(x)
        ax.set_xticklabels([""] * len(config_labels))
        ax.tick_params(axis="y", labelsize=12)
        for i, val in enumerate(vals):
            ax.text(
                i,
                val - 0.1,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                i,
                -0.10,
                config_labels[i],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=11,
            )

    title_dataset = dataset
    if subsample:
        title_dataset = f"{title_dataset} (subsampled)"
    fig.suptitle(
        f"Sklearn Models Test Accuracy for {title_dataset}", fontsize=20
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    output_stem = f"{dataset}{COMBINE_SUFFIX}"
    if subsample:
        output_stem = f"{output_stem}{SUBSAMPLED_SUFFIX}"
    dataset_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / f"{output_stem}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file = dataset_dir / f"sklearn_combined_{dataset}.png"
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    return output_file


def all_combine_visualizations(subsample: bool = False) -> Path:
    all_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / "all_combined"
    all_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in all_dir.glob("pycol_combined_*.png"):
        stale_file.unlink()
    for stale_file in all_dir.glob("sklearn_combined_*.png"):
        stale_file.unlink()
    total_steps = len(COMBINE_DATASET_CONFIGS) * 2
    progress = _make_progress(total=total_steps, desc="all_combine_visu")

    try:
        for dataset in COMBINE_DATASET_CONFIGS:
            dataset_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / f"{dataset}{COMBINE_SUFFIX}"
            use_subsample = subsample and dataset == SUBSAMPLE_DATASET_TOKEN
            pycol_stem = dataset
            if use_subsample:
                pycol_stem = f"{pycol_stem}{SUBSAMPLED_SUFFIX}"
            if progress is not None:
                progress.set_postfix_str(f"{dataset} pycol")
            pycol_file = dataset_dir / f"pycol_combined_{pycol_stem}.png"
            arff_configs = download_combined_dataset_arffs(
                dataset, subsample=use_subsample
            )
            pycol_file = pycol_combined_figure(
                dataset, dataset_dir, arff_configs, subsample=use_subsample
            )
            shutil.copy2(pycol_file, all_dir / pycol_file.name)
            if progress is not None:
                progress.update(1)
                progress.refresh()

            if progress is not None:
                progress.set_postfix_str(f"{dataset} sklearn")
            sklearn_dir = dataset_dir
            if use_subsample:
                sklearn_dir = (
                    Path(ROOT_DIR)
                    / "datasets"
                    / "downloaded"
                    / f"{dataset}{COMBINE_SUFFIX}{SUBSAMPLED_SUFFIX}"
                )
            sklearn_file = sklearn_dir / f"sklearn_combined_{dataset}.png"
            sklearn_file = sklearn_combined_accuracy_figure(
                dataset, subsample=use_subsample
            )
            shutil.copy2(sklearn_file, all_dir / sklearn_file.name)
            if progress is not None:
                progress.update(1)
                progress.refresh()
    finally:
        if progress is not None:
            progress.close()

    return all_dir


def pycol_complexity_figure(dataset_dir: Path, arff_file: Path) -> Path:
    """Generate a train-vs-test complexity figure saved next to the merged ARFF."""
    from pycol_complexity.complexity import Complexity
    import matplotlib.pyplot as plt

    arff_file = Path(arff_file)
    train_file = dataset_dir / f"{arff_file.stem}_train.arff"
    test_file = dataset_dir / f"{arff_file.stem}_test.arff"
    outdir = dataset_dir / f"pycol_{arff_file.stem}"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Missing split ARFF files. Expected {train_file} and {test_file}."
        )

    complexity_train = Complexity(train_file, distance_func="default", file_type="arff")
    complexity_test = Complexity(test_file, distance_func="default", file_type="arff")

    metric_specs = [
        ("Feature Overlap (F1)", "F1"),
        ("Instance Overlap (R-value)", "R_value"),
        ("Structural Overlap (N1)", "N1"),
        ("Multiresolution Overlap (MRCA)", "MRCA"),
    ]
    metric_labels = [
        label for label, _ in metric_specs
    ]
    train_means = []
    test_means = []
    train_stds = []
    test_stds = []
    for label, method_name in metric_specs:
        train_raw = getattr(complexity_train, method_name)()
        test_raw = getattr(complexity_test, method_name)()
        train_mean, train_std = _to_metric_stats(train_raw, label, "train")
        test_mean, test_std = _to_metric_stats(test_raw, label, "test")
        train_means.append(train_mean)
        test_means.append(test_mean)
        train_stds.append(train_std)
        test_stds.append(test_std)

    arff_file.parent.mkdir(parents=True, exist_ok=True)
    output_file = outdir.with_suffix(".png")

    plt.figure(figsize=(10, 5))
    x = np.arange(len(metric_labels))
    width = 0.38
    bars_train = plt.bar(
        x - width / 2,
        train_means,
        yerr=train_stds,
        capsize=4,
        width=width,
        label="Train",
        color="#1f77b4",
    )
    bars_test = plt.bar(
        x + width / 2,
        test_means,
        yerr=test_stds,
        capsize=4,
        width=width,
        label="Test",
        color="#ff7f0e",
    )
    plt.ylabel("Metric Value")
    plt.title(f"Pycol-Complexity Metrics for {arff_file.stem} (mean +/- std)")
    plt.xticks(x, metric_labels, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.legend()

    for bars, means, stds in (
        (bars_train, train_means, train_stds),
        (bars_test, test_means, test_stds),
    ):
        for bar, mean, std in zip(bars, means, stds):
            label_y = bar.get_height()
            label_va = "bottom"
            if mean > 0.5:
                label_y = bar.get_height() - 0.03
                label_va = "top"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{mean:.2f} +/- {std:.2f}",
                ha="center",
                va=label_va,
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a benchmark dataset and export it to ARFF format."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Dataset identifier, e.g. hidden_manifold_5_3 or two_curves_2_5",
    )
    parser.add_argument(
        "--download_arff",
        action="store_true",
        help="Export selected dataset to ARFF under datasets/downloaded/{dataset}",
    )
    parser.add_argument(
        "--pycol_visu",
        action="store_true",
        help="Generate a pycol-complexity figure for the selected dataset",
    )
    parser.add_argument(
        "--subsample",
        dest="subsample",
        action="store_true",
        help=(
            f"Only for datasets containing '{SUBSAMPLE_DATASET_TOKEN}': use a cached "
            f"subsampled ARFF export with {SUBSAMPLE_TRAIN_SIZE} train and "
            f"{SUBSAMPLE_TEST_SIZE} test samples under datasets/downloaded/"
            "{dataset}_subsampled"
        ),
    )
    parser.add_argument(
        "--pycol_combine_visu",
        action="store_true",
        help=(
            "Combine mode for dataset families: "
            "downscaled_mnist_pca, hidden_manifold, hidden_manifold_diff, "
            "two_curves, two_curves_diff."
        ),
    )
    parser.add_argument(
        "--sklearn_visu",
        action="store_true",
        help="Generate sklearn model accuracy visualization (train + test).",
    )
    parser.add_argument(
        "--sklearn_combine_visu",
        action="store_true",
        help=(
            "Generate sklearn test-accuracy combine visualization for dataset "
            "families."
        ),
    )
    parser.add_argument(
        "--all_combine_visu",
        action="store_true",
        help=(
            "Generate all pycol/sklearn combine visualizations and copy them to "
            "datasets/downloaded/all_combined/."
        ),
    )
    args = parser.parse_args()

    task_flags = [
        args.download_arff,
        args.pycol_visu,
        args.pycol_combine_visu,
        args.sklearn_visu,
        args.sklearn_combine_visu,
        args.all_combine_visu,
    ]
    if sum(task_flags) != 1:
        parser.error(
            "Use exactly one task flag: --download_arff, --pycol_visu, "
            "--pycol_combine_visu, --sklearn_visu, --sklearn_combine_visu, "
            "or --all_combine_visu."
        )

    if args.all_combine_visu:
        if args.dataset is not None:
            parser.error("--dataset cannot be used with --all_combine_visu.")
        output_dir = all_combine_visualizations(subsample=args.subsample)
        print(output_dir)
        return

    if args.dataset is None:
        parser.error("--dataset is required for this task.")

    try:
        _validate_subsample_args(args.dataset, args.subsample)
        _validate_combine_args(
            args.dataset, args.pycol_combine_visu or args.sklearn_combine_visu
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.pycol_combine_visu:
        dataset_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / f"{args.dataset}{COMBINE_SUFFIX}"
        arff_configs = download_combined_dataset_arffs(
            args.dataset, subsample=args.subsample
        )
        figure_file = pycol_combined_figure(
            args.dataset, dataset_dir, arff_configs, subsample=args.subsample
        )
        print(figure_file)
        return

    if args.sklearn_combine_visu:
        figure_file = sklearn_combined_accuracy_figure(
            args.dataset, subsample=args.subsample
        )
        print(figure_file)
        return

    if args.download_arff:
        output_file = download_dataset_as_arff(args.dataset, subsample=args.subsample)
        print(output_file)
        return

    if args.sklearn_visu:
        figure_file = sklearn_accuracy_figure(args.dataset, subsample=args.subsample)
        print(figure_file)
        return

    if args.pycol_visu:
        dataset_key = args.dataset
        if args.subsample:
            dataset_key = f"{args.dataset}{SUBSAMPLED_SUFFIX}"
        dataset_dir = Path(ROOT_DIR) / "datasets" / "downloaded" / dataset_key
        arff_file = dataset_dir / f"{dataset_key}.arff"
        train_file = dataset_dir / f"{dataset_key}_train.arff"
        test_file = dataset_dir / f"{dataset_key}_test.arff"
        if not arff_file.exists() or not train_file.exists() or not test_file.exists():
            download_dataset_as_arff(args.dataset, subsample=args.subsample)
        figure_file = pycol_complexity_figure(dataset_dir, arff_file)
        print(figure_file)
        return

    parser.error(
        "No action selected. Use exactly one task flag: --download_arff, "
        "--pycol_visu, --pycol_combine_visu, --sklearn_visu, "
        "--sklearn_combine_visu, or --all_combine_visu."
    )


if __name__ == "__main__":
    main()
