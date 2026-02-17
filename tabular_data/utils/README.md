# Utils

This folder contains utility scripts for dataset export, complexity analysis, metric handling, and visualization.

## Guide: `dataset_complexity.py`

### Location

```bash
tabular_data/utils/dataset_complexity.py
```

### Available tasks

Choose exactly one task flag per command:
- `--download_arff`
- `--pycol_visu`
- `--pycol_combine_visu`
- `--sklearn_visu`
- `--sklearn_combine_visu`
- `--all_combine_visu`

Important rules:
- Do not combine task flags.
- `--download_arff` cannot be used with any visualization task flag.
- `--subsample` is only valid for datasets containing `downscaled_mnist_pca`.

### Basic usage

From `tabular_data/`:

```bash
python utils/dataset_complexity.py --dataset <dataset_name> --<task_flag>
```

Example:

```bash
python utils/dataset_complexity.py --dataset two_curves_2_5 --download_arff
```

### Task: `--download_arff`

Exports train, test, and merged ARFF files for one concrete dataset.

Useful flags:
- `--dataset <dataset_name>` such as `downscaled_mnist_pca_2`, `hidden_manifold_5_3`, `two_curves_2_5`
- `--subsample` only for `downscaled_mnist_pca_*` datasets

Example:

```bash
python utils/dataset_complexity.py --dataset downscaled_mnist_pca_2 --download_arff --subsample
```

Outputs:
- Without subsample:
  `tabular_data/datasets/downloaded/{dataset}/{dataset}_train.arff`
  `tabular_data/datasets/downloaded/{dataset}/{dataset}_test.arff`
  `tabular_data/datasets/downloaded/{dataset}/{dataset}.arff`
- With subsample:
  `tabular_data/datasets/downloaded/{dataset}_subsampled/{dataset}_subsampled_train.arff`
  `tabular_data/datasets/downloaded/{dataset}_subsampled/{dataset}_subsampled_test.arff`
  `tabular_data/datasets/downloaded/{dataset}_subsampled/{dataset}_subsampled.arff`

### Task: `--pycol_visu`

Generates pycol-complexity visualization for one concrete dataset (train vs test bars for 4 metrics).

Useful flags:
- `--dataset <dataset_name>`
- `--subsample` only for `downscaled_mnist_pca_*`

Example:

```bash
python utils/dataset_complexity.py --dataset downscaled_mnist_pca_2 --pycol_visu --subsample
```

Outputs:
- `tabular_data/datasets/downloaded/{dataset_or_dataset_subsampled}/pycol_{dataset_or_dataset_subsampled}.png`

### Task: `--pycol_combine_visu`

Generates a single pycol-complexity comparison figure across fixed configurations of a dataset family.

Useful flags:
- `--dataset` must be one of:
  `downscaled_mnist_pca`, `hidden_manifold`, `hidden_manifold_diff`, `two_curves`, `two_curves_diff`
- `--subsample` allowed only with `--dataset downscaled_mnist_pca`

Examples:

```bash
python utils/dataset_complexity.py --dataset downscaled_mnist_pca --pycol_combine_visu
python utils/dataset_complexity.py --dataset hidden_manifold --pycol_combine_visu
```

Outputs:
- ARFF cache: `tabular_data/datasets/downloaded/{dataset}_combine/*.arff`
- Figure: `tabular_data/datasets/downloaded/{dataset}_combine/pycol_combined_{dataset}.png`
- Figure with subsample:
  `tabular_data/datasets/downloaded/{dataset}_combine/pycol_combined_{dataset}_subsampled.png`

### Task: `--sklearn_visu`

Trains 3 sklearn models on one concrete dataset and plots train + test accuracy.

Models:
- `LogisticRegression(max_iter=2000, solver="lbfgs")`
- `KNeighborsClassifier(n_neighbors=5)`
- `HistGradientBoostingClassifier()`

Useful flags:
- `--dataset <dataset_name>`
- `--subsample` only for `downscaled_mnist_pca_*`

Example:

```bash
python utils/dataset_complexity.py --dataset two_curves_2_5 --sklearn_visu
```

Outputs:
- Without subsample:
  `tabular_data/datasets/downloaded/{dataset}/sklearn_{dataset}_combine.png`
- With subsample:
  `tabular_data/datasets/downloaded/{dataset} (subsampled)/sklearn_{dataset}_combine_subsampled.png`

Notes:
- With subsample, the directory name includes ` (subsampled)` for this task.

### Task: `--sklearn_combine_visu`

Trains the same 3 sklearn models across a dataset family and plots test accuracy across configurations.

Useful flags:
- `--dataset` must be one of:
  `downscaled_mnist_pca`, `hidden_manifold`, `hidden_manifold_diff`, `two_curves`, `two_curves_diff`
- `--subsample` allowed only with `--dataset downscaled_mnist_pca`

Examples:

```bash
python utils/dataset_complexity.py --dataset downscaled_mnist_pca --sklearn_combine_visu
python utils/dataset_complexity.py --dataset two_curves --sklearn_combine_visu
```

Outputs:
- `tabular_data/datasets/downloaded/{dataset}_combine/sklearn_combined_{dataset}.png`
- With subsample:
  `tabular_data/datasets/downloaded/{dataset}_combine_subsampled/sklearn_combined_{dataset}.png`

### Task: `--all_combine_visu`

Generates all combine visualizations for dataset families and collects them into one folder.

Behavior:
- Runs both:
  `--pycol_combine_visu` equivalent and `--sklearn_combine_visu` equivalent
  for each dataset family:
  `downscaled_mnist_pca`, `hidden_manifold`, `hidden_manifold_diff`, `two_curves`, `two_curves_diff`.
- Generates exactly 10 figures total:
  5 pycol combine + 5 sklearn combine.
- Copies all generated combine figures into:
  `tabular_data/datasets/downloaded/all_combined/`
- Existing `pycol_combined_*.png` and `sklearn_combined_*.png` in that folder are refreshed each run.
- `--dataset` is not required for this task.
- Using `--dataset` with `--all_combine_visu` is invalid and raises an error.
- `--subsample` is supported and only applies to `downscaled_mnist_pca`
  (to speed up that family while keeping total output at 10 figures).

Example:

```bash
python utils/dataset_complexity.py --all_combine_visu
python utils/dataset_complexity.py --all_combine_visu --subsample
```

Outputs:
- `tabular_data/datasets/downloaded/all_combined/*.png`

Typical file names in `all_combined`:
- `pycol_combined_<dataset>.png` (or `pycol_combined_downscaled_mnist_pca_subsampled.png` when `--subsample`)
- `sklearn_combined_<dataset>.png`
