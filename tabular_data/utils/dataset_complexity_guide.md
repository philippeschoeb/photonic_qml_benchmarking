# Guide: `dataset_complexity.py`

## Location

```bash
tabular_data/utils/dataset_complexity.py
```

## What It Does

Provides CLI tools for dataset export and complexity visualizations:
- export datasets to ARFF files
- generate pycol-complexity plots
- generate sklearn baseline accuracy plots
- generate combined family plots (e.g., hidden_manifold, two_curves, spiral)

## Basic Usage

Run from `tabular_data/` and choose exactly one task flag.

### 1) Download ARFF

```bash
python utils/dataset_complexity.py --dataset hidden_manifold_10_2 --download_arff
```

### 2) Pycol complexity visualization

```bash
python utils/dataset_complexity.py --dataset hidden_manifold_10_2 --pycol_visu
```

### 3) sklearn train/test visualization

```bash
python utils/dataset_complexity.py --dataset hidden_manifold_10_2 --sklearn_visu
```

### 4) Combined family figures

```bash
python utils/dataset_complexity.py --dataset hidden_manifold --pycol_combine_visu
python utils/dataset_complexity.py --dataset hidden_manifold --sklearn_combine_visu
```

### 5) Generate all combined visualizations

```bash
python utils/dataset_complexity.py --all_combine_visu
```

## Important Flags

- `--dataset <name>`: required for all tasks except `--all_combine_visu`
- `--subsample`: use cached subsampled exports for supported datasets
- task flags (exactly one required):
  - `--download_arff`
  - `--pycol_visu`
  - `--pycol_combine_visu`
  - `--sklearn_visu`
  - `--sklearn_combine_visu`
  - `--all_combine_visu`

## Outputs

Outputs are written under `tabular_data/datasets/downloaded/` (dataset-specific folders and combined-figure folders).
