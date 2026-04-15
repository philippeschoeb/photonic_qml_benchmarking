# `tabular_data/utils` Usage Docs

This folder contains both helper modules and runnable scripts.

Runnability rule used here:
- If a file is designed to run directly as `python utils/<file>.py`, it is documented below.

## Runnable Utility Scripts

### `visualize_dataset.py` (quick usage in this README)

Purpose:
- Generate a 2D scatter visualization for a dataset and save it as an image.

Run from `tabular_data/`:

```bash
python utils/visualize_dataset.py --dataset two_curves_2_5
```

Useful flags:
- `--preprocess`: apply preprocessing before plotting
- `--show`: open interactive display
- `--save_dir <dir>`: output directory for the figure

### `visu_hp_minimal.py`

If you need information on `visu_hp_minimal.py`, refer to the guide [`visu_hp_minimal_guide.md`](./visu_hp_minimal_guide.md).

### `visu_ablation_runs.py`

If you need information on `visu_ablation_runs.py`, refer to the guide [`visu_ablation_runs_guide.md`](./visu_ablation_runs_guide.md).

### `dataset_complexity.py`

If you need information on `dataset_complexity.py`, refer to the guide [`dataset_complexity_guide.md`](./dataset_complexity_guide.md).

### `analyze_model_outputs.py`

If you need information on `analyze_model_outputs.py`, refer to the guide [`analyze_model_outputs_guide.md`](./analyze_model_outputs_guide.md).

### `summarize_long_training.py` (quick usage in this README)

Purpose:
- Summarize timeout/cut-short events from `long_training*.jsonl` reports.
- Groups by `model/backend/dataset` and prints top hyperparameter signatures.

Run from `tabular_data/`:

```bash
python utils/summarize_long_training.py --input results/run_all_single/hidden_manifold_2_6_YYYY-MM-DD_HH-MM-SS
```

Useful flags:
- `--input <path>`: JSONL file or directory to scan recursively.
- `--top_k_hyperparams <int>`: Number of top hyperparameter signatures per group.

## Non-runnable Helper Modules

These are utility modules imported by other code (not primary CLI scripts):
- `photonic_dims.py`
- `save_metrics.py`
