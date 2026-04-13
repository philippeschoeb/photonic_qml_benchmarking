# Guide: `visu_hp_minimal.py`

## Location

```bash
tabular_data/utils/visu_hp_minimal.py
```

## What It Does

Builds summary figures from minimal hyperparameter-search outputs (`hp_search_*.csv`), including:
- final train/test accuracy
- number of parameters / support vectors
- search + optimal-model training/eval time
- per-model breakdown figures in `each_model/`

When available, it also plots `number_of_configs` from `hp_search_*_number_of_configs.csv`.

## Basic Usage

From `tabular_data/`:

```bash
python utils/visu_hp_minimal.py --file_path <result_dir_or_csv>
```

Example:

```bash
python utils/visu_hp_minimal.py \
  --file_path results/run_all_hyperparam_search_minimal/<run_name>
```

## Combine Multiple Runs

You can pass multiple `--file_path` values to aggregate/compare runs:

```bash
python utils/visu_hp_minimal.py --file_path <run_dir_1> <run_dir_2> <run_dir_3>
```

## Useful Flags

- `--output_dir <path>`: choose exactly where figures are saved.
- `--point`: render point-style figures instead of bars (useful for dense multi-run comparisons).

## Outputs

Main outputs in the chosen/inferred output directory:
- `hp_minimal_accuracy.png`
- `hp_minimal_params_vectors.png`
- `hp_minimal_times.png`
- `hp_minimal_number_of_configs.png` (when config-count CSV exists for single-run input)
- `each_model/` (per-model figures)
