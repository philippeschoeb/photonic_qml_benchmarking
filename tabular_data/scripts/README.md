# `tabular_data/scripts` Overview

This folder contains convenience shell scripts for running common benchmark sweeps.

Run scripts from `tabular_data/` unless noted otherwise.

## Scripts

### `run_all_single.sh`
- Runs one **single** training/evaluation pass per model/backend combination for a fixed dataset.
- Writes grouped outputs under one batch folder (for easier comparison).
- Dataset choice: edit the `DATASET` variable in the script.

Example:
```bash
bash scripts/run_all_single.sh
```

### `run_all_single_ablation.sh`
- Same as `run_all_single.sh`, but also runs all compatible `_abla_q` / `_abla_c` variants.
- Uses the same single-run hyperparameter presets.
- Dataset choice: edit the `DATASET` variable in the script.

Example:
```bash
bash scripts/run_all_single_ablation.sh
```

### `run_photonic_hp_search_minimal.sh`
- Runs **minimal-profile hyperparameter search** for photonic-backed models only.
- Dataset choice precedence:
  1. first positional argument
  2. `DATASET` environment variable
  3. script fallback value

Example:
```bash
bash scripts/run_photonic_hp_search_minimal.sh hidden_manifold_2_6
```

### `run_gate_hp_search_minimal.sh`
- Runs minimal-profile hyperparameter search for gate-backed models only.
- Dataset choice precedence:
  1. first positional argument
  2. `DATASET` environment variable
  3. script fallback value

Example:
```bash
bash scripts/run_gate_hp_search_minimal.sh hidden_manifold_2_6
```

### `run_classical_hp_search_minimal.sh`
- Runs minimal-profile hyperparameter search for classical baselines only.
- Dataset choice precedence:
  1. first positional argument
  2. `DATASET` environment variable
  3. script fallback value

Example:
```bash
bash scripts/run_classical_hp_search_minimal.sh hidden_manifold_2_6
```

### `run_all_hp_search_minimal.sh`
- Runs minimal-profile hyperparameter search across all model families.
- Explicitly disables ablations.
- Dataset choice precedence:
  1. first positional argument
  2. `DATASET` environment variable
  3. script fallback value

Example:
```bash
bash scripts/run_all_hp_search_minimal.sh hidden_manifold_10_10
```

### `run_all_hp_search_minimal_ablation.sh`
- Runs minimal-profile hyperparameter search across all model families.
- Supports optional ablation execution.
- Dataset choice precedence:
  1. first positional argument
  2. `DATASET` environment variable
  3. script fallback value

Examples:
```bash
bash scripts/run_all_hp_search_minimal_ablation.sh hidden_manifold_10_10
bash scripts/run_all_hp_search_minimal_ablation.sh hidden_manifold_10_10 --with-ablation
```

### `run_all_hp_search.sh`
- Legacy broad hyperparameter sweep script (multi-dataset, runtime guardrails, timeout-based control).
- Useful for exploratory large sweeps, but slower than minimal scripts.
- Dataset list is fixed in the script’s `DATASETS` array:
  - `downscaled_mnist_pca_2`
  - `downscaled_mnist_pca_10`
  - `downscaled_mnist_pca_20`

Example:
```bash
bash scripts/run_all_hp_search.sh
```

### `run_all.sh`
- Top-level orchestrator for weekend-style multi-dataset HP sweeps.
- Reads settings from `run_all_config.json` (or a custom config path).
- Dataset list source:
  - `datasets` field in `run_all_config.json` (or in the custom config passed as first argument).

Examples:
```bash
bash scripts/run_all.sh
bash scripts/run_all.sh scripts/run_all_config.json
```

### `run_all_config.json`
- Configuration file consumed by `run_all.sh`.
- Defines dataset list, model list (or `all`), hyperparameter profile, and whether to run ablations.

## Notes

- Most scripts support turning Weights & Biases logging on/off via a variable near the top of the file.
- Output folders are created under `tabular_data/results/` with script-specific grouping.
