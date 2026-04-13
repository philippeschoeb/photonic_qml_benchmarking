# Guide: `visu_ablation_runs.py`

## Location

```bash
tabular_data/utils/visu_ablation_runs.py
```

## What It Does

Generates visualization figures for ablation experiments by aggregating multiple run versions and computing:
- mean across versions
- standard deviation across versions

It targets folder patterns like:
- `hm_10_2_v1_*`, `hm_10_2_v2_*`, `hm_10_2_v3_*`
- `hm_10_10_v1_*`, `hm_10_10_v2_*`, `hm_10_10_v3_*`

For each `(dataset, model, backend)` combination, it computes mean/std for the same metrics as `visu_hp_minimal.py` and plots them with error bars.
Top-level figures are separated by model family, and dataset versions (for example `hidden_manifold_10_2` vs `hidden_manifold_10_10`) are on the x axis.

## Basic Usage

From `tabular_data/`:

```bash
python utils/visu_ablation_runs.py
```

Default behavior:
- `--dataset_prefix hm`
- `--settings 10_2 10_10`
- `--versions 1 2 3`
- `--runs_root results/run_all_hp_search_minimal_ablation`

## Custom Usage

```bash
python utils/visu_ablation_runs.py \
  --runs_root results/run_all_hp_search_minimal_ablation \
  --dataset_prefix hm \
  --settings 10_2 10_10 \
  --versions 1 2 3
```

## Useful Flags

- `--output_dir <path>`: choose where aggregate CSV and figures are saved.
- `--point`: render point-style figures instead of bars (error bars still shown). In `each_model/`, this uses data-complexity values on the x axis and colors ablation variants as:
  `None` (original), `Quantum Ablation`, `Classical Ablation`.

## Outputs

Main outputs in the chosen/inferred output directory:
- `hp_ablation_aggregate_mean_std.csv`
- `hp_ablation_accuracy_<family>.png`
- `hp_ablation_params_vectors_<family>.png`
- `hp_ablation_times_<family>.png`
- `each_model/` (per-model ablation figures with mean/std error bars; displayed accuracy values are means)
