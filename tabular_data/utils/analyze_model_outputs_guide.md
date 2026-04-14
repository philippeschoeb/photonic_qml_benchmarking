# Guide: `analyze_model_outputs.py`

## Location

```bash
tabular_data/utils/analyze_model_outputs.py
```

## What It Does

Runs a structured analysis of model outputs and circuit outputs across selected `hidden_manifold_d_6` settings, then writes:
- a CSV table
- a Markdown report with grouped result tables

It evaluates representative `d` values from the requested range (min, midpoint, max).

## Basic Usage

Run from `tabular_data/`:

```bash
python utils/analyze_model_outputs.py
```

This uses defaults and writes outputs under:
- `tabular_data/results/analyze_model_outputs/<timestamp>/`

## Common Custom Usage

```bash
python utils/analyze_model_outputs.py \
  --d-min 2 \
  --d-max 20 \
  --manifold-dim 6 \
  --max-samples 25 \
  --batch-size 4 \
  --photonic-measurement default \
  --out-csv tabular_data/results/analyze_model_outputs/custom/model_output_analysis.csv \
  --out-md tabular_data/results/analyze_model_outputs/custom/model_output_analysis.md
```

## Useful Flags

- `--skip-classical`: exclude classical model specs
- `--photonic-measurement {default,mode_only,probs_only}`: choose photonic measurement config set
- `--debug-memory`: print RSS memory stats per config
- `--force-gc-per-config`: run GC after each configuration

## Outputs

- CSV with one row per evaluated configuration
- Markdown report with:
  - column glossary
  - grouped tables by `d`, backend/model, measurement mode, and ablation
