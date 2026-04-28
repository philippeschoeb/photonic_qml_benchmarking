# Tabular Data Benchmarking Guide

`tabular_data/` contains the tabular-data benchmarking pipeline for comparing
photonic quantum models, gate-based quantum models, and classical baselines on a
shared set of datasets. The code covers dataset loading/generation, model
construction, training, hyperparameter search, batch-run orchestration, metric
saving, and post-run analysis.

Use this README as the entry point for the folder. For script-specific details,
follow the links to the folder READMEs and guides.

## Where To Go

| Task | Start here |
| --- | --- |
| Run one model or one hyperparameter search | [`main.py`](main.py), then the CLI examples below |
| Run common benchmark sweeps | [`scripts/`](scripts/) and [`scripts/README.md`](scripts/README.md) |
| Change single-run defaults | [`hyperparameters/single_run/`](hyperparameters/single_run/) |
| Change HP-search spaces | [`hyperparameters/hyperparam_search/`](hyperparameters/hyperparam_search/) |
| Add or inspect datasets | [`datasets/`](datasets/) |
| Add or inspect models | [`models/`](models/) |
| Inspect training dispatch and loops | [`training/`](training/) |
| Analyze results, plot summaries, or visualize datasets | [`utils/`](utils/) and [`utils/README.md`](utils/README.md) |
| Inspect saved benchmark outputs | [`results/`](results/) and [`kept_results/`](kept_results/) |
| Work with Merlin/reuploading extras | [`merlin_additional/`](merlin_additional/) |

## Repository Map

### Top-level entry points

- [`main.py`](main.py) is the CLI entry point. It validates arguments, resolves
  defaults, sets the backend, initializes Weights & Biases when enabled, and
  calls either `run_single` or `run_search`.
- [`registry.py`](registry.py) defines supported dataset families, model names,
  run types, backends, and HP profiles. It also exposes `list_models()` and
  `get_dataset_base_name()`.
- [`helper.py`](helper.py) contains `architecture_help()`, the source behind
  `--architecture help`.
- [`visualize_2d_datasets.ipynb`](visualize_2d_datasets.ipynb) is a notebook for
  inspecting 2D dataset variants.

### Datasets

Dataset code lives in [`datasets/`](datasets/).

- [`datasets/data.py`](datasets/data.py) is the main dataset utility module:
  `download_datasets()`, `get_data()`, `get_summary()`, `preprocess_data()`,
  `preprocess_labels()`, `convert_array_to_tensor()`,
  `convert_tensor_to_loader()`, and `subsample()`.
- [`datasets/fetch_data.py`](datasets/fetch_data.py) exposes the training-facing
  loaders: `fetch_data()` for PyTorch-style training and `fetch_sk_data()` for
  sklearn-style models/searches.
- Dataset loaders read prebuilt HDF5 data:
  [`downscaled_mnist_pca_loading.py`](datasets/downscaled_mnist_pca_loading.py),
  [`hidden_manifold_loading.py`](datasets/hidden_manifold_loading.py), and
  [`two_curves_loading.py`](datasets/two_curves_loading.py).
- Dataset generators live in [`mnist.py`](datasets/mnist.py),
  [`hidden_manifold.py`](datasets/hidden_manifold.py),
  [`two_curves.py`](datasets/two_curves.py), and [`spiral.py`](datasets/spiral.py).
- [`datasets/select_dataset.py`](datasets/select_dataset.py) documents the
  intended dataset-parameter conventions.
- [`datasets/test_dimensions.py`](datasets/test_dimensions.py) checks dataset
  dimensional availability.

Supported dataset names:

1. `downscaled_mnist_pca_{d}`
2. `hidden_manifold_{d}_{m}`
3. `two_curves_{d}_{D}`
4. `spiral_{d}`

Parameter ranges:

- `downscaled_mnist_pca`: `d` in `[2, 3, ... , 20]`
- `hidden_manifold`: `d,m` in `[2, 3, ... , 20]`
- `two_curves`: `d,D` in `[2, 3, ... , 20]`
- `spiral`: `d` in `[2, 3, ... , 100]`, with 3 classes

### Models

Model code lives in [`models/`](models/).

- [`models/fetch_model.py`](models/fetch_model.py) is the model factory. Use
  `fetch_model()` for direct training and `fetch_sk_model()` for sklearn-compatible
  HP search.
- [`models/parameter_counting.py`](models/parameter_counting.py) counts total,
  quantum, classical, and support-vector parameters for saved summaries.
- [`models/ablation.py`](models/ablation.py) implements model-name parsing and
  ablation support through `parse_ablation_model_name()`,
  `can_apply_ablation()`, `ablate_model()`, and `apply_ablation_if_requested()`.
- [`models/photonic_based_utils.py`](models/photonic_based_utils.py) contains
  photonic circuit, Fock-state, measurement, grouping, and scaling helpers.
- [`models/gate_based_utils.py`](models/gate_based_utils.py) contains shared
  gate-model training and batching utilities.

Model families:

- Photonic models:
  [`models/photonic_models/`](models/photonic_models/)
  - `dressed_quantum_circuit`
  - `dressed_quantum_circuit_reservoir`
  - `multiple_paths_model`
  - `multiple_paths_model_reservoir`
  - `data_reuploading`
  - `q_kernel_method`
  - `q_kernel_method_reservoir`
  - `q_rks`
- Gate-based models:
  [`models/gate_based_models/`](models/gate_based_models/)
  - `dressed_quantum_circuit`
  - `dressed_quantum_circuit_reservoir`
  - `multiple_paths_model`
  - `multiple_paths_model_reservoir`
  - `data_reuploading`
  - `q_kernel_method_reservoir`
  - `q_rks`
- Classical baselines:
  [`models/classical_models/`](models/classical_models/)
  - `mlp`
  - `rbf_svc`
  - `rks`

Notes:

- Classical models automatically use backend `classical`, even if another
  backend is supplied.
- `q_kernel_method` is not supported on the gate backend; use
  `q_kernel_method_reservoir` for the gate/IQP kernel path.
- Model ablations use suffixes:
  - `_abla_q` for replacing a supported quantum block.
  - `_abla_c` for freezing supported classical parameters.

### Training and run orchestration

The run layer lives in [`run_scripts/`](run_scripts/) and
[`training/`](training/).

- [`run_scripts/run.py`](run_scripts/run.py) is the main orchestration module:
  `run_single()`, `run_search()`, `set_up_logging()`, and
  `set_up_random_state()`.
- [`run_scripts/single_run.py`](run_scripts/single_run.py) builds single-run
  hyperparameter dictionaries from JSON config and architecture strings.
- [`run_scripts/hyperparam_search_run.py`](run_scripts/hyperparam_search_run.py)
  builds HP-search grids/spaces for `minimal` and `full` profiles, including
  grid, halving-grid, and Bayesian-search compatible formats.
- [`run_scripts/run_output_utils.py`](run_scripts/run_output_utils.py) saves
  HP-search summaries and long-training event aggregates.
- [`training/distribute_training.py`](training/distribute_training.py) dispatches
  each model type to the correct training implementation.
- [`training/training_torch.py`](training/training_torch.py) contains PyTorch,
  Merlin data-reuploading, and sklearn quantum-kernel training paths.
- [`training/training_scikit_learn.py`](training/training_scikit_learn.py)
  handles sklearn and sklearn-kernel baselines.
- [`training/gate_based_training/`](training/gate_based_training/) contains
  gate-model training utilities and sklearn wrappers.

### Hyperparameters

Hyperparameter configuration is stored as JSON under
[`hyperparameters/`](hyperparameters/).

- [`hyperparameters/single_run/`](hyperparameters/single_run/) contains
  defaults for one training/evaluation run:
  - `dataset_hps.json`
  - `photonic_model_hps.json`
  - `gate_model_hps.json`
  - `classical_model_hps.json`
  - `training_hps.json`
  - `generator.py`
- [`hyperparameters/hyperparam_search/`](hyperparameters/hyperparam_search/)
  contains full search definitions and search assignment metadata:
  - `dataset_hps.json`
  - `model_search_assignment.json`
  - `halving_grid/`
  - `bayes/`
- [`hyperparameters/hyperparam_search/minimal/`](hyperparameters/hyperparam_search/minimal/)
  contains reduced grid-search spaces for faster smoke-test and comparison runs.

HP profiles:

1. `minimal`
2. `full`

### Batch scripts

Convenience shell scripts live in [`scripts/`](scripts/). Read
[`scripts/README.md`](scripts/README.md) for exact behavior and supported
environment variables.

Common scripts:

- `run_all_single.sh`: one single run per model/backend for a fixed dataset.
- `run_all_single_ablation.sh`: single runs plus compatible ablation variants.
- `run_photonic_hp_search_minimal.sh`: minimal HP search for photonic models.
- `run_gate_hp_search_minimal.sh`: minimal HP search for gate models.
- `run_classical_hp_search_minimal.sh`: minimal HP search for classical models.
- `run_all_hp_search_minimal.sh`: minimal HP search across all model families.
- `run_all_hp_search_minimal_ablation.sh`: minimal HP search with optional
  ablations.
- `run_all_hp_search.sh`: legacy broad HP-search sweep.
- `run_all.sh`: top-level orchestrator driven by `run_all_config.json`.

All scripts support `RANDOM_STATE`. Single-run and supported HP-search paths
also support `MAX_TRAIN_TIME_SECONDS`.

### Utilities and analysis

Utility scripts live in [`utils/`](utils/). Start with
[`utils/README.md`](utils/README.md).

Runnable utilities:

- [`utils/visualize_dataset.py`](utils/visualize_dataset.py): plot raw or
  preprocessed 2D dataset views.
- [`utils/visu_hp_minimal.py`](utils/visu_hp_minimal.py): visualize minimal
  HP-search summaries. Guide:
  [`utils/visu_hp_minimal_guide.md`](utils/visu_hp_minimal_guide.md).
- [`utils/visu_ablation_runs.py`](utils/visu_ablation_runs.py): visualize
  ablation run summaries. Guide:
  [`utils/visu_ablation_runs_guide.md`](utils/visu_ablation_runs_guide.md).
- [`utils/dataset_complexity.py`](utils/dataset_complexity.py): export ARFFs,
  compute/plot dataset-complexity and baseline summaries. Guide:
  [`utils/dataset_complexity_guide.md`](utils/dataset_complexity_guide.md).
- [`utils/analyze_model_outputs.py`](utils/analyze_model_outputs.py): inspect
  model-output behavior across configurations. Guide:
  [`utils/analyze_model_outputs_guide.md`](utils/analyze_model_outputs_guide.md).
- [`utils/summarize_long_training.py`](utils/summarize_long_training.py):
  summarize timeout/cut-short JSONL reports.

Helper modules:

- [`utils/save_metrics.py`](utils/save_metrics.py): writes accuracy/loss curves,
  final metrics, hyperparameters, and search metadata.
- [`utils/photonic_dims.py`](utils/photonic_dims.py): computes default photonic
  `(m, n)` from feature dimension.
- [`utils/long_training_events.py`](utils/long_training_events.py): writes
  timeout/cut-short event JSONL and CSV files.

### Results and kept outputs

- [`results/`](results/) is the default output location for new runs.
- [`kept_results/`](kept_results/) stores curated/reference outputs such as
  dataset plots, complexity figures, and benchmark artifacts worth preserving.
- [`wandb/`](wandb/) contains local Weights & Biases run state when W&B logging is
  enabled.

Single `main.py` runs write to:

```text
tabular_data/results/{run_type}_{timestamp}/{model}_{backend}_{dataset}
```

Bulk scripts usually write to:

```text
tabular_data/results/{big_script_name}/{run_type}_{timestamp}/{model}_{backend}_{dataset}
```

Where `run_type` is `single_run` or `hp_search`, and `backend` is `photonic`,
`gate`, or `classical`.

### Merlin additions

[`merlin_additional/`](merlin_additional/) contains supporting Merlin/perceval
code used by photonic kernels and reuploading experiments:

- `quantum_kernels.py`
- `loss.py`
- `utils.py`
- `reuploading_merlin/` experiments, notebooks, plotting helpers, and paper
  datasets.

## Using `main.py`

Run commands from inside `tabular_data/`:

```bash
cd tabular_data
```

List supported datasets, models, run types, and HP profiles:

```bash
python main.py --list
```

Minimal usage, with photonic backend by default:

```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type}
```

Run a gate-based model:

```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type} --backend gate
```

Run a classical model:

```bash
python main.py --dataset {dataset_name} --model mlp --run_type single
```

Classical models ignore `--backend` and are routed to backend `classical`.

Specify an architecture:

```bash
python main.py --dataset {dataset_name} --model {model_name} --architecture {architecture} --run_type {run_type}
```

Ask for architecture formatting help:

```bash
python main.py --dataset {dataset_name} --model {model_name} --architecture help --run_type {run_type}
```

Run HP search with the default minimal profile, or the full profile:

```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type hyperparam_search --hp_profile minimal
python main.py --dataset {dataset_name} --model {model_name} --run_type hyperparam_search --hp_profile full
```

Disable W&B logging:

```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type single --no-wandb
```

Set reproducibility and runtime budget:

```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type single --random_state 7 --max_train_time_seconds 900
```

## Backend and Architecture Reference

Backends:

1. `photonic`
2. `gate`
3. `classical` for classical models

Run types:

1. `single`
2. `hyperparam_search`

Architecture strings depend on model and backend. The canonical source is:

```bash
python main.py --dataset {dataset_name} --model {model_name} --backend {backend} --architecture help --run_type single
```

Examples:

- Photonic `dressed_quantum_circuit`: `m_{i}_n_{j}`
- Photonic `multiple_paths_model`: `m_{i}_n_{j}_numNeurons_{k}_{l}_...`
- Photonic `data_reuploading`: `numLayers_{i}`
- Photonic `q_rks`: `m_{i}_n_{j}_R_{k}_gamma_{l}`
- Gate `dressed_quantum_circuit`: `numLayers_{i}`
- Gate `multiple_paths_model`: `numLayers_{i}_numNeurons_{j}_{k}_...`
- Gate `q_kernel_method_reservoir`: `repeats_{i}`
- Classical `mlp`: `numNeurons_{i}_{j}_{k}_...`
- Classical `rbf_svc`: `C_{f}`
- Classical `rks`: `R_{i}_gamma_{j}`

## Photonic Defaults

- Data scaling: `minmax`
- Scaling layer: `pi`
- Photonic input state for m/n-based models: `spaced`
- Default photonic dimension mode: `feature_plus_one`
  - `m = n_features + 1`
  - `n = ceil(n_features / 2)`
- Override default mode with:

```bash
PHOTONIC_DIM_MODE=feature_equal python main.py ...
PHOTONIC_DIM_MODE=feature_double python main.py ...
```

Accepted `PHOTONIC_DIM_MODE` values:

1. `feature_plus_one`
2. `feature_equal`
3. `feature_double`

## Special Thanks

Anthony Walsh for the Quantum Kernel Method implementation.

Hugo Izadi for the Data Re-uploading model implementation.

Bowles, Ahmed and Schuld for their benchmarking implementation.

Cassandre Notton for her support throughout all steps of this project.
