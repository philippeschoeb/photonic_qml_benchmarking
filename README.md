# photonic_qml_benchmarking

## Overview
Benchmark suite for quantum and classical models. The repository currently focuses on photonic circuits (via MerLin backend), gate-based models implemented with PennyLane/JAX, and baseline classical estimators. This is a work in progress so the benchmarking study is solely on tabular data for now, but it will cover vision data and time series data eventually.

## Environment Setup
- Create a virtual environment (Python 3.10+ recommended) and install the Python dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # or .venv\Scripts\activate on Windows
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Install the Merlin photonic simulator that matches your hardware. Merlin is not published on PyPI; follow the internal installation instructions from Xanadu. Place the resulting package on your `PYTHONPATH` so imports like `import merlin as ml` succeed.
- For GPU acceleration make sure the CUDA toolkit versions used by PyTorch and JAX match the drivers on your machine. Installing the wheels linked from https://pytorch.org and https://jax.readthedocs.io/en/latest/installation.html is recommended.

## Data Availability
The tabular datasets used in this study (`downscaled_mnist_pca`, `hidden_manifold`, `two_curves`) are expected to be present locally. If you already have direct access, no extra download is required. When running in a fresh environment you can optionally obtain them through PennyLane's `qml.data` utilities (`qml.data.load("other", name="hidden-manifold")`, etc.).

## Running Benchmarks
All tabular experiments are launched from `tabular_data/main.py`.

Single configuration:
```bash
cd tabular_data
python main.py --dataset {dataset} --model {model} --run_type single
```

Hyperparameter search:
```bash
cd tabular_data
python main.py --dataset {dataset} --model {model} --run_type hyperparam_search --backend {backend}
```

Use `--architecture help` to inspect model-specific architecture strings. Classical models automatically switch to the classical backend.

## GPU vs CPU Notes
- Photonic simulations (MerLin) are CPU-heavy; run with `--backend photonic`. If you encounter worker crashes during large hyperparameter sweeps, prefer serial execution (`n_jobs=1`) or trim the search space.
- JAX-based gate models can leverage GPU execution when `jaxlib` is installed with CUDA support. Ensure `XLA_PYTHON_CLIENT_PREALLOCATE=false` is set if GPU memory becomes constrained.

## Repository Layout
- `tabular_data/` – core pipelines for datasets, models, training loops, hyperparameter definitions.
- `time_series_data/`, `vision_data/` – placeholders for future modalities.
- `tabular_data/results/` – metrics and artifacts written by completed runs.

