# Tabular Benchmarking

This folder contains the tabular benchmarking pipeline (datasets, models, training, and result logging).

# Usage
Minimal usage (photonic backend by default):
```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type}
```

Run a gate-based model:
```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type} --backend gate
```

Specify an architecture:
```bash
python main.py --dataset {dataset_name} --model {model_name} --architecture {architecture} --run_type {run_type}
```

List supported values (datasets/models/run types/profiles):
```bash
python main.py --list
```

Reduce HP search size (default is minimal) or expand to full:
```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type hyperparam_search --hp_profile minimal
python main.py --dataset {dataset_name} --model {model_name} --run_type hyperparam_search --hp_profile full
```

# Datasets
Dataset names follow:
1. `downscaled_mnist_pca_{d}`
2. `hidden_manifold_{d}_{m}`
3. `two_curves_{d}_{D}`

With `d`, `m`, `D` in `[2, 3, ... , 20]`.

# Models
Quantum models:
- `dressed_quantum_circuit`
- `dressed_quantum_circuit_reservoir`
- `multiple_paths_model`
- `multiple_paths_model_reservoir`
- `data_reuploading`
- `q_kernel_method`
- `q_kernel_method_reservoir`
- `q_rks`

Classical models:
- `mlp`
- `rbf_svc`
- `rks`

# Run Types
1. `single`
2. `hyperparam_search`

# Backend
1. `photonic`
2. `gate`
3. (implicit) `classical` for classical models

# Architectures
Architectures depend on the model/backend.
Use the built-in help:
```bash
python main.py --dataset {dataset_name} --model {model_name} --architecture help --run_type {run_type}
```

# Photonic Defaults
- Data scaling: `minmax`
- Scaling layer: `pi`
- Photonic input state (for m/n-based models): `spaced`, which yields `[1,0,1,0,1,0,1,0,1,0]` when `m=10`, `n=5`
- Modes/photons: `m=10`, `n=5` for all photonic models that accept `m`/`n`

# Results Layout
All runs write to:
```
tabular_data/results/{dataset}/{model}_{backend}/{architecture}/{timestamp}
```

# Special Thanks
Anthony Walsh for the Quantum Kernel Method implementation
Hugo Izadi for the Data Re-uploading model implementation
Bowles, Ahmed and Schuld for their benchmarking implementation
