# Special thanks to:

Anthony Walsh for the Quantum Kernel Method implementation

Hugo Izadi for the Data Re-uploading model implementation

Bowles, Ahmed and Schuld for their benchmarking implementation

...

# Reminders

- Set up logging in main function
- Set up random state in main function

### Hyperparams

-Batch_size to consider

# Usage
Minimal usage (for photonic based models):
```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type}
```
Running a gate based model:
```bash
python main.py --dataset {dataset_name} --model {model_name} --run_type {run_type} --backend gate
```
You can specify an architecture:
```bash
python main.py --dataset {dataset_name} --model {model_name} --architecture {architecture} --run_type {run_type}
```

Different model names:
1. Quantum models
- dressed_quantum_circuit
- dressed_quantum_circuit_reservoir
- multiple_paths_model
- multiple_paths_model_reservoir
- data_reuploading
- q_kernel_method
- q_rks

2. Classical models
- mlp
- rbf_svc
- rks

Different datasets:
1. downscaled_mnist_pca_{d}
2. hidden_manifold_{d}_{m}
3. two_curves_{d}_{D}

with d in [2, 3, ... , 20]; m in [2, 3, ... , 20]; D in [2, 3, ... , 20]

Two run types:
1. single
2. hyperparam_search

Backend:
1. photonic
2. gate
3. {empty}

Architectures depend on the model chosen