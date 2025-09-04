import torch
import logging
import json

from prompt_toolkit.utils import to_int


def get_dataset_hps(dataset_name, model, backend):
    # Need labels -1 vs 1 for q_kernel_method and for gate_based models
    if model == 'q_kernel_method' or backend == 'gate':
        labels_treatment = ['-1_1']
    else:
        labels_treatment = ['0_1']

    # Only keep 250 training samples and 250 test samples if utilizing a kernel method and if using the
    # downscaled_mnist_pca dataset, because the two others already have a low amount of data points.
    if dataset_name == 'downscaled_mnist_pca' and model in ['q_kernel_method', 'q_kernel_method_reservoir', 'q_rks', 'rbf_svc', 'rks']:
        num_train = 250
        num_test = 250
    else:
        num_train = None
        num_test = None

    # Load other hps
    with open("hyperparameters/hyperparam_search/dataset_hps.json", "r") as f:
        hps = json.load(f)

    dataset_hps = hps[dataset_name]
    dataset_hps['labels_treatment'] = labels_treatment
    dataset_hps['num_train'] = [num_train]
    dataset_hps['num_test'] = [num_test]

    return dataset_hps


def get_model_hps_halving_grid_photonic(model, architecture):
    hp_path = "hyperparameters/hyperparam_search/halving_grid/photonic_model_hps.json"

    # Load hps
    with open(hp_path, "r") as f:
        hps = json.load(f)

    # Access model hps
    try:
        model_hps = hps[model]
    except Exception:
        raise Exception(f'Model {model} not found in hyperparams dictionary.')
    model_hps = model_hps['default']

    # Modify some hps based on architecture
    if architecture != 'default':
        # numNeurons is always the last hp presented in the architecture string
        architecture_args = architecture.split('_')
        for i in range(0, len(architecture_args), 2):
            hp = architecture_args[i]
            if hp == 'numNeurons':
                numNeurons = []
                numLayers = len(architecture_args[i + 1:])
                if numLayers == 0:
                    model_hps['numNeurons'] = [numNeurons]
                    break
                else:
                    for layer in range(numLayers):
                        numNeurons.append(int(architecture_args[i + 1 + layer]))
                    model_hps['numNeurons'] = [numNeurons]
                break

            hp_value = int(architecture_args[i + 1])
            model_hps[hp] = [hp_value]

    # Define model type
    if model in ['dressed_quantum_circuit', 'dressed_quantum_circuit_reservoir', 'multiple_paths_model', 'multiple_paths_model_reservoir']:
        model_hps['type'] = ['torch']
    elif model in ['data_reuploading', 'data_reuploading_reservoir']:
        model_hps['type'] = ['reuploading']
    elif model in ['q_rks']:
        model_hps['type'] = ['sklearn_kernel']
    elif model in ['q_kernel_method', 'q_kernel_method_reservoir']:
        model_hps['type'] = ['sklearn_q_kernel']
    else:
        raise Exception(f'Model {model} has no defined type.')

    # Keep model name
    model_hps['name'] = [model + f'_({architecture})']
    return model_hps


def get_model_hps_bayes_photonic(model, architecture):
    hp_path = "hyperparameters/hyperparam_search/bayes/photonic_model_hps.json"

    #TODO

    # Load hps
    with open(hp_path, "r") as f:
        hps = json.load(f)

    # Access model hps
    try:
        model_hps = hps[model]
    except Exception:
        raise Exception(f'Model {model} not found in hyperparams dictionary.')
    model_hps = model_hps['default']

    # Modify some hps based on architecture
    if architecture != 'default':
        # numNeurons is always the last hp presented in the architecture string
        architecture_args = architecture.split('_')
        for i in range(0, len(architecture_args), 2):
            hp = architecture_args[i]
            if hp == 'numNeurons':
                numNeurons = []
                numLayers = len(architecture_args[i + 1:])
                if numLayers == 0:
                    model_hps['numNeurons'] = [numNeurons]
                    break
                else:
                    for layer in range(numLayers):
                        numNeurons.append(int(architecture_args[i + 1 + layer]))
                    model_hps['numNeurons'] = [numNeurons]
                break

            hp_value = int(architecture_args[i + 1])
            model_hps[hp] = [hp_value]

    # Define model type
    if model in ['dressed_quantum_circuit', 'dressed_quantum_circuit_reservoir', 'multiple_paths_model',
                 'multiple_paths_model_reservoir']:
        model_hps['type'] = ['torch']
    elif model in ['data_reuploading', 'data_reuploading_reservoir']:
        model_hps['type'] = ['reuploading']
    elif model in ['q_rks']:
        model_hps['type'] = ['sklearn_kernel']
    elif model in ['q_kernel_method', 'q_kernel_method_reservoir']:
        model_hps['type'] = ['sklearn_q_kernel']
    else:
        raise Exception(f'Model {model} has no defined type.')

    # Keep model name
    model_hps['name'] = [model + f'_({architecture})']
    return model_hps


def get_model_hps_halving_grid_gate(model, architecture, random_state):
    hp_path = "hyperparameters/hyperparam_search/halving_grid/gate_model_hps.json"

    # Load hps
    with open(hp_path, "r") as f:
        hps = json.load(f)

    # Access model hps
    try:
        model_hps = hps[model]
    except Exception:
        raise Exception(f'Model {model} not found in hyperparams dictionary.')
    model_hps = model_hps['default']

    # Set random state
    model_hps['random_state'] = [random_state]

    # Modify some hps based on architecture
    if architecture != 'default':
        # numNeurons is always the last hp presented in the architecture string
        architecture_args = architecture.split('_')
        for i in range(0, len(architecture_args), 2):
            hp = architecture_args[i]
            if hp == 'numNeurons':
                numNeurons = []
                numLayers = len(architecture_args[i + 1:])
                if numLayers == 0:
                    model_hps['numNeurons'] = [numNeurons]
                    break
                else:
                    for layer in range(numLayers):
                        numNeurons.append(int(architecture_args[i + 1 + layer]))
                    model_hps['numNeurons'] = [numNeurons]
                break

            hp_value = int(architecture_args[i + 1])
            model_hps[hp] = [hp_value]

    # Define model type
    if model in ['dressed_quantum_circuit', 'dressed_quantum_circuit_reservoir', 'multiple_paths_model', 'multiple_paths_model_reservoir', 'data_reuploading']:
        model_hps['type'] = ['jax_sklearn_gate']
    elif model in ['q_rks']:
        model_hps['type'] = ['gate_rks']
    else:
        model_hps['type'] = ['sklearn_gate']

    # Keep model name
    model_hps['name'] = [model + f'_({architecture})']
    return model_hps


def get_model_hps_bayes_gate(model, architecture, random_state):
    hp_path = "hyperparameters/hyperparam_search/bayes/gate_model_hps.json"

    #TODO

    # Load hps
    with open(hp_path, "r") as f:
        hps = json.load(f)

    # Access model hps
    try:
        model_hps = hps[model]
    except Exception:
        raise Exception(f'Model {model} not found in hyperparams dictionary.')
    model_hps = model_hps['default']

    # Set random state
    model_hps['random_state'] = random_state

    # Modify some hps based on architecture
    if architecture != 'default':
        architecture_args = architecture.split('_')
        for i in range(0, len(architecture_args), 2):
            hp = architecture_args[i]
            hp_value = int(architecture_args[i + 1])

            model_hps[hp] = hp_value
            if hp == 'numClassicalHLayers':
                model_hps['numNeurons'] = [4] * hp_value

    # Define model type
    if model in ['dressed_quantum_circuit', 'dressed_quantum_circuit_reservoir', 'multiple_paths_model', 'multiple_paths_model_reservoir', 'data_reuploading']:
        model_hps['type'] = 'jax_sklearn_gate'
    elif model in ['q_rks']:
        model_hps['type'] = 'gate_rks'
    else:
        model_hps['type'] = 'sklearn_gate'

    # Keep model name
    model_hps['name'] = model + f'_({architecture})'
    return model_hps


def get_model_hps_halving_grid_classical(model, architecture, random_state):
    # Handle custom mlp architecture
    if model == 'mlp' and architecture != 'default':
        architecture_split = architecture.split('_')
        assert architecture_split[0] == 'numNeurons', 'Wrong formatting for architecture: "numNeurons_i_j_k_l_..." where each index is the number of neurons in its layer.'
        num_neurons = architecture_split[1:]
        num_neurons = [int(num) for num in num_neurons]
        hps = {'numNeurons': num_neurons}
        model_hps = hps

    # All other cases
    else:
        # Load hps
        with open("hyperparameters/hyperparam_search/halving_grid/classical_model_hps.json", "r") as f:
            hps = json.load(f)

        # Access model hps
        try:
            model_hps = hps[model]
        except Exception:
            raise Exception(f'Model {model} not found in hyperparams dictionary.')
        model_hps = model_hps['default']

        # Modify some hps based on architecture
        if architecture != 'default':
            architecture_args = architecture.split('_')
            for i in range(0, len(architecture_args), 2):
                hp = architecture_args[i]
                hp_value = architecture_args[i + 1]

                if hp == 'C':
                    hp_value = float(hp_value)
                else:
                    hp_value = int(hp_value)

                model_hps[hp] = [hp_value]

    # Setup random state
    model_hps['random_state'] = [random_state]

    # Define model type
    if model == 'mlp':
        model_hps['type'] = ['torch']
    elif model == 'rbf_svc':
        model_hps['type'] = ['sklearn']
    elif model == 'rks':
        model_hps['type'] = ['sklearn_kernel']
    else:
        raise Exception(f'Model {model} has no defined type.')

    # Keep model name
    model_hps['name'] = [model + f'_({architecture})']
    return model_hps


def get_model_hps_bayes_classical(model, architecture, random_state):
    #TODO

    # Handle custom mlp architecture
    if model == 'mlp' and architecture != 'default':
        architecture_split = architecture.split('_')
        assert architecture_split[0] == 'numNeurons', 'Wrong formatting for architecture: "numNeurons_i_j_k_l_..." where each index is the number of neurons in its layer.'
        num_neurons = architecture_split[1:]
        num_neurons = [int(num) for num in num_neurons]
        hps = {'numNeurons': num_neurons}
        model_hps = hps

    # All other cases
    else:
        # Load hps
        with open("hyperparameters/hyperparam_search/bayes/classical_model_hps.json", "r") as f:
            hps = json.load(f)

        # Access model hps
        try:
            model_hps = hps[model]
        except Exception:
            raise Exception(f'Model {model} not found in hyperparams dictionary.')
        model_hps = model_hps['default']

        # Modify some hps based on architecture
        if architecture != 'default':
            architecture_args = architecture.split('_')
            for i in range(0, len(architecture_args), 2):
                hp = architecture_args[i]
                hp_value = architecture_args[i + 1]

                if hp == 'C':
                    hp_value = float(hp_value)
                else:
                    hp_value = int(hp_value)

                model_hps[hp] = hp_value

    # Setup random state
    model_hps['random_state'] = random_state

    # Define model type
    if model == 'mlp':
        model_hps['type'] = 'torch'
    elif model == 'rbf_svc':
        model_hps['type'] = 'sklearn'
    elif model == 'rks':
        model_hps['type'] = 'sklearn_kernel'
    else:
        raise Exception(f'Model {model} has no defined type.')

    # Keep model name
    model_hps['name'] = model + f'_({architecture})'
    return model_hps


def get_training_hps_halving_grid(model_type, dataset_name, model):
    # Determine number of epochs for training
    if dataset_name == 'downscaled_mnist_pca':
        epochs = 5
        if model_type == 'sklearn_q_kernel':
            epochs = 2
    elif dataset_name == 'hidden_manifold':
        epochs = 25
        if model_type == 'sklearn_q_kernel':
            epochs = 10
    elif dataset_name == 'two_curves':
        epochs = 25
        if model_type == 'sklearn_q_kernel':
            epochs = 10
    else:
        raise Exception(f'Dataset name {dataset_name} not found.')

    # Determine if pre_train for q_kernel_method
    if model == 'q_kernel_method_reservoir':
        pre_train = False
    else:
        pre_train = True

    # Load hps
    with open("hyperparameters/hyperparam_search/halving_grid/training_hps.json", "r") as f:
        hps = json.load(f)

    # Access training hps
    try:
        training_hps = hps[model_type]
    except Exception:
        raise Exception(f'Model type {model_type} not found in hyperparams dictionary.')

    # Setup device
    training_hps['device'] = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    # Setup number of epochs
    training_hps['epochs'] = [epochs]
    # Setup pre_train
    training_hps['pre_train'] = [pre_train]

    return training_hps


def get_training_hps_bayes(model_type, dataset_name, model):
    # Determine number of epochs for training
    if dataset_name == 'downscaled_mnist_pca':
        epochs = 5
        if model_type == 'sklearn_q_kernel':
            epochs = 2
    elif dataset_name == 'hidden_manifold':
        epochs = 25
        if model_type == 'sklearn_q_kernel':
            epochs = 10
    elif dataset_name == 'two_curves':
        epochs = 25
        if model_type == 'sklearn_q_kernel':
            epochs = 10
    else:
        raise Exception(f'Dataset name {dataset_name} not found.')

    # Determine if pre_train for q_kernel_method
    if model == 'q_kernel_method_reservoir':
        pre_train = False
    else:
        pre_train = True

    # Load hps
    with open("hyperparameters/hyperparam_search/bayes/training_hps.json", "r") as f:
        hps = json.load(f)

    # Access training hps
    try:
        training_hps = hps[model_type]
    except Exception:
        raise Exception(f'Model type {model_type} not found in hyperparams dictionary.')

    # Setup device
    training_hps['device'] = [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
    # Setup number of epochs
    training_hps['epochs'] = [epochs]
    # Setup pre_train
    training_hps['pre_train'] = [pre_train]

    return training_hps


def get_hyperparams_halving_grid(dataset, model, architecture, backend, sk_random):
    # Dataset HPs #####################################################
    # List of allowed dataset names
    dataset_names = ['downscaled_mnist_pca', 'hidden_manifold', 'two_curves']

    # Find which dataset_name matches the start of the string
    dataset_name = next(name for name in dataset_names if dataset.startswith(name))

    # Remove dataset_name + underscore and split the rest
    args_part = dataset[len(dataset_name) + 1:]  # skip underscore
    args = args_part.split("_") if args_part else []

    arg1 = args[0] if len(args) >= 1 else None
    arg2 = args[1] if len(args) >= 2 else None

    dataset_hps = get_dataset_hps(dataset_name, model, backend)

    # Model HPs #####################################################
    if backend == 'photonic':
        model_hps = get_model_hps_halving_grid_photonic(model, architecture)
    elif backend == 'gate':
        model_hps = get_model_hps_halving_grid_gate(model, architecture, sk_random)
    elif backend == 'classical':
        model_hps = get_model_hps_halving_grid_classical(model, architecture, sk_random)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    # Training HPs #####################################################
    model_type = model_hps['type'][0]
    training_hps = get_training_hps_halving_grid(model_type, dataset_name, model)

    try:
        device = training_hps['device'][0]
        if device == torch.device('cuda'):
            logging.warning('Training on GPU, cuda available.')
        elif device == torch.device('cpu'):
            logging.warning('Training on CPU.')
        else:
            raise NotImplementedError(f'Unknown device: {device}')
    except KeyError:
        logging.warning('Training on CPU.')

    param_grid = {**{f"data_params__{k}": v for k, v in dataset_hps.items()},
                  **{f"model_params__{k}": v for k, v in model_hps.items()},
                  **{f"training_params__{k}": v for k, v in training_hps.items()},
                  'model_params__input_size': [int(arg1)]}
    param_grid['model_params__output_size'] = [int(param_grid['training_params__output_size'][0])]
    return param_grid, dataset_hps, model_hps, training_hps


def get_hyperparams_bayes(dataset, model, architecture, backend, sk_random):
    # Dataset HPs #####################################################
    # List of allowed dataset names
    dataset_names = ['downscaled_mnist_pca', 'hidden_manifold', 'two_curves']

    # Find which dataset_name matches the start of the string
    dataset_name = next(name for name in dataset_names if dataset.startswith(name))

    # Remove dataset_name + underscore and split the rest
    args_part = dataset[len(dataset_name) + 1:]  # skip underscore
    args = args_part.split("_") if args_part else []

    arg1 = args[0] if len(args) >= 1 else None
    arg2 = args[1] if len(args) >= 2 else None

    dataset_hps = get_dataset_hps(dataset_name, arg1, arg2, model, backend)

    # Model HPs #####################################################
    if backend == 'photonic':
        model_hps = get_model_hps_bayes_photonic(model, architecture)
    elif backend == 'gate':
        model_hps = get_model_hps_bayes_gate(model, architecture, sk_random)
    elif backend == 'classical':
        model_hps = get_model_hps_bayes_classical(model, architecture, sk_random)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    # Training HPs #####################################################
    model_type = model_hps['type'][0]
    training_hps = get_training_hps_bayes(model_type, dataset_name, model)

    try:
        device = training_hps['device'][0]
        if device == torch.device('cuda'):
            logging.warning('Training on GPU, cuda available.')
        elif device == torch.device('cpu'):
            logging.warning('Training on CPU.')
        else:
            raise NotImplementedError(f'Unknown device: {device}')
    except KeyError:
        logging.warning('Training on CPU.')

    param_grid = {**{f"data_params__{k}": v for k, v in dataset_hps.items()},
                  **{f"model_params__{k}": v for k, v in model_hps.items()},
                  **{f"training_params__{k}": v for k, v in training_hps.items()},
                  'model_params__input_size': [int(arg1)]}
    param_grid['model_params__output_size'] = [int(param_grid['training_params__output_size'][0])]
    return param_grid, dataset_hps, model_hps, training_hps