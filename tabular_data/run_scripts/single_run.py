import torch
import logging

from prompt_toolkit.utils import to_int


def get_dataset_hps(dataset_name, arg1, arg2, model):
    # Need labels -1 vs 1 for q_kernel_method
    if model == 'q_kernel_method':
        labels_treatment = 'q_kernel'
    else:
        labels_treatment = '0-1'

    # Only keep 250 training samples and 250 test samples if utilizing a kernel method
    if model in ['q_kernel_method', 'q_kernel_method_reservoir', 'q_rks', 'rbf_svc', 'rks']:
        num_train = 250
        num_test = 250
    else:
        num_train = None
        num_test = None

    hps = {'downscaled_mnist_pca': {'scaling': 'minmax', 'batch_size': 32, 'labels_treatment': labels_treatment, 'num_train': num_train, 'num_test': num_test},
           'hidden_manifold': {'scaling': 'minmax', 'batch_size': 32, 'labels_treatment': labels_treatment},
           'two_curves': {'scaling': 'minmax', 'batch_size': 32, 'labels_treatment': labels_treatment},}

    return hps[dataset_name]


def get_model_hps_photonic(model, architecture):
    hps = {'dressed_quantum_circuit': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'reservoir': False, 'no_bunching': True},
                                       **{f'm_{i}_n_{j}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'reservoir': False, 'no_bunching': True}
                                       for i in range(2, 21)
                                       for j in range(1, 11)}},
           'dressed_quantum_circuit_reservoir': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'reservoir': True, 'no_bunching': True},
                                                 **{f'm_{i}_n_{j}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'reservoir': True, 'no_bunching': True}
                                                 for i in range(2, 21)
                                                 for j in range(1, 11)}},
           'multiple_paths_model': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'reservoir': False, 'no_bunching': True, 'post_circuit_scaling': 'standardize', 'num_h_layers': 0, 'num_neurons': []},
                                    **{f'm_{i}_n_{j}_num_h_layers_{k}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'reservoir': False, 'no_bunching': True, 'post_circuit_scaling': 'standardize', 'num_h_layers': k, 'num_neurons': [4] * k}
                                    for i in range(2, 21)
                                    for j in range(1, 11)
                                    for k in range(0, 4)}},
           'multiple_paths_model_reservoir': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'reservoir': True, 'no_bunching': True, 'post_circuit_scaling': 'standardize', 'num_h_layers': 0, 'num_neurons': []},
                                   **{f'm_{i}_n_{j}_num_h_layers_{k}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'reservoir': True, 'no_bunching': True, 'post_circuit_scaling': 'standardize', 'num_h_layers': k, 'num_neurons': [4] * k}
                                   for i in range(2, 21)
                                   for j in range(1, 11)
                                   for k in range(0, 4)}},
           'data_reuploading': {'default': {'scaling': '1/pi', 'num_layers': 10, 'design': 'AA'},
                                **{f'num_layers_{i}': {'scaling': '1/pi', 'num_layers': i, 'design': 'AA'}
                                   for i in range(2, 21)}},
           'data_reuploading_reservoir': {'default': {'scaling': '1/pi', 'num_layers': 10, 'design': 'AD'},
                                **{f'num_layers_{i}': {'scaling': '1/pi', 'num_layers': i, 'design': 'AA'}
                                   for i in range(2, 21)}},
           'q_kernel_method': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'no_bunching': True, 'pre_train': True, 'C': 1.0},
                                **{f'm_{i}_n_{j}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'no_bunching': True, 'pre_train': True, 'C': 1.0}
                                for i in range(2, 21)
                                for j in range(1, 11)}},
           'q_kernel_method_reservoir': {'default': {'scaling': '1', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'no_bunching': True, 'pre_train': False, 'C': 1.0},
                                **{f'm_{i}_n_{j}': {'scaling': '1', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'no_bunching': True, 'pre_train': False, 'C': 1.0}
                                for i in range(2, 21)
                                for j in range(1, 11)}},
           'q_rks': {'default': {'scaling': '1/pi', 'm': 20, 'n': 4, 'circuit': 'generic_mzi', 'no_bunching': True, 'R': 50, 'gamma': 3, 'C': 1.0},
                     **{f'm_{i}_n_{j}_R_{k}_gamma_{l}': {'scaling': '1/pi', 'm': i, 'n': j, 'circuit': 'generic_mzi', 'no_bunching': True, 'R': k, 'gamma': l, 'C': 1.0}
                        for i in range(2, 21)
                        for j in range(1, 11)
                        for k in range(10, 110, 10)
                        for l in range(1, 10)}}
           }
    # Access model hyperparams
    try:
        model_hps = hps[model]
    except Exception:
        raise Exception(f'Model {model} not found in hyperparams dictionary.')
    try:
        model_hps = model_hps[architecture]
    except Exception:
        raise Exception(f'Architecture {architecture} of model {model} not found in hyperparams dictionary.')

    # Define model type
    if model in ['dressed_quantum_circuit', 'dressed_quantum_circuit_reservoir', 'multiple_paths_model', 'multiple_paths_model_reservoir']:
        model_hps['type'] = 'torch'
    elif model in ['data_reuploading', 'data_reuploading_reservoir']:
        model_hps['type'] = 'reuploading'
    elif model in ['q_rks']:
        model_hps['type'] = 'sklearn_kernel'
    elif model in ['q_kernel_method', 'q_kernel_method_reservoir']:
        model_hps['type'] = 'sklearn_q_kernel'
    else:
        raise Exception(f'Model {model} has no defined type.')

    # Keep model name
    model_hps['name'] = model + f'_({architecture})'
    return model_hps


def get_model_hps_gate(model, architecture):
    #TODO
    return {}


def get_model_hps_classical(model, architecture, random_state):
    if model == 'mlp' and architecture != 'default':
        architecture_split = architecture.split('_')
        assert architecture_split[0] == 'neurons', 'Wrong formatting for architecture: "neurons_i_j_k_l_..." where each index is the number of neurons in its layer.'
        num_neurons = architecture_split[1:]
        num_neurons = [int(num) for num in num_neurons]
        num_h_layers = len(num_neurons)
        hps = {'num_h_layers': num_h_layers, 'num_neurons': num_neurons}
        model_hps = hps
    else:
        hps = {'mlp': {'default': {'num_h_layers': 4, 'num_neurons': [16, 32, 32, 16]}},
            'rbf_svc': {'default': {'C': 1.0, 'gamma': 'scale', 'random_state': random_state},
                   **{f'C_{i}': {'C': i, 'gamma': 'scale', 'random_state': random_state}
                      for i in [0.1, 0.5, 1.0, 5.0, 10.0]}},
               'rks': {'default': {'R': 50, 'gamma': 3, 'C': 1.0, 'random_state': random_state},
                   **{f'R_{i}_gamma_{j}_C_{k}': {'R': i, 'gamma': j, 'C': k, 'random_state': random_state}
                      for i in range(10, 110, 10)
                      for j in range(1, 11)
                      for k in [0.1, 0.5, 1.0, 5.0, 10.0]}}
        }
        # Access model hyperparams
        try:
            model_hps = hps[model]
        except Exception:
            raise Exception(f'Model {model} not found in hyperparams dictionary.')
        try:
            model_hps = model_hps[architecture]
        except Exception:
            raise Exception(f'Architecture {architecture} of model {model} not found in hyperparams dictionary.')

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


def get_training_hps(model_type, dataset_name, model):
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

    if model == 'q_kernel_method_reservoir':
        pre_train = False
    else:
        pre_train = True

    if model_type == 'torch':
        return {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'criterion': 'CrossEntropyLoss', 'output_size': 2, 'optimizer': 'Adam', 'scheduler': 'None', 'epochs': epochs, 'lr': 1e-3, 'betas': (0.9, 0.999), 'momentum': 0.9, 'weight_decay': 0.0}
    elif model_type == 'reuploading':
        return {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'track_history': True, 'max_epochs': epochs, 'lr': 1e-3, 'batch_size': 32, 'patience': 50, 'tau': 1.0, 'convergence_tolerance': 1e-6, 'output_size': 2}
    elif model_type == 'sklearn_q_kernel':
        return {'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'optimizer': 'Adam', 'lr': 1e-3, 'epochs': epochs, 'output_size': 2, 'pre_train': pre_train}
    elif model_type == 'sklearn_kernel':
        return {'output_size': 2}
    elif model_type == 'sklearn':
        return {'output_size': 2}
    else:
        raise Exception(f'Model type {model_type} has no training hyperparameters.')


def get_hyperparams(dataset, model, architecture, backend, sk_random):
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

    dataset_hps = get_dataset_hps(dataset_name, arg1, arg2, model)

    # Model HPs #####################################################
    if backend == 'photonic':
        model_hps = get_model_hps_photonic(model, architecture)
    elif backend == 'gate':
        model_hps = get_model_hps_gate(model, architecture)
    elif backend == 'classical':
        model_hps = get_model_hps_classical(model, architecture, sk_random)
    else:
        raise ValueError(f'Unknown backend: {backend}')

    # Training HPs #####################################################
    model_type = model_hps['type']
    training_hps = get_training_hps(model_type, dataset_name, model)

    try:
        device = training_hps['device']
        if device == torch.device('cuda'):
            logging.warning('Training on GPU, cuda available.')
        elif device == torch.device('cpu'):
            logging.warning('Training on CPU.')
        else:
            raise NotImplementedError(f'Unknown device: {device}')
    except KeyError:
        logging.warning('Training on CPU.')

    return {'dataset': dataset_hps, 'model': model_hps, 'training': training_hps}