

def architecture_help(model, backend):
    # For photonic backend
    if backend == 'photonic':
        if model == 'dressed_quantum_circuit' or model == 'dressed_quantum_circuit_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture m_{i}_n_{j}\n\nWhere i is the number of modes,\nand j is the number of photons.\n'
            return message
        elif model == 'multiple_paths_model' or model == 'multiple_paths_model_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture m_{i}_n_{j}_numNeurons_{k}_{l}_{m}_...\n\nWhere i is the number of modes,\nj is the number of photons,\n'
                        'k is the number of neurons in the first classical hidden layer,\n'
                        'l is the number of neurons in the second classical hidden layer,\n'
                        'm is the number of neurons in the third classical hidden layer,\n'
                        'and so on...\n')
            return message
        elif model == 'data_reuploading':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture numLayers_{i}\n\nWhere i is the number of 2 modes layers to encode data and parameters.\n'
            return message
        elif model == 'data_reuploading_reservoir':
            return ('The data_reuploading_reservoir variant is not supported in this benchmarking study.\n'
                    'Please choose another model.')
        elif model == 'q_kernel_method' or model == 'q_kernel_method_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture m_{i}_n_{j}\n\nWhere i is the number of modes,\nand j is the number of photons.\n'
            return message
        elif model == 'q_rks':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture m_{i}_n_{j}_R_{k}_gamma_{l}\n\nWhere i is the number of modes,\n'
                        'j is the number of photons.\nk is the number of random Fourier components,\n'
                        'and l is the inverse of the standard deviation of the approximated Gaussian.\n')
            return message
        else:
            message = f'There is no current help for the architecture string formatting of the {model} with backend {backend}.\n'
            return message

    # For gate based backend
    elif backend == 'gate':
        if model == 'dressed_quantum_circuit' or model == 'dressed_quantum_circuit_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture numLayers_{i}\n\nWhere i is the number of layers in the variational part of the circuit.\n'
            return message
        elif model == 'multiple_paths_model' or model == 'multiple_paths_model_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture numLayers_{i}_numNeurons_{j}_{k}_{l}_...\n\n'
                        'Where i is the number of layers in the variational part of the circuit,\n'
                        'j is the number of neurons in the first classical hidden layer,\n'
                        'k is the number of neurons in the second classical hidden layer,\n'
                        'l is the number of neurons in the third classical hidden layer,\n'
                        'and so on...\n')
            return message
        elif model == 'data_reuploading':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture numLayers_{i}\n\nWhere i is the number of blocks used in the trainable embedding.\n'
            return message
        elif model == 'data_reuploading_reservoir':
            return ('The data_reuploading_reservoir variant is not supported in this benchmarking study.\n'
                    'Please choose another model.')
        elif model == 'q_kernel_method' or model == 'q_kernel_method_reservoir':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture repeats_{i}\n\nWhere i is the number of times the IQP structure is repeated in the embedding circuit.\n'
            return message
        elif model == 'q_rks':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture R_{i}_gamma_{j}\n\nWhere i is the number of random Fourier components,\n'
                        'and j is the inverse of the standard deviation of the approximated Gaussian.\n')
            return message
        else:
            message = f'There is no current help for the architecture string formatting of the {model} with backend {backend}.\n'
            return message

    # For classical models
    else:
        if model == 'mlp':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture numNeurons_{i}_{j}_{k}_...\n\nWhere i is the number of neurons in the first classical hidden layer,\n'
                        'j is the number of neurons in the second classical hidden layer,\n'
                        'k is the number of neurons in the third classical hidden layer,\n'
                        'and so on...\n')
            return message
        elif model == 'rbf_svc':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += '--architecture C_{f}\n\nWhere f (float) is the regularization parameter.\n'
            return message
        elif model == 'rks':
            message = f'Here is the architecture string formatting of the {model} with backend {backend}:\n\n'
            message += ('--architecture R_{i}_gamma_{j}\n\nWhere i is the number of random Fourier components,\n'
                        'and j is the inverse of the standard deviation of the approximated Gaussian.\n')
            return message
        else:
            message = f'There is no current help for the architecture string formatting of the {model} with backend {backend}.\n'
            return message
