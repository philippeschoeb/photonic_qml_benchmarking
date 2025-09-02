from datasets.data import get_data, preprocess_data, preprocess_labels, convert_array_to_tensor, convert_tensor_to_loader, subsample
import logging

def fetch_data(dataset, random_state, **hyperparams):
    # List of allowed dataset names
    dataset_names = ['downscaled_mnist_pca', 'hidden_manifold', 'two_curves']

    # Find which dataset_name matches the start of the string
    dataset_name = next(name for name in dataset_names if dataset.startswith(name))

    # Remove dataset_name + underscore and split the rest
    args_part = dataset[len(dataset_name) + 1:]  # skip underscore
    args = args_part.split("_") if args_part else []

    arg1 = args[0] if len(args) >= 1 else None
    arg2 = args[1] if len(args) >= 2 else None

    x_train, x_test, y_train, y_test = get_data(dataset_name, arg1=int(arg1), arg2=int(arg2) if arg2 else None)

    # Subsampling dataset if kernel method on downscaled_mnist_pca
    num_train = hyperparams.get('num_train', None)
    num_test = hyperparams.get('num_test', None)
    if num_train is not None and num_test is not None:
        x_train, x_test, y_train, y_test = subsample(x_train, x_test, y_train, y_test, num_train, num_test, random_state)
        logging.info(f'Subsample {num_train} training and {num_test} testing data')

    scaling = hyperparams.get("scaling", "none")
    batch_size = hyperparams.get("batch_size", 32)
    labels_treatment = hyperparams.get("labels_treatment", "none")
    x_train, x_test = preprocess_data(x_train, x_test, scaling)
    y_train, y_test = preprocess_labels(y_train, y_test, labels_treatment)
    x_train, x_test, y_train, y_test = convert_array_to_tensor(x_train, x_test, y_train, y_test, labels_treatment=labels_treatment)
    train_loader = convert_tensor_to_loader(x_train, y_train, batch_size=batch_size)
    test_loader = convert_tensor_to_loader(x_test, y_test, batch_size=batch_size)

    return train_loader, test_loader, x_train, x_test, y_train, y_test


def fetch_sk_data(dataset, random_state, **hyperparams):
    # List of allowed dataset names
    dataset_names = ['downscaled_mnist_pca', 'hidden_manifold', 'two_curves']

    # Find which dataset_name matches the start of the string
    dataset_name = next(name for name in dataset_names if dataset.startswith(name))

    # Remove dataset_name + underscore and split the rest
    args_part = dataset[len(dataset_name) + 1:]  # skip underscore
    args = args_part.split("_") if args_part else []

    arg1 = args[0] if len(args) >= 1 else None
    arg2 = args[1] if len(args) >= 2 else None

    x_train, x_test, y_train, y_test = get_data(dataset_name, arg1=int(arg1), arg2=int(arg2) if arg2 else None)

    # Subsampling dataset if kernel method on downscaled_mnist_pca
    num_train = hyperparams.get('num_train', 'None')
    if num_train != 'None':
        num_train = num_train[0]
    num_test = hyperparams.get('num_test', 'None')
    if num_test != 'None':
        num_test = num_test[0]
    if num_train != 'None' and num_test != 'None':
        x_train, x_test, y_train, y_test = subsample(x_train, x_test, y_train, y_test, num_train, num_test)
        logging.info(f'Subsample {num_train} training and {num_test} testing data')

    scaling = hyperparams.get("scaling", "none")[0]
    labels_treatment = hyperparams.get("labels_treatment", "none")[0]
    x_train, x_test = preprocess_data(x_train, x_test, scaling)
    y_train, y_test = preprocess_labels(y_train, y_test, labels_treatment)
    return x_train, x_test, y_train, y_test