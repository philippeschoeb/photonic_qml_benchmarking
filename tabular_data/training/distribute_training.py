from training.training_torch import (
    training_torch,
    training_reuploading,
    training_sklearn_q_kernel,
)
from training.training_scikit_learn import training_sklearn, training_sklearn_kernel
from training.gate_based_training.training_sklearn_gate import training_sklearn_gate


def distribute_training(
    model_dict,
    train_loader,
    test_loader,
    x_train,
    x_test,
    y_train,
    y_test,
    **hyperparams,
):
    if model_dict["type"] == "torch":
        return training_torch(
            model_dict,
            train_loader,
            test_loader,
            criterion=hyperparams["criterion"],
            optimizer=hyperparams["optimizer"],
            scheduler=hyperparams["scheduler"],
            epochs=hyperparams["epochs"],
            lr=hyperparams["lr"],
            betas=hyperparams["betas"],
            momentum=hyperparams["momentum"],
            weight_decay=hyperparams["weight_decay"],
            device=hyperparams["device"],
        )
    elif model_dict["type"] == "reuploading":
        return training_reuploading(
            model_dict,
            x_train,
            x_test,
            y_train,
            y_test,
            track_history=hyperparams["track_history"],
            max_epochs=hyperparams["epochs"],
            learning_rate=hyperparams["lr"],
            batch_size=hyperparams["batch_size"],
            patience=hyperparams["patience"],
            tau=hyperparams["tau"],
            convergence_tolerance=hyperparams["convergence_tolerance"],
        )
    elif model_dict["type"] == "sklearn_q_kernel":
        return training_sklearn_q_kernel(
            model_dict,
            train_loader,
            x_train,
            x_test,
            y_train,
            y_test,
            optimizer=hyperparams["optimizer"],
            lr=hyperparams["lr"],
            epochs=hyperparams["epochs"],
            pre_train=hyperparams["pre_train"],
            device=hyperparams["device"],
        )
    elif model_dict["type"] == "sklearn_kernel":
        return training_sklearn_kernel(model_dict, x_train, x_test, y_train, y_test)
    elif model_dict["type"] == "sklearn":
        return training_sklearn(model_dict, x_train, x_test, y_train, y_test)
    elif model_dict["type"] == "jax_sklearn_gate":
        return training_sklearn_gate(model_dict, x_train, x_test, y_train, y_test)
    elif model_dict["type"] == "sklearn_gate" or model_dict["type"] == "gate_rks":
        return training_sklearn_gate(model_dict, x_train, x_test, y_train, y_test)
    else:
        raise ValueError(f"Unknown model type: {model_dict['type']}")
