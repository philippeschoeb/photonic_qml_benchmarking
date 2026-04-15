from sklearn.metrics import accuracy_score
import logging
from tqdm import tqdm


def training_sklearn_gate(
    model_dict, x_train, x_test, y_train, y_test, max_train_time_seconds=None
):
    model_type = model_dict["type"]
    model_name = model_dict["name"]
    model = model_dict["model"]
    if max_train_time_seconds is not None:
        model.max_train_time_seconds = float(max_train_time_seconds)

    # Convert tensors to numpy arrays
    x_train = x_train.detach().cpu().numpy()
    x_test = x_test.detach().cpu().numpy()
    y_train = y_train.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()

    with tqdm(total=1, desc="Fitting sklearn gate model", unit="fit") as pbar:
        model.fit(x_train, y_train)
        pbar.update(1)
    y_pred_test = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    y_pred_train = model.predict(x_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    logging.warning(
        f"Final Train Accuracy: {train_accuracy:.4f} | Final Test Accuracy: {test_accuracy:.4f}"
    )
    return {
        "type": model_type,
        "name": model_name,
        "model": model,
        "final_train_acc": train_accuracy,
        "final_test_acc": test_accuracy,
        "timed_out": bool(getattr(model, "timed_out_", False)),
        "timeout_stage": "gate_jax_training"
        if bool(getattr(model, "timed_out_", False))
        else None,
        "max_train_time_seconds": max_train_time_seconds,
    }
