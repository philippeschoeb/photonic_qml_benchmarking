import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from models.photonic_models.scaling_layer import scale_from_string_to_value
from merlin_additional.reuploading_merlin.reuploading_experiment import (
    MerlinReuploadingClassifier,
)


class DataReuploading(MerlinReuploadingClassifier):
    """
    Always assumes m=2, n=1.
    """

    def __init__(self, input_size, numLayers, design="AA", scaling="1/pi", **kwargs):
        assert input_size <= numLayers * 2, (
            f"Not enough layers ({numLayers}) to encode all the input data of size {input_size}."
        )
        scaling_value = scale_from_string_to_value(scaling)
        super().__init__(input_size, numLayers, design, scaling_value)


# Scikit-learn version of the DataReuploading
class SKDataReuploading(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = DataReuploading
        self.model_type = "sklearn"
        self.model_name = "data_reuploading"
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None
        self.train_losses = None
        self.train_accuracies = None

    # Override get_params to make nested dicts compatible with sklearn
    def get_params(self, deep=True):
        params = dict(self.data_params)
        params.update({f"model_params__{k}": v for k, v in self.model_params.items()})
        params.update(
            {f"training_params__{k}": v for k, v in self.training_params.items()}
        )
        return params

    # Override set_params to handle nested dict keys
    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith("data_params__"):
                subkey = key.split("__", 1)[1]
                self.data_params[subkey] = value
            elif key.startswith("model_params__"):
                subkey = key.split("__", 1)[1]
                self.model_params[subkey] = value
            elif key.startswith("training_params__"):
                subkey = key.split("__", 1)[1]
                self.training_params[subkey] = value
            else:
                setattr(self, key, value)
        return self

    def fit(self, x, y):
        self.model = self.model_class(**self.model_params)
        total = len(x)

        # Get hyperparams
        track_history = self.training_params.get("track_history", True)
        max_epochs = self.training_params.get("epochs", 50)
        learning_rate = self.training_params.get("lr", 0.001)
        batch_size = self.data_params.get("batch_size", 32)
        patience = self.training_params.get("patience", 50)
        tau = self.training_params.get("tau", 1.0)
        convergence_tolerance = self.training_params.get("convergence_tolerance", 1e-06)

        self.model.fit(
            x,
            y,
            track_history=track_history,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            tau=tau,
            convergence_tolerance=convergence_tolerance,
        )
        train_losses = self.model.training_history_["loss"]

        y_pred_train = self.model.predict(x)
        train_acc = accuracy_score(y, y_pred_train)

        # Count number of parameters
        num_params = sum(
            p.numel() for p in self.model.quantum_model.parameters() if p.requires_grad
        )
        logging.info(f"Number of parameters: {num_params}")
        # logging.warning(f'Final Train Accuracy: {train_acc:.4f} out of total train size: {total}')  # Does not work for some reason
        print(f"Final Train Accuracy: {train_acc:.4f} out of total train size: {total}")
        self.train_losses = train_losses
        self.final_train_acc = train_acc
        return self

    def predict(self, x):
        return self.model.predict(x)

    def predict_proba(self, x):
        return self.model.predict_proba(x)

    def score(self, x, y):
        return self.model.score(x, y)
