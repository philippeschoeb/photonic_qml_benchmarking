from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin


class RBFSVC:
    def __init__(
        self, C=1.0, gamma="scale", probability=False, random_state=None, **kwargs
    ):
        """
        Wrapper for sklearn's SVC with RBF kernel.

        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter.
        gamma : {"scale", "auto"} or float, default="scale"
            Kernel coefficient.
        probability : bool, default=False
            Whether to enable probability estimates.
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation.
        """
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

    def fit(self, x, y):
        """Fit the RBF SVC model."""
        self.model.fit(x, y)
        return self

    def predict(self, x):
        """Predict class labels for samples in x."""
        return self.model.predict(x)

    def predict_proba(self, x):
        """Probability estimates for samples in x (if probability=True)."""
        return self.model.predict_proba(x)

    def decision_function(self, x):
        """Distance of samples x to the separating hyperplane."""
        return self.model.decision_function(x)

    def score(self, x, y):
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(x, y)


# Scikit-learn version of the RBFSVC
class SKRBFSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = RBFSVC
        self.model_type = "sklearn"
        self.model_name = "rbf_svc"
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
        self.model.fit(x, y)
        return self

    def predict(self, x):
        preds = self.model.predict(x)
        return preds

    def predict_proba(self, x):
        probs = self.model.predict_proba(x)
        return probs

    def score(self, x, y):
        preds = self.model.predict(x)
        return accuracy_score(preds, y)
