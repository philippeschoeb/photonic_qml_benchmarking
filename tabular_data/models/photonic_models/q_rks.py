import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from models.photonic_models.input_fock_state import get_input_fock_state
from models.photonic_models.circuits import get_circuit
from models.photonic_models.scaling_layer import scale_from_string_to_value
import merlin as ml
import torch

def get_random_w_b(r, input_size):
    """
    Generate random weights and biases for random Fourier features.

    Args:
        r (int): Number of random features

    Returns:
        tuple: (w, b) where w is weight matrix and b is bias vector
    """
    w = np.random.normal(size=(r, input_size))
    b = np.random.uniform(low=0.0, high=2.0 * np.pi, size=(r,))
    return w, b


def get_x_r_i_s(x_s, w, b, r, gamma):
    """
    Given input data points x_s, of size [num_points, num_features],
    Return the x_{r, i}_s of size [num_points, r] such that
    x_{r, i} = gamma * (w_r * x_i + b_r)
    """
    num_points, num_features = x_s.shape

    x_r_i_s = gamma * (np.matmul(x_s, w.T) + np.tile(b, (num_points, 1)))
    assert x_r_i_s.shape == (num_points, r), f"Wrong shape for x_r_i_s: {x_r_i_s.shape}"

    return x_r_i_s


def get_approx_kernel_train(z_s):
    """
    Compute approximate kernel matrix for training data.

    Args:
        z_s (np.array): Random features for training data

    Returns:
        np.array: Approximate kernel matrix K ≈ z(x) * z(x)^T
    """
    result_matrix = np.matmul(z_s, z_s.T)
    assert result_matrix.shape == (z_s.shape[0], z_s.shape[0]), (
        f"Wrong shape for result_matrix: {result_matrix.shape}"
    )
    return result_matrix


def get_approx_kernel_predict(z_s_test, z_s_train):
    """
    Compute approximate kernel matrix between test and training data.

    Args:
        z_s_test (np.array): Random features for test data
        z_s_train (np.array): Random features for training data

    Returns:
        np.array: Approximate kernel matrix K ≈ z(x_test) * z(x_train)^T
    """
    result_matrix = np.matmul(z_s_test, z_s_train.T)
    assert result_matrix.shape == (z_s_test.shape[0], z_s_train.shape[0]), (
        f"Wrong shape for result_matrix: {result_matrix.shape}"
    )
    return result_matrix


class QRKS:
    """
    Quantum random kitchen sinks model. The quantum circuit is not trained
    """
    def __init__(self, R, gamma, input_size, m, n, circuit, no_bunching, scaling='1/pi', C=1.0, probability=False, random_state=None, **kwargs):
        self.scaling = scale_from_string_to_value(scaling)
        self.input_size = input_size

        circuit = get_circuit(circuit, m, 1, True)
        input_fock_state = get_input_fock_state('standard', m, n)
        self.pqc = ml.QuantumLayer(
        input_size=1,
        output_size=1,
        circuit=circuit,
        trainable_parameters=[],
        input_parameters=["px"],
        input_state=input_fock_state,
        no_bunching=no_bunching,
        output_mapping_strategy=ml.OutputMappingStrategy.LINEAR,
        )

        self.model = SVC(
            kernel="precomputed",
            C=C,
            probability=probability,
            random_state=random_state,
        )
        self.R = R
        self.gamma = gamma
        self._w = None
        self._b = None
        self._train_features = None

    def get_kernels(self, x_train, x_test):
        w, b = get_random_w_b(self.R, self.input_size)
        x_r_i_s_train = get_x_r_i_s(x_train, w, b, self.R, self.gamma)
        x_r_i_s_test = get_x_r_i_s(x_test, w, b, self.R, self.gamma)

        self.pqc.eval()
        train_input = torch.tensor(x_r_i_s_train, dtype=torch.float32).view(len(x_r_i_s_train) * self.R, -1) * self.scaling
        test_input = torch.tensor(x_r_i_s_test, dtype=torch.float32).view(len(x_r_i_s_test) * self.R, -1) * self.scaling
        with torch.no_grad():
            z_s_train = self.pqc(train_input).view(len(x_r_i_s_train), self.R) * 10
            z_s_test = self.pqc(test_input).view(len(x_r_i_s_test), self.R) * 10

        train_features = z_s_train.cpu().numpy()
        test_features = z_s_test.cpu().numpy()

        self._w = w
        self._b = b
        self._train_features = train_features

        kernel_matrix_training = get_approx_kernel_train(train_features)
        kernel_matrix_test = get_approx_kernel_predict(test_features, train_features)

        return kernel_matrix_training, kernel_matrix_test

    def _ensure_random_features(self):
        if self._w is None or self._b is None or self._train_features is None:
            raise RuntimeError("Random features not initialized. Call get_kernels before computing derived kernels.")

    def compute_features(self, x):
        self._ensure_random_features()
        x_r_i_s = get_x_r_i_s(x, self._w, self._b, self.R, self.gamma)
        num_points = x_r_i_s.shape[0]
        self.pqc.eval()
        inputs = torch.tensor(x_r_i_s, dtype=torch.float32).view(num_points * self.R, -1) * self.scaling
        with torch.no_grad():
            outputs = self.pqc(inputs).view(num_points, self.R) * 10
        return outputs.cpu().numpy()


    def fit(self, k_train, y):
        """Fit the QSVC model with the precomputed kernel matrix, k_train. k_train has shape: (n_train, n_train)"""
        self.model.fit(k_train, y)
        return self

    def predict(self, k_test):
        """Predict class labels for K_test kernel matrix. k_test has shape: (n_test, n_train)"""
        return self.model.predict(k_test)

    def predict_proba(self, k_test):
        """Probability estimates for samples in k_test kernel matrix (if probability=True)."""
        return self.model.predict_proba(k_test)

    def decision_function(self, k_test):
        """Distance of samples in kernel matrix k_test to the separating hyperplane."""
        return self.model.decision_function(k_test)

    def score(self, k_test, y):
        """Return the mean accuracy on the given test data in kernel matrix k_test and labels."""
        return self.model.score(k_test, y)


class SKQRKS(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = QRKS
        self.model_type = 'sklearn_kernel'
        self.model_name = 'q_rks'
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None
        self.train_losses = None
        self.train_accuracies = None
        self.final_train_acc = None
        self._x_train = None

    def get_params(self, deep=True):
        params = dict(self.data_params)
        params.update({f"model_params__{k}": v for k, v in self.model_params.items()})
        params.update({f"training_params__{k}": v for k, v in self.training_params.items()})
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if key.startswith('data_params__'):
                subkey = key.split('__', 1)[1]
                self.data_params[subkey] = value
            elif key.startswith('model_params__'):
                subkey = key.split('__', 1)[1]
                self.model_params[subkey] = value
            elif key.startswith('training_params__'):
                subkey = key.split('__', 1)[1]
                self.training_params[subkey] = value
            else:
                setattr(self, key, value)
        return self

    def fit(self, x, y):
        model_kwargs = dict(self.model_params)
        if "circuit_type" in model_kwargs and "circuit" not in model_kwargs:
            model_kwargs["circuit"] = model_kwargs.pop("circuit_type")
        self.model = self.model_class(**model_kwargs)
        kernel_matrix_train, _ = self.model.get_kernels(x, x)
        self._x_train = np.array(x)

        self.model.fit(kernel_matrix_train, y)

        train_predictions = self.model.predict(kernel_matrix_train)
        train_acc = accuracy_score(y, train_predictions)

        self.train_losses = [0.0]
        self.train_accuracies = [train_acc]
        self.final_train_acc = train_acc
        self.is_fitted_ = True
        return self

    def _kernel_with_training(self, x):
        if self._x_train is None:
            raise ValueError("Estimator must be fitted before calling predict or score.")
        features_test = self.model.compute_features(x)
        kernel_matrix = get_approx_kernel_predict(features_test, self.model._train_features)
        return kernel_matrix

    def predict(self, x):
        kernel_matrix = self._kernel_with_training(x)
        return self.model.predict(kernel_matrix)

    def predict_proba(self, x):
        kernel_matrix = self._kernel_with_training(x)
        return self.model.predict_proba(kernel_matrix)

    def decision_function(self, x):
        kernel_matrix = self._kernel_with_training(x)
        return self.model.decision_function(kernel_matrix)

    def score(self, x, y):
        preds = self.predict(x)
        return accuracy_score(y, preds)
