import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

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


def get_z_s_classically(x_r_i_s):
    """
    Compute classical random kitchen sinks features.

    Args:
        x_r_i_s (np.array): Transformed input features

    Returns:
        np.array: Classical random features z(x) = sqrt(2/r) * cos(gamma * (w * x + b))
    """
    n, r = x_r_i_s.shape
    z_s = np.sqrt(2) * np.cos(x_r_i_s)
    z_s = z_s / np.sqrt(r)
    return z_s


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


def classical_rand_kitchen_sinks(x_train, x_test, w, b, r, gamma):
    # Transform data
    x_r_i_s_train = get_x_r_i_s(x_train, w, b, r, gamma)
    x_r_i_s_test = get_x_r_i_s(x_test, w, b, r, gamma)

    z_s_train = get_z_s_classically(x_r_i_s_train)
    z_s_test = get_z_s_classically(x_r_i_s_test)

    kernel_matrix_training = get_approx_kernel_train(z_s_train)
    kernel_matrix_test = get_approx_kernel_predict(z_s_test, z_s_train)

    return kernel_matrix_training, kernel_matrix_test


class RKS:
    """
    Classical random kitchen sinks model.
    """
    def __init__(self, R, gamma, input_size, C=1.0, probability=False, random_state=None, **kwargs):
        self.input_size = input_size
        self.model = SVC(
            kernel="precomputed",
            C=C,
            probability=probability,
            random_state=random_state,
        )
        self.R = R
        self.gamma = gamma

    def get_kernels(self, x_train, x_test):
        w, b = get_random_w_b(self.R, self.input_size)
        x_r_i_s_train = get_x_r_i_s(x_train, w, b, self.R, self.gamma)
        x_r_i_s_test = get_x_r_i_s(x_test, w, b, self.R, self.gamma)
        z_s_train = get_z_s_classically(x_r_i_s_train)
        z_s_test = get_z_s_classically(x_r_i_s_test)
        kernel_matrix_training = get_approx_kernel_train(z_s_train)
        kernel_matrix_test = get_approx_kernel_predict(z_s_test, z_s_train)
        return kernel_matrix_training, kernel_matrix_test

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


class HalvingGridRKS(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = RKS
        self.model_type = 'sklearn'
        self.model_name = 'rks'
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None
        self.train_losses = None
        self.train_accuracies = None

        self.x_train = None

    # Override get_params to make nested dicts compatible with sklearn
    def get_params(self, deep=True):
        params = dict(self.data_params)
        params.update({f"model_params__{k}": v for k, v in self.model_params.items()})
        params.update({f"training_params__{k}": v for k, v in self.training_params.items()})
        return params

    # Override set_params to handle nested dict keys
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
        self.x_train = x
        self.model = self.model_class(**self.model_params)
        k_train, _ = self.model.get_kernels(x, x)
        self.model.fit(k_train, y)
        return self

    def predict(self, x):
        k_train, k_test = self.model.get_kernels(self.x_train, x)
        preds = self.model.predict(k_test)
        return preds

    def predict_proba(self, x):
        k_train, k_test = self.model.get_kernels(self.x_train, x)
        probs = self.model.predict_proba(k_test)
        return probs

    def score(self, x, y):
        k_train, k_test = self.model.get_kernels(self.x_train, x)
        preds = self.model.predict(k_test)
        return accuracy_score(preds, y)