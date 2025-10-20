import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import merlin_additional as mla
from models.photonic_models.input_fock_state import get_input_fock_state
from models.photonic_models.circuits import get_circuit
from models.photonic_models.scaling_layer import scale_from_string_to_value
from merlin_additional.loss import NKernelAlignment
from training.training_torch import assign_optimizer

class QSVC:
    def __init__(self, input_size, m, n, circuit, no_bunching, pre_train=True, C=1.0, probability=False, scaling="1", random_state=None, **kwargs):
        """
        Wrapper for sklearn's SVC with quantum kernel.

        Parameters
        ----------
        pre_train : bool, default True
        C : float, default=1.0
            Regularization parameter.
        probability : bool, default=False
            Whether to enable probability estimates.
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation.
        """
        self.scaling = scale_from_string_to_value(scaling)
        circuit = get_circuit(circuit, m, input_size, reservoir=(not pre_train))
        input_fock_state = get_input_fock_state('standard', m, n)
        trainable_parameters = ['theta'] if pre_train else []
        self.feature_map = mla.FeatureMap(
            circuit=circuit,
            input_size=input_size,
            input_parameters='px',
            trainable_parameters=trainable_parameters
        )
        self.quantum_kernel = mla.FidelityKernel(
            feature_map=self.feature_map,
            input_state=input_fock_state,
            no_bunching=no_bunching,
        )
        self.model = SVC(
            kernel="precomputed",
            C=C,
            probability=probability,
            random_state=random_state,
        )
        self._pretrained = False if pre_train else None

    def pretraining_done(self):
        self._pretrained = True

    def get_q_kernels(self, x_train, x_test):
        if self._pretrained or self._pretrained is None:
            # Scale data
            x_train = self.scale(x_train)
            x_test = self.scale(x_test)

            # Get kernel matrices
            k_train = self.quantum_kernel(x_train)
            k_test = self.quantum_kernel(x_test, x_train)
            return k_train, k_test
        else:
            raise Exception('quantum_kernel has to be pretrained before computing kernels')

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

    def scale(self, x):
        return self.scaling * x


class _BaseSKQSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, data_params=None, model_params=None, training_params=None):
        self.model_class = QSVC
        self.model_type = 'sklearn_q_kernel'
        self.model_name = model_name
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None
        self.train_losses = None
        self.train_accuracies = None
        self.final_train_acc = None
        self.device = torch.device('cpu')
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

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def fit(self, x, y):
        model_kwargs = dict(self.model_params)
        if "circuit_type" in model_kwargs and "circuit" not in model_kwargs:
            model_kwargs["circuit"] = model_kwargs.pop("circuit_type")
        self.model = self.model_class(**model_kwargs)

        batch_size = self.data_params.get('batch_size', 32)
        optimizer_name = self.training_params.get('optimizer', 'Adam')
        lr = self.training_params.get('lr', 0.001)
        epochs = self.training_params.get('epochs', 5)
        pre_train = self.training_params.get('pre_train', True)

        device_cfg = self.training_params.get('device', torch.device('cpu'))
        if isinstance(device_cfg, list):
            device_cfg = device_cfg[0]
        if isinstance(device_cfg, str):
            device_cfg = torch.device(device_cfg)
        self.device = device_cfg if isinstance(device_cfg, torch.device) else torch.device(device_cfg)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        if np.issubdtype(np.asarray(y).dtype, np.floating) or np.array_equal(np.unique(y), np.array([-1, 1])):
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        if pre_train:
            optimizable_model = self.model.quantum_kernel
            optimizable_model.to(self.device)
            criterion = NKernelAlignment()
            optimizer = assign_optimizer(optimizer_name, optimizable_model, lr)

            for _ in range(epochs):
                epoch_loss = 0.0
                total_samples = 0
                for batch_x, batch_y in loader:
                    batch_x = self._to_device(batch_x)
                    batch_y = self._to_device(batch_y)

                    optimizer.zero_grad()
                    scaled_batch = self.model.scale(batch_x)
                    outputs = self.model.quantum_kernel(scaled_batch)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    batch_size_actual = batch_x.size(0)
                    epoch_loss += loss.item() * batch_size_actual
                    total_samples += batch_size_actual

                train_losses.append(epoch_loss / max(total_samples, 1))

            self.model.pretraining_done()

        self._x_train = self._to_device(x_tensor)
        with torch.no_grad():
            kernel_matrix_train, _ = self.model.get_q_kernels(self._x_train, self._x_train)

        kernel_matrix_train = kernel_matrix_train.detach().cpu().numpy()
        self.model.fit(kernel_matrix_train, y)

        train_predictions = self.model.predict(kernel_matrix_train)
        train_acc = accuracy_score(y, train_predictions)
        history_length = len(train_losses) if train_losses else 1
        train_acc_history = [train_acc] * history_length

        if not train_losses:
            train_losses = [0.0]

        self.train_losses = train_losses
        self.train_accuracies = train_acc_history
        self.final_train_acc = train_acc
        self.is_fitted_ = True
        return self

    def _kernel_with_training(self, x):
        if self._x_train is None:
            raise ValueError("Estimator must be fitted before calling predict or score.")

        x_tensor = torch.tensor(x, dtype=torch.float32)
        x_tensor = self._to_device(x_tensor)
        with torch.no_grad():
            _, kernel_matrix = self.model.get_q_kernels(self._x_train, x_tensor)
        return kernel_matrix.detach().cpu().numpy()

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


class SKQKernelMethod(_BaseSKQSVC):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        super().__init__("q_kernel_method", data_params, model_params, training_params)


class SKQKernelMethodReservoir(_BaseSKQSVC):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        super().__init__("q_kernel_method_reservoir", data_params, model_params, training_params)
