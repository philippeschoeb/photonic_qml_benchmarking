# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pennylane as qml
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import optax
import jax
import jax.numpy as jnp
from models.gate_based_utils import train, chunk_vmapped_fn


class MultiplePathsModelClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_layers=3,
        n_classical_h_layers=0,
        num_neurons=[],
        learning_rate=0.001,
        batch_size=32,
        max_vmap=None,
        jit=True,
        max_steps=100000,
        convergence_interval=200,
        dev_type="default.qubit",
        qnode_kwargs={"interface": "jax-jit"},
        scaling=1.0,
        random_state=42,
        reservoir=False,
        **kwargs,
    ):
        r"""
        Gate-based multiple paths model.

        The model concatenates raw input features with expectation values from a
        variational quantum circuit, then passes the combined vector through a
        classical output network to produce class logits. Probabilities are
        obtained via softmax, and the model is trained with binary cross entropy.

        Args:
            n_layers (int): Number of layers in the variational part of the circuit.
            n_classical_h_layers (int): Number of hidden layers in the classical output network.
            num_neurons (list[int]): Hidden layer sizes for the classical output network.
            learning_rate (float): Initial learning rate for gradient descent.
            max_steps (int): Maximum number of training steps. A warning will be raised if training did not converge.
            max_vmap (int or None): The maximum size of a chunk to vectorise over. Lower values use less memory.
                must divide batch_size.
            batch_size (int): Size of batches used for computing parameter updates.
            convergence_interval (int): The number of loss values to consider to decide convergence.
            jit (bool): Whether to use just in time compilation.
            dev_type (str): Pennylane device type; e.g. 'default.qubit'.
            qnode_kwargs (dict): Keyword arguments passed to the circuit qnode.
            scaling (float): Factor by which to scale the input data.
            random_state (int): Seed used for pseudorandom number generation.
            reservoir (bool): If True, keep the quantum circuit weights fixed after random initialization.
        """
        # attributes that do not depend on data
        self.n_layers = n_layers
        self.n_classical_h_layers = n_classical_h_layers
        self.num_neurons = num_neurons
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.dev_type = dev_type
        self.qnode_kwargs = qnode_kwargs
        self.jit = jit
        self.scaling = scaling
        self.random_state = random_state
        self.reservoir = reservoir
        self.rng = np.random.default_rng(random_state)

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        # data-dependant attributes
        # which will be initialised by calling "fit"
        self.params_ = None  # Dictionary containing the trainable parameters
        self.non_train_params_ = None
        self.n_qubits_ = None
        self.n_features_ = None
        self.scaler = None  # data scaler will be fitted on training data

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def output_transform(self, params, x):
        """
        The final neural network
        """
        if self.n_classical_h_layers == 0:
            x = jnp.matmul(params["output_weights"], x)
        else:
            for param_layer in params["output_weights"]:
                x = jnp.matmul(param_layer, x)
        return x

    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            """
            The variational circuit taken from the plots
            """
            weight_source = self.non_train_params_ if self.reservoir else params
            # data encoding
            for i in range(self.n_qubits_):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)
            # trainable unitary
            for layer in range(self.n_layers):
                for i in range(self.n_qubits_):
                    qml.RY(weight_source["circuit_weights"][layer, i], wires=i)
                # qml.broadcast(qml.CNOT, wires=range(self.n_qubits_), pattern="ring")
                for i in range(self.n_qubits_):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits_])

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def post_circuit(x: jnp.ndarray) -> jnp.ndarray:
            # mean and std over batch dimension (axis=0)
            mean = jnp.mean(x, axis=0, keepdims=True)
            std = jnp.std(x, axis=0, keepdims=True)

            return (x - mean) / (std + 1e-8)

        def dressed_circuit(params, x):
            x_1 = x
            x_2 = jnp.array(circuit(params, x)).T
            x_2 = post_circuit(x_2)
            x_3 = jnp.concatenate([x_1, x_2])
            x_out = self.output_transform(params, x_3)
            return x_out

        if self.jit:
            dressed_circuit = jax.jit(dressed_circuit)
        self.forward = jax.vmap(dressed_circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize(self, n_features, classes=None):
        """Initialize attributes that depend on the number of features and the class labels.
        Args:
            n_features (int): Number of features that the classifier expects
            classes (array-like): class labels that the classifier expects
        """
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_

        self.n_features_ = n_features
        self.n_qubits_ = self.n_features_

        self.initialize_params()
        self.construct_model()

    def initialize_params(self):
        # initialise the trainable parameters
        circuit_weights = (
            2
            * jnp.pi
            * jax.random.uniform(
                shape=(self.n_layers, self.n_qubits_), key=self.generate_key()
            )
        )
        """input_weights = (
            jax.random.normal(
                shape=(self.n_qubits_, self.n_qubits_), key=self.generate_key()
            )
            / self.n_features_
        )"""
        if self.n_classical_h_layers == 0:
            output_weights = (
                jax.random.normal(
                    shape=(2, self.n_qubits_ + self.n_features_),
                    key=self.generate_key(),
                )
                / self.n_features_
            )
            # output_weights = tuple(output_weights)
        else:
            output_weights = []
            previous_size = self.n_qubits_ + self.n_features_
            for i, num in enumerate(self.num_neurons):
                output_weights.append(
                    jax.random.normal(
                        shape=(num, previous_size), key=self.generate_key()
                    )
                )
                previous_size = num
            output_weights.append(
                jax.random.normal(shape=(2, previous_size), key=self.generate_key())
            )
            output_weights = tuple(output_weights)

        if self.reservoir:
            self.params_ = {
                # "input_weights": input_weights,
                "output_weights": output_weights,
            }
            self.non_train_params_ = {
                "circuit_weights": circuit_weights,
            }
        else:
            self.params_ = {
                "circuit_weights": circuit_weights,
                # "input_weights": input_weights,
                "output_weights": output_weights,
            }
            self.non_train_params_ = None

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
            y (np.ndarray): Labels of shape (n_samples,)
        """

        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        self.initialize(X.shape[1], classes=np.unique(y))

        optimizer = optax.adam

        def loss_fn(params, X, y):
            vals = self.forward(params, X)
            # convert to 0,1 one hot encoded labels
            labels = jax.nn.one_hot(jax.nn.relu(y), 2)
            return jnp.mean(optax.softmax_cross_entropy(vals, labels))

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )

        return self

    def predict(self, X):
        """Predict labels for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred (np.ndarray): Predicted labels of shape (n_samples,)
        """
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)

        Returns:
            y_pred_proba (np.ndarray): Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        X = self.transform(X)
        return jax.nn.softmax(self.chunked_forward(self.params_, X))

    def transform(self, X, preprocess=True):
        """
        Args:
            X (np.ndarray): Data of shape (n_samples, n_features)
        """
        if preprocess:
            if self.scaler is None:
                # if the model is unfitted, initialise the scaler here
                self.scaler = StandardScaler()
                self.scaler.fit(X)

            X = self.scaler.transform(X)
        return X * self.scaling


"""class DressedQuantumCircuitClassifierOnlyNN(DressedQuantumCircuitClassifier):

    def construct_model(self):
        def dressed_circuit(params, x):
            x = self.input_transform(params, x)
            x = self.mid_transform(params, x)
            x = self.output_transform(params, x)
            return x

        if self.jit:
            dressed_circuit = jax.jit(dressed_circuit)
        self.forward = jax.vmap(dressed_circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward

    def initialize_params(self):
        # initialise the trainable parameters
        mid_weights = (
            jax.random.normal(
                shape=(self.n_qubits_, self.n_qubits_), key=self.generate_key()
            )
            / self.n_features_
        )
        '''input_weights = (
            jax.random.normal(
                shape=(self.n_qubits_, self.n_qubits_), key=self.generate_key()
            )
            / self.n_features_
        )'''
        output_weights = (
            jax.random.normal(shape=(2, self.n_qubits_), key=self.generate_key())
            / self.n_features_
        )
        self.params_ = {
            "mid_weights": mid_weights,
            #"input_weights": input_weights,
            "output_weights": output_weights,
        }

    def mid_transform(self, params, x):
        x = jnp.matmul(params["mid_weights"], x)
        x = jnp.tanh(x) * jnp.pi / 2
        return x


class DressedQuantumCircuitClassifierSeparable(DressedQuantumCircuitClassifier):
    def construct_model(self):
        dev = qml.device(self.dev_type, wires=self.n_qubits_)

        @qml.qnode(dev, **self.qnode_kwargs)
        def circuit(params, x):
            '''
            The variational circuit taken from the plots
            '''
            # data encoding
            for i in range(self.n_qubits_):
                qml.Hadamard(wires=i)
                qml.RY(x[i], wires=i)
            # trainable unitary
            for layer in range(self.n_layers):
                for i in range(self.n_qubits_):
                    qml.RY(params["circuit_weights"][layer, i], wires=i)

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits_)]

        self.circuit = circuit

        def dressed_circuit(params, x):
            x = self.input_transform(params, x)
            x = jnp.array(circuit(params, x)).T
            x = self.output_transform(params, x)
            return x

        if self.jit:
            dressed_circuit = jax.jit(dressed_circuit)
        self.forward = jax.vmap(dressed_circuit, in_axes=(None, 0))
        self.chunked_forward = chunk_vmapped_fn(self.forward, 1, self.max_vmap)

        return self.forward"""


class SKMultiplePathsModelGate(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for the gate-based multiple paths model."""

    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = MultiplePathsModelClassifier
        self.model_type = "sklearn_gate"
        self.model_name = "multiple_paths_model"
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None
        self.train_losses = None
        self.train_accuracies = None
        self.final_train_acc = None

    def get_params(self, deep=True):
        params = dict(self.data_params)
        params.update({f"model_params__{k}": v for k, v in self.model_params.items()})
        params.update(
            {f"training_params__{k}": v for k, v in self.training_params.items()}
        )
        return params

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

    def _prepare_model_kwargs(self):
        kwargs = dict(self.model_params)
        kwargs.pop("type", None)
        kwargs.pop("name", None)
        kwargs.pop("input_size", None)
        kwargs.pop("output_size", None)
        if "numLayers" in kwargs and "n_layers" not in kwargs:
            kwargs["n_layers"] = kwargs.pop("numLayers")
        if "lr" in kwargs and "learning_rate" not in kwargs:
            kwargs["learning_rate"] = kwargs.pop("lr")
        if "numNeurons" in kwargs and "num_neurons" not in kwargs:
            kwargs["num_neurons"] = kwargs.pop("numNeurons")
        num_neurons = kwargs.get("num_neurons", [])
        if "n_classical_h_layers" not in kwargs:
            kwargs["n_classical_h_layers"] = len(num_neurons) if num_neurons else 0
        if kwargs.get("max_vmap") is None:
            kwargs["max_vmap"] = kwargs.get("batch_size", 32)
        return kwargs

    def fit(self, X, y):
        model_kwargs = self._prepare_model_kwargs()
        self.model = self.model_class(**model_kwargs)

        X_np = np.asarray(X)
        y_np = np.asarray(y)

        self.model.fit(X_np, y_np)
        self.train_losses = getattr(self.model, "loss_history_", None)
        train_predictions = self.model.predict(X_np)
        self.final_train_acc = accuracy_score(y_np, train_predictions)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict(np.asarray(X))

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")
        return self.model.predict_proba(np.asarray(X))

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(np.asarray(y), preds)
