import logging
import torch
from torch.utils.data import DataLoader, TensorDataset
import merlin as ml
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from training.training_torch import assign_criterion, assign_optimizer, assign_scheduler
from models.photonic_based_utils import ScalingLayer, StandardizationLayer, MinMaxScalingLayer, get_circuit, get_input_fock_state, get_computation_space
from models.classical_models.mlp import MLP


class MultiplePathsModel(torch.nn.Module):
    """
    Assuming angle encoding (for now)
    """

    def __init__(
        self,
        scaling,
        input_size,
        output_size,
        m,
        n,
        circuit_type,
        reservoir,
        no_bunching,
        post_circuit_scaling,
        numNeurons,
        input_state_type="standard",
        **kwargs,
    ):
        super().__init__()
        self.scaling = ScalingLayer(scaling)

        circuit = get_circuit(circuit_type, m, input_size, reservoir)
        input_fock_state = get_input_fock_state(input_state_type, m, n)
        trainable_params = [] if reservoir else ["theta"]
        computation_space = get_computation_space(no_bunching)
        self.pqc = ml.QuantumLayer(
            input_size=input_size,
            circuit=circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=["px"],
            measurement_strategy=ml.MeasurementStrategy.probs(computation_space=computation_space),
        )
        quantum_output_size = self.pqc.output_size
        self.post_circuit = self.set_up_post_circuit_scaling(post_circuit_scaling)

        self.mlp = MLP(input_size + quantum_output_size, output_size, numNeurons)

    def set_up_post_circuit_scaling(self, post_circuit_scaling):
        if post_circuit_scaling == "none" or post_circuit_scaling is None:
            return None
        elif post_circuit_scaling == "standardize":
            return StandardizationLayer()
        elif post_circuit_scaling == "minmax":
            return MinMaxScalingLayer()
        else:
            raise NotImplementedError(
                f"post_circuit_scaling {post_circuit_scaling} not implemented"
            )

    def forward(self, x):
        x_1 = x
        x_2 = self.scaling(x)
        x_2 = self.pqc(x_2)
        x_2 = self.post_circuit(x_2) if self.post_circuit is not None else x_2
        x_3 = torch.cat((x_1, x_2), dim=1)
        output = self.mlp(x_3)
        return output


# Scikit-learn version of the MultiplePathsModel
class SKMultiplePathsModel(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = MultiplePathsModel
        self.model_type = "sklearn"
        self.model_name = "multiple_paths_model"
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
        input_size = x.shape[1]
        self.model_params["m"] = 2 * input_size
        self.model_params["n"] = input_size
        self.model = self.model_class(**self.model_params)

        # Get hyperparams
        criterion = self.training_params.get("criterion", "CrossEntropyLoss")
        optimizer = self.training_params.get("optimizer", "Adam")
        scheduler = self.training_params.get("scheduler", "None")
        epochs = self.training_params.get("epochs", 50)
        lr = self.training_params.get("lr", 0.001)
        betas = self.training_params.get("betas", (0.9, 0.999))
        momentum = self.training_params.get("momentum", 0.9)
        weight_decay = self.training_params.get("weight_decay", 0)
        device = self.training_params.get("device", "cpu")

        # Prepare data
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, self.data_params["batch_size"], shuffle=True)

        train_losses = []
        train_accuracies = []

        criterion = assign_criterion(criterion)
        optimizer = assign_optimizer(
            optimizer, self.model, lr, betas, momentum, weight_decay
        )
        scheduler = assign_scheduler(scheduler, optimizer)

        self.model.to(device)

        for epoch in range(epochs):
            # --- Training ---
            self.model.train()
            train_loss, correct, total = 0, 0, 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_x.size(0)

            avg_train_loss = train_loss / total
            train_acc = correct / total
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)

            logging.info(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}"
            )

            # ---- Scheduler Step ----
            if scheduler is not None:
                scheduler.step()  # standard schedulers

        # Count number of parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: {num_params}")
        logging.warning(
            f"Final Train Accuracy: {train_acc:.4f} out of total train size: {total}"
        )
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        return self

    def predict(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        return preds.cpu().numpy()

    def predict_proba(self, x):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def score(self, x, y):
        self.model.eval()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        with torch.no_grad():
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
        return accuracy_score(preds, y)
