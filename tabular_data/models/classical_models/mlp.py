import torch
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
from training.training_torch import assign_criterion, assign_optimizer, assign_scheduler
from torch.utils.data import DataLoader, TensorDataset


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, numNeurons, **kwargs):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = len(numNeurons)
        self.num_neurons = numNeurons
        assert len(numNeurons) == self.num_layers, (
            f"Number of neurons ({self.num_neurons}) must contain num_h_layers ({self.num_layers}) elements"
        )

        self.mlp = torch.nn.Sequential()
        for i, n_neurons in enumerate(self.num_neurons):
            if i == 0:
                in_size = input_size
            else:
                in_size = self.num_neurons[i - 1]
            out_size = self.num_neurons[i]

            linear_layer = torch.nn.Linear(
                in_features=in_size, out_features=out_size, bias=True
            )
            activation = torch.nn.ReLU()

            self.mlp.add_module(f"linear_{i}", linear_layer)
            self.mlp.add_module(f"activation_{i}", activation)

        if self.num_layers == 0:
            last_linear = torch.nn.Linear(
                in_features=input_size, out_features=output_size, bias=True
            )
        else:
            last_linear = torch.nn.Linear(
                in_features=self.num_neurons[-1], out_features=output_size, bias=True
            )
        self.mlp.add_module(f"linear_{len(self.num_neurons)}", last_linear)

    def forward(self, x):
        return self.mlp(x)


# Scikit-learn version of the MLP
class SKMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, data_params=None, model_params=None, training_params=None):
        self.model_class = MLP
        self.model_type = "sklearn"
        self.model_name = "mlp"
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
