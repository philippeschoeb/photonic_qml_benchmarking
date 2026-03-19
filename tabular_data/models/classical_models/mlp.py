import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
import logging
from training.training_torch import assign_criterion, assign_optimizer, assign_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
        max_steps = self.training_params.get("max_steps")
        convergence_interval = self.training_params.get("convergence_interval", 200)
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
        steps_per_epoch = max(1, len(loader))
        if max_steps is None:
            max_steps = max(1, epochs * steps_per_epoch)

        train_iter = iter(loader)
        step_loss_history = []
        window_train_loss = 0.0
        window_correct = 0
        window_total = 0
        converged = False

        with tqdm(total=max_steps, desc="Torch Training Progress", unit="step") as pbar:
            for step in range(max_steps):
                try:
                    batch_x, batch_y = next(train_iter)
                except StopIteration:
                    train_iter = iter(loader)
                    batch_x, batch_y = next(train_iter)

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                batch_size = batch_x.size(0)
                step_loss = loss.item()
                step_loss_history.append(step_loss)
                window_train_loss += step_loss * batch_size
                _, predicted = outputs.max(1)
                window_correct += predicted.eq(batch_y).sum().item()
                window_total += batch_size
                pbar.update(1)

                if np.isnan(step_loss):
                    logging.info("nan encountered at step %s. Training aborted.", step + 1)
                    break

                if convergence_interval is not None and len(step_loss_history) > 2 * convergence_interval:
                    average1 = np.mean(step_loss_history[-convergence_interval:])
                    average2 = np.mean(
                        step_loss_history[-2 * convergence_interval : -convergence_interval]
                    )
                    std1 = np.std(step_loss_history[-convergence_interval:])
                    if np.abs(average2 - average1) <= std1 / np.sqrt(convergence_interval) / 2:
                        logging.info(
                            "Model %s converged after %s steps.",
                            self.model.__class__.__name__,
                            step + 1,
                        )
                        converged = True

                if ((step + 1) % steps_per_epoch == 0) or (step == max_steps - 1) or converged:
                    avg_train_loss = window_train_loss / max(window_total, 1)
                    train_acc = window_correct / max(window_total, 1)
                    train_losses.append(avg_train_loss)
                    train_accuracies.append(train_acc)
                    logging.info(
                        "Step %s/%s | Train Loss: %.4f, Acc: %.4f",
                        step + 1,
                        max_steps,
                        avg_train_loss,
                        train_acc,
                    )
                    if scheduler is not None:
                        scheduler.step()
                    window_train_loss = 0.0
                    window_correct = 0
                    window_total = 0

                if converged:
                    break

        # Count number of parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Number of parameters: {num_params}")

        final_train_acc = self.score(x.cpu().numpy(), y.cpu().numpy())
        logging.warning(
            "Final Train Accuracy: %.4f out of total train size: %s",
            final_train_acc,
            len(dataset),
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
