import torch
import logging
from sklearn.metrics import accuracy_score
from merlin_additional.loss import NKernelAlignment


def training_torch(
    model_dict,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    lr,
    betas,
    momentum,
    weight_decay,
    device=torch.device("cpu"),
):
    model_type = model_dict["type"]
    model_name = model_dict["name"]
    model = model_dict["model"]

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    criterion = assign_criterion(criterion)
    optimizer = assign_optimizer(optimizer, model, lr, betas, momentum, weight_decay)
    scheduler = assign_scheduler(scheduler, optimizer)

    model.to(device)

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss, correct, total = 0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
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

        # --- Test ---
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                test_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(batch_y).sum().item()
                test_total += batch_x.size(0)

        avg_test_loss = test_loss / test_total
        test_acc = test_correct / test_total
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)

        logging.info(
            f"Epoch {epoch + 1}/{epochs} "
            f"| Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} "
            f"| Test Loss: {avg_test_loss:.4f}, Acc: {test_acc:.4f}"
        )

        # ---- Scheduler Step ----
        if scheduler is not None:
            scheduler.step()  # standard schedulers

    logging.warning(
        f"Final Train Accuracy: {train_acc:.4f} | Final Test Accuracy: {test_acc:.4f}"
    )

    return {
        "type": model_type,
        "name": model_name,
        "model": model,
        "train_losses": train_losses,
        "train_accs": train_accuracies,
        "test_losses": test_losses,
        "test_accs": test_accuracies,
    }


def training_reuploading(
    model_dict,
    x_train,
    x_test,
    y_train,
    y_test,
    track_history,
    max_epochs,
    learning_rate,
    batch_size,
    patience,
    tau,
    convergence_tolerance,
):
    model_type = model_dict["type"]
    model_name = model_dict["name"]
    model = model_dict["model"]

    model.fit(
        x_train,
        y_train,
        track_history=track_history,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        tau=tau,
        convergence_tolerance=convergence_tolerance,
    )
    train_losses = model.training_history_["loss"]

    y_pred_test = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    y_pred_train = model.predict(x_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    logging.warning(
        f"Final Train Accuracy: {train_acc:.4f} | Final Test Accuracy: {test_acc:.4f}"
    )
    return {
        "type": model_type,
        "name": model_name,
        "model": model,
        "train_losses": train_losses,
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
    }


def training_sklearn_q_kernel(
    model_dict,
    train_loader,
    x_train,
    x_test,
    y_train,
    y_test,
    optimizer,
    lr,
    epochs,
    pre_train,
    device,
):
    model_type = model_dict["type"]
    model_name = model_dict["name"]
    model = model_dict["model"]
    optimizable_model = model.quantum_kernel

    if pre_train:
        optimizable_model.to(device)

        criterion = NKernelAlignment()
        optimizer = assign_optimizer(optimizer, optimizable_model, lr)

        train_losses = []
        for epoch in range(epochs):
            train_loss = 0
            total = 0
            for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                # Add scaling before entering circuit
                x_batch = model.scale(x_batch)
                outputs = model.quantum_kernel(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x_batch.size(0)
                total += x_batch.size(0)

            avg_train_loss = train_loss / total
            train_losses.append(avg_train_loss)

        model.pretraining_done()
    else:
        train_losses = []

    # Get kernel matrices
    kernel_matrix_train, kernel_matrix_test = model.get_q_kernels(x_train, x_test)
    # Detach and convert to numpy
    kernel_matrix_train = kernel_matrix_train.detach().cpu().numpy()
    kernel_matrix_test = kernel_matrix_test.detach().cpu().numpy()

    model.fit(kernel_matrix_train, y_train)

    y_pred_test = model.predict(kernel_matrix_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    y_pred_train = model.predict(kernel_matrix_train)
    train_acc = accuracy_score(y_train, y_pred_train)

    logging.warning(
        f"Final Train Accuracy: {train_acc:.4f} | Final Test Accuracy: {test_acc:.4f}"
    )
    return {
        "type": model_type,
        "name": model_name,
        "model": model,
        "train_losses": train_losses,
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
    }


def assign_criterion(criterion):
    if criterion == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif criterion == "MSELoss":
        return torch.nn.MSELoss()
    elif criterion == "BCELoss":
        return torch.nn.BCELoss()
    elif criterion == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif criterion == "L1Loss":
        return torch.nn.L1Loss()
    else:
        raise ValueError(f"Unknown criterion {criterion}")


def assign_optimizer(
    optimizer, model, lr, betas=(0.9, 0.999), momentum=0.9, weight_decay=0
):
    if optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif optimizer == "AdaGrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "RMSprop":
        return torch.optim.RMSprop(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer == "Rprop":
        return torch.optim.Rprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer {optimizer}")


def assign_scheduler(scheduler, optimizer, step_size=10, gamma=0.1, T_max=50):
    if scheduler is None or scheduler == "None":
        return None
    elif scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 80], gamma=gamma
        )
    elif scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler {scheduler}")
