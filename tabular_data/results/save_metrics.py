from matplotlib import pyplot as plt
import os
import json
import torch
import numpy as np

def save_train_losses_accs(train_loss, test_loss, train_accs, test_accs, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 2 plots side by side

    # Plot train loss
    axes[0].plot(train_loss, label="Train Loss", color="tab:blue")
    axes[0].plot(test_loss, label="Test Loss", color="tab:red")
    axes[0].set_title("Train & Test Losses", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid()

    # Plot train accuracy
    axes[1].plot(train_accs, label="Train Accuracy", color="tab:blue")
    axes[1].plot(test_accs, label="Test Accuracy", color="tab:red")
    axes[1].set_title("Train & Test Accuracies", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid()

    # Reduce spacing
    plt.tight_layout(pad=1.0, w_pad=2.0)

    # Save
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict = {
        "train_loss": serialize(train_loss),
        "test_loss": serialize(test_loss),
        "train_acc": serialize(train_accs),
        "test_acc": serialize(test_accs),
    }
    save_path = os.path.join(save_dir, "training_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return


def save_train_loss_final_accs(train_loss, final_train_acc, final_test_acc, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 2 plots side by side

    # Plot train loss
    axes[0].plot(train_loss, label="Train Loss", color="tab:blue")
    axes[0].set_title("Train Loss", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid()

    # Bar chart for final accuracies
    bars = axes[1].bar(
        ["Train Acc", "Test Acc"],
        [final_train_acc, final_test_acc],
        color=["tab:blue", "tab:red"],
    )
    axes[1].set_title("Final Accuracies", fontsize=16)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=12)
    #axes[1].grid()

    # Annotate bar values
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f"{height:.2f}",
                         xy=(bar.get_x() + bar.get_width() / 2, height - 0.07),
                         xytext=(0, 5),  # offset
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=16, fontweight="bold")

    # Reduce spacing
    plt.tight_layout(pad=1.0, w_pad=2.0)

    # Save
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict = {
        "train_loss": serialize(train_loss),
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
    }
    save_path = os.path.join(save_dir, "training_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return


def save_final_accs(final_train_acc, final_test_acc, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Bar chart
    bars = ax.bar(
        ["Train Acc", "Test Acc"],
        [final_train_acc, final_test_acc],
        color=["tab:blue", "tab:red"],
    )
    ax.set_title("Final Accuracies", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    #ax.grid()

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height - 0.07),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save
    save_path = os.path.join(save_dir, "final_accuracies.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict = {
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
    }
    save_path = os.path.join(save_dir, "training_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return


def serialize(obj):
    try:
        return obj.tolist()  # for tensors/arrays
    except AttributeError:
        return obj


def save_hyperparams(hps, save_dir):
    """Save dataset, model, and training hyperparameters as JSON."""
    os.makedirs(save_dir, exist_ok=True)

    # Helper to make everything JSON serializable
    def make_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)  # e.g., "cuda:0" or "cpu"
        if isinstance(obj, (tuple, set)):
            return list(obj)  # convert to list
        if isinstance(obj, np.random.RandomState):
            seed = obj.get_state()[1][0]
            return int(seed)
        # Add more conversions if needed
        return obj

    # Walk recursively through dicts
    def serialize_dict(d):
        return {k: make_serializable(v) if not isinstance(v, dict) else serialize_dict(v)
                for k, v in d.items()}

    hps = serialize_dict(hps)

    # Save
    save_path = os.path.join(save_dir, "hyperparams.json")
    with open(save_path, "w") as f:
        json.dump(hps, f, indent=4)

    return


def save_sk_train_losses_accs(train_loss, train_accs, final_test_acc, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 2 plots side by side

    # Plot train loss
    axes[0].plot(train_loss, label="Train Loss", color="tab:blue")
    axes[0].set_title("Train Losses", fontsize=16)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].tick_params(axis="both", labelsize=12)
    #axes[0].legend(fontsize=12)
    axes[0].grid()

    # Plot train accuracy
    axes[1].plot(train_accs, label="Train Accuracy", color="tab:blue")
    axes[1].axhline(y=final_test_acc, color='red', linestyle='--', label='Test Accuracy')
    axes[1].set_title("Train Accuracies", fontsize=16)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid()

    # Reduce spacing
    plt.tight_layout(pad=1.0, w_pad=2.0)

    # Save
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict = {
        "train_loss": serialize(train_loss),
        "train_acc": serialize(train_accs),
        "final_test_acc": final_test_acc,
    }
    save_path = os.path.join(save_dir, "training_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return


def save_sk_final_test_acc(final_test_acc, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Bar chart
    bars = ax.bar(
        ["Test Acc"],
        [final_test_acc],
        color=["tab:red"],
    )
    ax.set_title("Final Test Accuracy", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    # ax.grid()

    # Annotate values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height - 0.07),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, fontweight="bold")

    plt.tight_layout()

    # Save
    save_path = os.path.join(save_dir, "final_test_accuracy.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    metrics_dict = {
        "final_test_acc": final_test_acc,
    }
    save_path = os.path.join(save_dir, "training_metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    return


def save_search_hyperparams(hps, best_hps, save_dir):
    """Save dataset, model, and training hyperparameters as JSON."""
    os.makedirs(save_dir, exist_ok=True)

    # Helper to make everything JSON serializable
    def make_serializable(obj):
        if isinstance(obj, torch.device):
            return str(obj)  # e.g., "cuda:0" or "cpu"
        if isinstance(obj, (tuple, set)):
            return list(obj)  # convert to list
        if isinstance(obj, np.random.RandomState):
            seed = obj.get_state()[1][0]
            return int(seed)
        # Add more conversions if needed
        return obj

    # Walk recursively through dicts
    def serialize_dict(d):
        return {k: make_serializable(v) if not isinstance(v, dict) else serialize_dict(v)
                for k, v in d.items()}

    hps = serialize_dict(hps)
    best_hps = serialize_dict(best_hps)

    # Save

    save_path = os.path.join(save_dir, "search_hps.json")
    with open(save_path, "w") as f:
        json.dump(hps, f, indent=4, default=str)

    save_path = os.path.join(save_dir, "best_hps.json")
    with open(save_path, "w") as f:
        json.dump(best_hps, f, indent=4, default=str)

    return