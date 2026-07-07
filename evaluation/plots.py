
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_sample_paths(generator, H_values=(0.1, 0.3, 0.5, 0.7, 0.9), save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for H in H_values:
        path = generator.generate_path(H)
        ax.plot(path, label=f"H = {H:.2f}", linewidth=1.5)
    ax.set_title("Sample Fractional Brownian Motion Trajectories")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_loss_curves(train_losses, val_losses, title="Training Curve", save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_prediction_scatter(y_true, y_pred, title="H_true vs H_pred", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()) - 0.02, max(y_true.max(), y_pred.max()) + 0.02]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True H")
    ax.set_ylabel("Predicted H")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_error_vs_H(y_true, y_pred, n_bins=10, title="Error vs H", save_path=None):
    errors = np.abs(y_true - y_pred)
    bins = np.linspace(y_true.min(), y_true.max() + 1e-8, n_bins + 1)
    bin_idx = np.digitize(y_true, bins) - 1

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_err = np.array([
        errors[bin_idx == b].mean() if np.any(bin_idx == b) else np.nan
        for b in range(n_bins)
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_centers, mean_err, width=(bins[1] - bins[0]) * 0.9,
           color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xlabel("True H")
    ax.set_ylabel("Mean Absolute Error")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_residual_distribution(y_true, y_pred, title="Residual Distribution", save_path=None):
    residuals = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=40, color="darkorange", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
