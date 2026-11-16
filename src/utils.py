"""
Utility functions for reproducibility, metrics formatting, parameter counting,
and publication-quality visualizations.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
# Use non-interactive backend for headless execution / Colab
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

# Set clean styling defaults
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "figure.dpi": 300,
})


def set_seed(seed: int = 42) -> None:
    """
    Enforce global determinism across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed integer.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def count_parameters(model: nn.Module) -> int:
    """
    Calculate the total number of trainable parameters in a PyTorch model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_curves(
    history: Dict[str, List[float]],
    title: str = "Training and Validation Loss",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot and save training vs validation loss curves over epochs.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    ax.plot(epochs, history["train_loss"], label="Training Loss", color="#1f77b4")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#ff7f0e", linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "Actual vs. Forecasted Values",
    ylabel: str = "Closing Price (INR)",
    save_path: Optional[Union[str, Path]] = None,
    max_points: int = 120,
) -> None:
    """
    Plot actual values against predicted curves for multiple models.
    
    Args:
        actual: 1D ground truth array.
        predictions: Dictionary mapping model_name -> 1D prediction array.
        title: Plot title.
        ylabel: Y-axis label (physical units).
        save_path: File path to save figure.
        max_points: Number of latest test points to display.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pts = min(len(actual), max_points)
    time_steps = np.arange(pts)

    # Plot actual ground truth
    ax.plot(time_steps, actual[:pts], label="Actual", color="#111111", linewidth=2.2)

    palette = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, pred) in enumerate(predictions.items()):
        color = palette[i % len(palette)]
        ax.plot(
            time_steps,
            pred[:pts],
            label=name,
            color=color,
            linestyle="--" if "Baseline" in name else "-",
            alpha=0.85,
        )

    ax.set_xlabel("Test Time Steps (Days)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.legend(frameon=True, loc="best")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_bar(
    df: pd.DataFrame,
    metric: str = "RMSE",
    x_col: str = "Model",
    title: str = "Model Performance Comparison",
    ylabel: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Generate a formatted bar chart comparing models on a specific evaluation metric.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df[x_col], df[metric], color="#3b528b", edgecolor="black", alpha=0.85, width=0.5)

    # Annotate bar values on top
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel or metric)
    ax.set_title(title, pad=12, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiseries_comparison(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Grouped bar plot for sensitivity / ablation studies (e.g. sequence lengths, hidden sizes).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="deep", edgecolor="black")

    ax.set_title(title, pad=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=hue_col, frameon=True)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

