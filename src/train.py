"""
Reusable training engine for PyTorch recurrent networks.
Features mini-batch optimization, validation tracking, gradient clipping,
early stopping, and best-checkpoint persistence.
"""

import copy
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early stopping monitor. Halts training when validation loss stops improving.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_weights: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if validation loss improved.
        Returns True if early stopping criteria triggered.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Run one epoch of training over the training DataLoader."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds, _ = model(batch_x)
        loss = criterion(preds, batch_y)

        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch of validation."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 150,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    patience: int = 15,
    min_delta: float = 1e-5,
    grad_clip: float = 1.0,
    device: str = "cpu",
    checkpoint_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    print_every: int = 10,
) -> Tuple[nn.Module, Dict[str, List[float]], float, int]:
    """
    Complete training pipeline for PyTorch RNN/LSTM/GRU models.
    
    Returns:
        Tuple:
            - best_model: Model restored with best validation loss weights
            - history: Dict containing 'train_loss' and 'val_loss' histories
            - total_time: Total elapsed training time in seconds
            - best_epoch: The epoch index where best validation loss was achieved
    """
    dev = torch.device(device)
    model.to(dev)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, dev, grad_clip=grad_clip)
        val_loss = validate_epoch(model, val_loader, criterion, dev)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % print_every == 0 or epoch == 1 or epoch == epochs):
            print(f"Epoch [{epoch:03d}/{epochs:03d}] - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Check early stopping
        stopped = early_stopping.step(val_loss, model, epoch)
        if stopped:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {early_stopping.best_loss:.6f} (Epoch {early_stopping.best_epoch})")
            break

    total_time = time.time() - start_time

    # Restore best checkpoint weights
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)

    # Save checkpoint to disk if path provided
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": early_stopping.best_epoch,
                "best_val_loss": early_stopping.best_loss,
                "history": history,
            },
            path,
        )

    return model, history, total_time, early_stopping.best_epoch

