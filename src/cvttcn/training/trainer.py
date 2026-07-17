"""Training loop for the CvT-TCN model.

Wraps the optimizer, learning-rate schedule, mixed precision, gradient clipping,
metric tracking, early stopping, and checkpointing behind a small Trainer class.
The trainer is device-aware: mixed precision is only enabled on CUDA.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from cvttcn.config import Config
from cvttcn.training.metrics import ClassificationMetrics, compute_metrics
from cvttcn.training.utils import resolve_device


@dataclass
class EpochResult:
    """Aggregated loss and metrics for one pass over a loader."""

    loss: float
    accuracy: float
    macro_f1: float
    kappa: float


def _cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup for ``warmup_epochs`` then cosine decay to zero."""

    def lr_scale(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)


class Trainer:
    """Trains a model with AdamW, a cosine schedule, AMP, and early stopping."""

    def __init__(self, model: nn.Module, cfg: Config, device: Optional[torch.device] = None):
        self.cfg = cfg
        self.device = device if device is not None else resolve_device(cfg.train.device)
        self.model = model.to(self.device)
        self.num_classes = cfg.model.num_classes

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.train.label_smoothing)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

    # -- single passes -------------------------------------------------------
    def train_epoch(self, loader: DataLoader) -> EpochResult:
        self.model.train()
        return self._run_epoch(loader, train=True)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> EpochResult:
        self.model.eval()
        return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader: DataLoader, train: bool) -> EpochResult:
        total_loss, n_seen = 0.0, 0
        preds, targets = [], []
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                with autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)
                if train:
                    self._optimizer_step(loss)

            total_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            preds.append(logits.argmax(dim=1).detach().cpu())
            targets.append(yb.detach().cpu())

        y_pred = torch.cat(preds)
        y_true = torch.cat(targets)
        metrics = compute_metrics(y_true, y_pred, self.num_classes)
        return EpochResult(
            loss=total_loss / max(1, n_seen),
            accuracy=metrics.accuracy,
            macro_f1=metrics.macro_f1,
            kappa=metrics.kappa,
        )

    def _optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if self.cfg.train.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    # -- full training loop --------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> list[dict]:
        """Train with early stopping on validation accuracy.

        Restores the best-scoring weights before returning, and (optionally)
        saves them to ``checkpoint_path``. Returns a per-epoch history.
        """
        scheduler = _cosine_warmup_scheduler(
            self.optimizer, self.cfg.train.warmup_epochs, self.cfg.train.epochs
        )
        best_acc = -float("inf")
        best_state: Optional[dict] = None
        epochs_without_improvement = 0
        history: list[dict] = []

        for epoch in range(self.cfg.train.epochs):
            train_result = self.train_epoch(train_loader)
            val_result = self.evaluate(val_loader)
            lr = self.optimizer.param_groups[0]["lr"]
            scheduler.step()
            history.append(
                {"epoch": epoch, "lr": lr, "train": train_result, "val": val_result}
            )

            if val_result.accuracy > best_acc:
                best_acc = val_result.accuracy
                best_state = self._snapshot()
                epochs_without_improvement = 0
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.cfg.train.early_stopping_patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    # -- checkpointing -------------------------------------------------------
    def _snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.cfg.to_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: Union[str, Path]) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint
