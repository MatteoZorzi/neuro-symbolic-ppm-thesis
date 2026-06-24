"""Model evaluation and serialisable metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ..data.prefixes import PrefixBatch
from .logic import forbidden_probability_mass, prediction_violations


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    macro_f1: float
    top_k_accuracy: float
    violation_rate: float
    forbidden_mass: float
    targets: np.ndarray
    predictions: np.ndarray

    def metrics(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "top_k_accuracy": self.top_k_accuracy,
            "violation_rate": self.violation_rate,
            "forbidden_mass": self.forbidden_mass,
        }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    allowed_mask: torch.Tensor,
    device: torch.device,
    top_k: int = 3,
) -> EvaluationResult:
    """Evaluate predictive quality and process-conformance quality together."""

    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_forbidden_mass = 0.0
    total_violations = 0
    total_top_k = 0
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []

    for raw_batch in data_loader:
        batch: PrefixBatch = raw_batch.to(device)
        logits = model(batch.tokens, batch.lengths)
        batch_size = batch.targets.size(0)
        total_loss += criterion(logits, batch.targets).item()
        total_forbidden_mass += (
            forbidden_probability_mass(
                logits, batch.tokens, batch.lengths, allowed_mask
            ).item()
            * batch_size
        )
        total_violations += int(
            prediction_violations(
                logits, batch.tokens, batch.lengths, allowed_mask
            ).sum()
        )
        effective_k = min(top_k, logits.size(1))
        top_predictions = logits.topk(effective_k, dim=1).indices
        total_top_k += int(
            top_predictions.eq(batch.targets.unsqueeze(1)).any(dim=1).sum()
        )
        targets.append(batch.targets.cpu())
        predictions.append(logits.argmax(dim=1).cpu())

    if not targets:
        raise ValueError("Cannot evaluate an empty data loader.")
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(predictions).numpy()
    count = len(y_true)
    return EvaluationResult(
        loss=total_loss / count,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        top_k_accuracy=total_top_k / count,
        violation_rate=total_violations / count,
        forbidden_mass=total_forbidden_mass / count,
        targets=y_true,
        predictions=y_pred,
    )
