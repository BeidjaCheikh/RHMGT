from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import classification_metrics


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch and optionally request deterministic CUDA behavior.

    Determinism is enabled by default because HERA-HGT experiments are intended
    to be reproducible. ``warn_only=True`` avoids crashing if a rare operation
    has no deterministic CUDA implementation while still surfacing a warning.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Required by some deterministic CUDA GEMM paths. It is harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """Deterministically seed NumPy/Python inside a DataLoader worker."""
    del worker_id  # torch.initial_seed() already contains the worker-specific offset.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Create an independent DataLoader RNG.

    Keeping shuffling on a dedicated generator prevents model initialization or
    dropout RNG consumption from changing the minibatch order.
    """
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def class_balance_pos_weight(labels: Iterable[int | float]) -> float:
    """Return N_negative / N_positive for BCEWithLogitsLoss(pos_weight=...).

    This is the standard binary class-balancing ratio. For the fixed hERGAT
    training split (12,528 positive; 8,014 negative) it is about 0.64, which
    reduces the relative positive-class contribution because positives are the
    majority class.
    """
    y = np.asarray(list(labels), dtype=np.int64)
    if y.size == 0:
        raise ValueError("Cannot compute class balance from an empty label set")
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"Balanced BCE requires both classes; positives={positives}, negatives={negatives}"
        )
    return float(negatives / positives)


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _training_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_name: str,
    pos_weight: Optional[float],
) -> torch.Tensor:
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels)

    if loss_name == "balanced_bce":
        if pos_weight is None:
            raise ValueError("balanced_bce requires a train-derived pos_weight")
        pw = torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw)

    raise ValueError(f"Unknown loss_name={loss_name!r}")


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip: float = 5.0,
    loss_name: str = "bce",
    pos_weight: Optional[float] = None,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        loss = _training_bce_loss(
            out["logits"],
            batch["label"],
            loss_name=loss_name,
            pos_weight=pos_weight,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_model(model, loader, device, threshold: float = 0.5):
    """Evaluate with the ordinary unweighted BCE plus classification metrics.

    The reported validation/test loss stays unweighted even when balanced BCE is
    used for training. This keeps evaluation losses directly comparable between
    training-loss experiments; AUROC/AUPR/F1/etc. are unchanged by this choice.
    """
    model.eval()
    ys, ps, smiles, rows, cross, fusion_weights = [], [], [], [], [], []
    losses = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        loss = F.binary_cross_entropy_with_logits(out["logits"], batch["label"])
        losses.append(float(loss.cpu()))
        ys.extend(batch["label"].cpu().numpy().tolist())
        ps.extend(out["prob"].cpu().numpy().tolist())
        rows.extend(batch["row_id"].cpu().numpy().tolist())
        smiles.extend(batch["smiles"])
        cross_tensor = out.get("cross_attention")
        if torch.is_tensor(cross_tensor) and cross_tensor.numel() > 0:
            cross.extend(cross_tensor.mean(dim=1).cpu().numpy().tolist())

        fusion_tensor = out.get("fusion_weights")
        if torch.is_tensor(fusion_tensor) and fusion_tensor.numel() > 0:
            fusion_weights.extend(fusion_tensor.cpu().numpy().tolist())
    metrics = classification_metrics(ys, ps, threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    cross_arr = np.asarray(cross) if cross else np.empty((len(ys), 0), dtype=float)
    fusion_arr = (
        np.asarray(fusion_weights)
        if fusion_weights
        else np.empty((len(ys), 0), dtype=float)
    )
    return {
        "metrics": metrics,
        "y_true": np.asarray(ys),
        "y_prob": np.asarray(ps),
        "row_id": np.asarray(rows),
        "smiles": smiles,
        "cross_attention": cross_arr,
        "fusion_weights": fusion_arr,
        "fusion_weight_names": list(getattr(model, "fusion_weight_names", [])),
    }


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)
