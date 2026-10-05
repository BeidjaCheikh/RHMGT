#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from hera_hgt.chemistry import pharmacophore_registry_issues
from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import (
    HERAHGTDataset,
    collate_hera_hgt,
    fit_descriptor_scaler,
    load_records_csv_with_report,
    split_audit,
)
from hera_hgt.hierarchy import RELATION_VOCAB_SIZE
from hera_hgt.metrics import best_f1_threshold, classification_metrics
from hera_hgt.model import HERAHGT
from hera_hgt.train_utils import (
    class_balance_pos_weight,
    evaluate_model,
    make_generator,
    save_json,
    seed_worker,
    set_seed,
    train_epoch,
)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deterministic validation-only Optuna tuning for HERA-HGT. "
            "The test set is NEVER read by this script."
        )
    )

    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")

    p.add_argument("--out", default="tuning/hera_hgt_deterministic_v2")
    p.add_argument("--cache-dir", default=".cache/hera_hgt_deterministic_v2")
    p.add_argument("--study-name", default="hera_hgt_deterministic_v2")
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Default: SQLite study.db inside --out.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an existing compatible study. By default, the script refuses "
            "to reuse an existing study.db so that search-space changes cannot be mixed."
        ),
    )

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timeout", type=int, default=None, help="Optional global timeout in seconds")

    # Canonical HERA-HGT architecture dimensions kept fixed during this tuning stage.
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--max-atom-dist", type=int, default=10)
    p.add_argument("--max-motif-dist", type=int, default=12)
    p.add_argument("--max-motifs", type=int, default=48)

    p.add_argument(
        "--loss",
        choices=["bce", "balanced_bce"],
        default="bce",
        help=(
            "Training loss. For the current HERA-HGT final search, use the default BCE. "
            "balanced_bce is retained only for reproducibility of previous experiments."
        ),
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic/reproducible PyTorch behavior (default: true).",
    )

    # Candidate-selection policy after Optuna finishes.
    p.add_argument(
        "--aupr-tolerance",
        type=float,
        default=0.002,
        help=(
            "Keep completed trials whose best validation AUPR is within this absolute "
            "distance of the best AUPR, then select by validation F1, MCC and Accuracy."
        ),
    )
    p.add_argument(
        "--top-candidates",
        type=int,
        default=5,
        help="Number of near-best candidate configurations written to candidates.json.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def make_loader(
    dataset: HERAHGTDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_hera_hgt,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=make_generator(seed),
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _safe_float(value, default: float = float("nan")) -> float:
    try:
        value = float(value)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _trial_summary_dict(trial: optuna.trial.FrozenTrial) -> Dict:
    return {
        "trial": int(trial.number),
        "state": trial.state.name,
        "objective_val_aupr": None if trial.value is None else float(trial.value),
        "params": dict(trial.params),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "best_val_aupr": trial.user_attrs.get("best_val_aupr"),
        "best_val_auroc": trial.user_attrs.get("best_val_auroc"),
        "best_val_threshold": trial.user_attrs.get("best_val_threshold"),
        "best_val_accuracy": trial.user_attrs.get("best_val_accuracy"),
        "best_val_balanced_accuracy": trial.user_attrs.get("best_val_balanced_accuracy"),
        "best_val_precision": trial.user_attrs.get("best_val_precision"),
        "best_val_recall": trial.user_attrs.get("best_val_recall"),
        "best_val_f1": trial.user_attrs.get("best_val_f1"),
        "best_val_mcc": trial.user_attrs.get("best_val_mcc"),
        "best_val_f1_at_05": trial.user_attrs.get("best_val_f1_at_05"),
        "best_val_mcc_at_05": trial.user_attrs.get("best_val_mcc_at_05"),
        "best_val_accuracy_at_05": trial.user_attrs.get("best_val_accuracy_at_05"),
        "epochs_ran": trial.user_attrs.get("epochs_ran"),
        "trial_seed": trial.user_attrs.get("trial_seed"),
    }


def _candidate_sort_key(trial: optuna.trial.FrozenTrial):
    """
    Among near-best AUPR trials, prefer:
      1) validation F1 at validation-selected threshold
      2) validation MCC
      3) validation Accuracy
      4) validation AUPR
      5) validation AUROC

    All these metrics are validation-only. The test set is never involved.
    """
    return (
        _safe_float(trial.user_attrs.get("best_val_f1"), -math.inf),
        _safe_float(trial.user_attrs.get("best_val_mcc"), -math.inf),
        _safe_float(trial.user_attrs.get("best_val_accuracy"), -math.inf),
        _safe_float(trial.user_attrs.get("best_val_aupr"), -math.inf),
        _safe_float(trial.user_attrs.get("best_val_auroc"), -math.inf),
    )


def select_candidate_trials(
    study: optuna.Study,
    aupr_tolerance: float,
    top_k: int,
) -> tuple[optuna.trial.FrozenTrial, List[optuna.trial.FrozenTrial], float]:
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and np.isfinite(float(t.value))
    ]
    if not completed:
        raise RuntimeError("No completed Optuna trial is available for candidate selection")

    best_aupr = max(float(t.value) for t in completed)
    cutoff = best_aupr - float(aupr_tolerance)

    near_best = [t for t in completed if float(t.value) >= cutoff]
    near_best.sort(key=_candidate_sort_key, reverse=True)

    selected = near_best[0]
    return selected, near_best[: max(1, int(top_k))], cutoff


# -----------------------------------------------------------------------------
# Optuna objective
# -----------------------------------------------------------------------------

def objective_factory(
    args: argparse.Namespace,
    train_ds: HERAHGTDataset,
    val_ds: HERAHGTDataset,
    feat_cfg: FeaturizerConfig,
    out: Path,
    train_pos_weight: Optional[float],
):
    device = torch.device(args.device)

    def objective(trial: optuna.Trial) -> float:
        # ------------------------------------------------------------------
        # Discrete, interpretable and publication-friendly search space.
        # ------------------------------------------------------------------
        lr = trial.suggest_categorical(
            "lr",
            [5e-5, 1e-4, 2e-4, 3e-4],
        )
        weight_decay = trial.suggest_categorical(
            "weight_decay",
            [0.0, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3],
        )
        dropout = trial.suggest_categorical(
            "dropout",
            [0.10, 0.20, 0.30, 0.40],
        )
        layers = trial.suggest_categorical(
            "layers",
            [2, 3, 4],
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            [32, 64, 100, 128],
        )

        # Every trial starts from the same model/data RNG seed. This isolates the
        # effect of the hyperparameter configuration during this search stage.
        trial_seed = int(args.seed)
        set_seed(trial_seed, deterministic=args.deterministic)

        model_cfg = ModelConfig(
            hidden_dim=args.hidden_dim,
            n_heads=args.heads,
            n_layers=int(layers),
            dropout=float(dropout),
            relation_vocab=RELATION_VOCAB_SIZE,
            max_pair_dist=max(args.max_atom_dist, args.max_motif_dist),
            morgan_dim=feat_cfg.morgan_dim,
            maccs_dim=feat_cfg.maccs_dim,
            atom_pair_fp_dim=feat_cfg.atom_pair_fp_dim,
        )
        model_cfg.validate()

        model = HERAHGT(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        train_loader = make_loader(
            train_ds,
            int(batch_size),
            args.num_workers,
            True,
            trial_seed + 101,
        )
        val_loader = make_loader(
            val_ds,
            int(batch_size),
            args.num_workers,
            False,
            trial_seed + 202,
        )

        best_aupr = -float("inf")
        best_epoch = 0
        best_validation_metrics: Optional[Dict[str, float]] = None
        best_metrics_05: Optional[Dict[str, float]] = None
        bad = 0
        history: List[Dict] = []

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_name=args.loss,
                pos_weight=train_pos_weight,
            )

            # Evaluate once at 0.5. The probability vector is then reused to
            # compute a validation-selected threshold only when a new best AUPR
            # checkpoint is found.
            val_pack = evaluate_model(model, val_loader, device, threshold=0.5)
            m05 = val_pack["metrics"]

            val_aupr = float(m05["aupr"])
            val_auroc = float(m05["auroc"])

            row = {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(m05["loss"]),
                "val_aupr": val_aupr,
                "val_auroc": val_auroc,
                "val_accuracy_05": float(m05["accuracy"]),
                "val_balanced_accuracy_05": float(m05["balanced_accuracy"]),
                "val_precision_05": float(m05["precision"]),
                "val_recall_05": float(m05["recall"]),
                "val_f1_05": float(m05["f1"]),
                "val_mcc_05": float(m05["mcc"]),
            }
            history.append(row)

            if np.isfinite(val_aupr) and val_aupr > best_aupr + 1e-8:
                best_aupr = val_aupr
                best_epoch = int(epoch)
                bad = 0

                threshold, _ = best_f1_threshold(
                    val_pack["y_true"],
                    val_pack["y_prob"],
                )
                selected_metrics = classification_metrics(
                    val_pack["y_true"],
                    val_pack["y_prob"],
                    threshold,
                )

                best_validation_metrics = {
                    **selected_metrics,
                    "loss": float(m05["loss"]),
                }
                best_metrics_05 = {
                    "threshold": 0.5,
                    "accuracy": float(m05["accuracy"]),
                    "balanced_accuracy": float(m05["balanced_accuracy"]),
                    "precision": float(m05["precision"]),
                    "recall": float(m05["recall"]),
                    "f1": float(m05["f1"]),
                    "mcc": float(m05["mcc"]),
                    "auroc": val_auroc,
                    "aupr": val_aupr,
                    "loss": float(m05["loss"]),
                }
            else:
                bad += 1

            # Report current validation AUPR, not best-so-far, to the pruner.
            # Pruning is intentionally conservative for this graph Transformer.
            if np.isfinite(val_aupr):
                trial.report(val_aupr, step=epoch)

            if trial.should_prune():
                trial.set_user_attr("best_epoch", int(best_epoch))
                trial.set_user_attr("best_val_aupr", float(best_aupr))
                trial.set_user_attr("epochs_ran", int(len(history)))
                trial.set_user_attr("trial_seed", int(trial_seed))
                raise optuna.TrialPruned()

            if bad >= args.patience:
                break

        if best_validation_metrics is None or best_metrics_05 is None:
            raise RuntimeError(f"Trial {trial.number} did not produce finite validation AUPR")

        # Store the complete validation-only picture of the epoch selected by AUPR.
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("best_val_aupr", float(best_validation_metrics["aupr"]))
        trial.set_user_attr("best_val_auroc", float(best_validation_metrics["auroc"]))
        trial.set_user_attr("best_val_threshold", float(best_validation_metrics["threshold"]))
        trial.set_user_attr("best_val_accuracy", float(best_validation_metrics["accuracy"]))
        trial.set_user_attr(
            "best_val_balanced_accuracy",
            float(best_validation_metrics["balanced_accuracy"]),
        )
        trial.set_user_attr("best_val_precision", float(best_validation_metrics["precision"]))
        trial.set_user_attr("best_val_recall", float(best_validation_metrics["recall"]))
        trial.set_user_attr("best_val_f1", float(best_validation_metrics["f1"]))
        trial.set_user_attr("best_val_mcc", float(best_validation_metrics["mcc"]))
        trial.set_user_attr("best_val_f1_at_05", float(best_metrics_05["f1"]))
        trial.set_user_attr("best_val_mcc_at_05", float(best_metrics_05["mcc"]))
        trial.set_user_attr("best_val_accuracy_at_05", float(best_metrics_05["accuracy"]))
        trial.set_user_attr("trial_seed", int(trial_seed))
        trial.set_user_attr("epochs_ran", int(len(history)))

        trial_dir = out / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "trial": int(trial.number),
                "params": dict(trial.params),
                "trial_seed": int(trial_seed),
                "best_epoch": int(best_epoch),
                "best_validation_metrics_selected_threshold": best_validation_metrics,
                "best_validation_metrics_threshold_0_5": best_metrics_05,
                "training_loss": args.loss,
                "train_pos_weight": train_pos_weight,
                "deterministic": bool(args.deterministic),
                "test_accessed": False,
                "history": history,
            },
            trial_dir / "summary.json",
        )

        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Primary Optuna objective remains validation AUPR.
        return float(best_aupr)

    return objective


# -----------------------------------------------------------------------------
# Output files
# -----------------------------------------------------------------------------

def save_trials_csv(study: optuna.Study, path: Path) -> None:
    rows: List[Dict] = []

    for t in study.trials:
        row = {
            "number": int(t.number),
            "state": t.state.name,
            "objective_val_aupr": t.value,
            "best_epoch": t.user_attrs.get("best_epoch"),
            "best_val_aupr": t.user_attrs.get("best_val_aupr"),
            "best_val_auroc": t.user_attrs.get("best_val_auroc"),
            "best_val_threshold": t.user_attrs.get("best_val_threshold"),
            "best_val_accuracy": t.user_attrs.get("best_val_accuracy"),
            "best_val_balanced_accuracy": t.user_attrs.get("best_val_balanced_accuracy"),
            "best_val_precision": t.user_attrs.get("best_val_precision"),
            "best_val_recall": t.user_attrs.get("best_val_recall"),
            "best_val_f1": t.user_attrs.get("best_val_f1"),
            "best_val_mcc": t.user_attrs.get("best_val_mcc"),
            "best_val_accuracy_at_05": t.user_attrs.get("best_val_accuracy_at_05"),
            "best_val_f1_at_05": t.user_attrs.get("best_val_f1_at_05"),
            "best_val_mcc_at_05": t.user_attrs.get("best_val_mcc_at_05"),
            "epochs_ran": t.user_attrs.get("epochs_ran"),
            "trial_seed": t.user_attrs.get("trial_seed"),
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        rows.append(row)

    keys = sorted({k for r in rows for k in r}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        if not keys:
            return
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_validation_command(args: argparse.Namespace, params: Dict, seed: int, out_name: str) -> str:
    return f'''python scripts/train.py \\
  --split-mode fixed_csv \\
  --train-csv {args.train_csv} \\
  --val-csv {args.val_csv} \\
  --validation-only \\
  --out {out_name} \\
  --cache-dir .cache/hera_hgt_candidate_validation \\
  --seed {seed} \\
  --epochs 150 \\
  --patience 25 \\
  --batch-size {params['batch_size']} \\
  --lr {float(params['lr']):.12g} \\
  --weight-decay {float(params['weight_decay']):.12g} \\
  --hidden-dim {args.hidden_dim} \\
  --heads {args.heads} \\
  --layers {params['layers']} \\
  --dropout {params['dropout']} \\
  --loss {args.loss} \\
  --monitor aupr
'''


def build_final_test_command(args: argparse.Namespace, params: Dict) -> str:
    return f'''python scripts/train.py \\
  --split-mode fixed_csv \\
  --train-csv {args.train_csv} \\
  --val-csv {args.val_csv} \\
  --test-csv data/paper_split/hERGAT_test_df.csv \\
  --out runs/hera_hgt_final_candidate \\
  --cache-dir .cache/hera_hgt_final_candidate \\
  --seed {args.seed} \\
  --epochs 150 \\
  --patience 25 \\
  --batch-size {params['batch_size']} \\
  --lr {float(params['lr']):.12g} \\
  --weight-decay {float(params['weight_decay']):.12g} \\
  --hidden-dim {args.hidden_dim} \\
  --heads {args.heads} \\
  --layers {params['layers']} \\
  --dropout {params['dropout']} \\
  --loss {args.loss} \\
  --monitor aupr
'''


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.aupr_tolerance < 0:
        raise ValueError("--aupr-tolerance must be >= 0")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.patience <= 0:
        raise ValueError("--patience must be > 0")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    issues = pharmacophore_registry_issues()
    if issues:
        raise RuntimeError("Invalid pharmacophore registry: " + "; ".join(issues))

    # ------------------------------------------------------------------
    # Load TRAIN + VALIDATION only. There is no test argument in this file.
    # ------------------------------------------------------------------
    train_records, train_report = load_records_csv_with_report(
        args.train_csv,
        args.smiles_col,
        args.label_col,
    )
    val_records, val_report = load_records_csv_with_report(
        args.val_csv,
        args.smiles_col,
        args.label_col,
    )

    if len(train_records) == 0 or len(val_records) == 0:
        raise RuntimeError("Train or validation split is empty")

    train_pos_weight = (
        class_balance_pos_weight(r.label for r in train_records)
        if args.loss == "balanced_bce"
        else None
    )

    audit = split_audit(train_records, val_records, [])
    save_json(
        {
            "train": train_report.to_dict(),
            "val": val_report.to_dict(),
            "test_accessed": False,
            "training_loss": args.loss,
            "train_pos_weight": train_pos_weight,
            "deterministic": bool(args.deterministic),
            "train_val_audit": audit,
        },
        out / "data_audit.json",
    )

    feat_cfg = FeaturizerConfig(
        max_atom_dist=args.max_atom_dist,
        max_motif_dist=args.max_motif_dist,
        max_motifs=args.max_motifs,
    )

    # Descriptor normalization is fitted on TRAIN only.
    scaler = fit_descriptor_scaler(train_records)
    save_json(scaler.to_dict(), out / "descriptor_scaler_train_only.json")

    # Featurization does not depend on LR, WD, dropout, layers or batch size,
    # therefore all trials safely reuse the same cached examples.
    train_ds = HERAHGTDataset(
        train_records,
        feat_cfg,
        descriptor_scaler=scaler,
        cache_dir=Path(args.cache_dir) / "train",
    )
    val_ds = HERAHGTDataset(
        val_records,
        feat_cfg,
        descriptor_scaler=scaler,
        cache_dir=Path(args.cache_dir) / "val",
    )

    # ------------------------------------------------------------------
    # Optuna storage: protect against accidentally mixing old/new spaces.
    # ------------------------------------------------------------------
    default_db_path = out / "study.db"
    if args.storage is None:
        if default_db_path.exists() and not args.resume:
            raise RuntimeError(
                f"{default_db_path} already exists. This script intentionally refuses "
                "to mix a new search space with an old Optuna study. Either use a new "
                "--out directory, remove/rename the old tuning directory, or pass "
                "--resume only when the existing study was created with THIS SAME tune.py."
            )
        storage = f"sqlite:///{default_db_path.resolve()}"
    else:
        storage = args.storage

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
    )

    # Conservative pruning: let several complete trials establish a reference,
    # and do not prune a Transformer after only a few epochs.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=10,
        interval_steps=2,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=args.resume,
    )

    objective = objective_factory(
        args,
        train_ds,
        val_ds,
        feat_cfg,
        out,
        train_pos_weight,
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    # ------------------------------------------------------------------
    # Candidate selection.
    # Primary objective remains AUPR. F1/MCC/Accuracy are only used to
    # choose among trials within a narrow AUPR tolerance.
    # ------------------------------------------------------------------
    selected, candidate_trials, cutoff = select_candidate_trials(
        study,
        aupr_tolerance=args.aupr_tolerance,
        top_k=args.top_candidates,
    )

    pure_best_aupr_trial = study.best_trial

    selection_policy = {
        "primary_metric": "validation AUPR",
        "best_validation_aupr": float(pure_best_aupr_trial.value),
        "near_best_cutoff": float(cutoff),
        "aupr_tolerance": float(args.aupr_tolerance),
        "secondary_ranking_within_near_best_pool": [
            "validation F1 at validation-selected threshold",
            "validation MCC at validation-selected threshold",
            "validation Accuracy at validation-selected threshold",
            "validation AUPR",
            "validation AUROC",
        ],
        "test_accessed": False,
    }

    result = {
        "study_name": args.study_name,
        "objective": "maximize validation AUPR",
        "test_accessed": False,
        "training_loss": args.loss,
        "train_pos_weight": train_pos_weight,
        "deterministic": bool(args.deterministic),
        "search_space": {
            "lr": [5e-5, 1e-4, 2e-4, 3e-4],
            "weight_decay": [0.0, 1e-5, 1e-4, 3e-4, 5e-4, 1e-3],
            "dropout": [0.10, 0.20, 0.30, 0.40],
            "layers": [2, 3, 4],
            "batch_size": [32, 64, 100, 128],
        },
        "fixed_parameters": {
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "max_atom_dist": args.max_atom_dist,
            "max_motif_dist": args.max_motif_dist,
            "max_motifs": args.max_motifs,
        },
        "selection_policy": selection_policy,
        "pure_best_aupr_trial": _trial_summary_dict(pure_best_aupr_trial),
        "selected_candidate": _trial_summary_dict(selected),
    }

    save_json(result, out / "best_params.json")
    save_json(
        {
            "selection_policy": selection_policy,
            "candidates": [_trial_summary_dict(t) for t in candidate_trials],
        },
        out / "candidates.json",
    )
    save_trials_csv(study, out / "trials.csv")

    selected_params = selected.params

    # Three validation-only confirmation runs. These are intentionally generated
    # before any final test command is used.
    confirmation_seeds = [42, 123, 2026]
    validation_commands = []
    for s in confirmation_seeds:
        validation_commands.append(
            build_validation_command(
                args,
                selected_params,
                seed=s,
                out_name=f"runs/hera_hgt_candidate_seed_{s}_val",
            )
        )

    (out / "VALIDATION_COMMANDS.txt").write_text(
        "\n".join(validation_commands),
        encoding="utf-8",
    )

    # Kept for the eventual final evaluation after configuration confirmation.
    (out / "FINAL_COMMAND.txt").write_text(
        build_final_test_command(args, selected_params),
        encoding="utf-8",
    )

    print("\n=== DETERMINISTIC TUNING COMPLETE ===")
    print(json.dumps(result, indent=2))
    print("\nIMPORTANT: tune.py did NOT read the test set.")
    print(f"Trials table:        {out / 'trials.csv'}")
    print(f"Candidate pool:      {out / 'candidates.json'}")
    print(f"Selected candidate:  {out / 'best_params.json'}")
    print(f"Validation commands: {out / 'VALIDATION_COMMANDS.txt'}")
    print(f"Future test command: {out / 'FINAL_COMMAND.txt'}")


if __name__ == "__main__":
    main()
