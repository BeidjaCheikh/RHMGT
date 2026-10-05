#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from hera_hgt.ablations import apply_ablation
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

METRIC_NAMES: Tuple[str, ...] = ("accuracy", "f1", "auroc", "aupr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validation-only multi-objective tuning of the frozen G0 HERA-HGT "
            "architecture (full + fhgnn_adaptive). The test split is never read."
        )
    )
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")

    p.add_argument("--out", default="tuning/g0_multiobjective")
    p.add_argument("--cache-dir", default=".cache/hera_hgt_g0_tuning")
    p.add_argument("--study-name", default="hera_hgt_g0_multiobjective")
    p.add_argument("--storage", default=None)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume only a study created with this exact script/search space.",
    )

    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=36)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--timeout", type=int, default=None)

    # Frozen G0 architecture dimensions.
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--max-atom-dist", type=int, default=10)
    p.add_argument("--max-motif-dist", type=int, default=12)
    p.add_argument("--max-motifs", type=int, default=48)
    p.add_argument(
        "--loss",
        choices=["bce", "balanced_bce"],
        default="bce",
        help="Keep BCE for the final G0 tuning unless reproducing an older experiment.",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Baseline G0 validation metrics. Used only to rank/interpret Pareto candidates;
    # they are NOT part of the optimization objective and do not touch the test set.
    p.add_argument("--baseline-acc", type=float, default=0.8812720848056537)
    p.add_argument("--baseline-f1", type=float, default=0.9025522041763341)
    p.add_argument("--baseline-auroc", type=float, default=0.9253103058134505)
    p.add_argument("--baseline-aupr", type=float, default=0.9463438030990943)
    p.add_argument(
        "--noninferiority-tolerance",
        type=float,
        default=0.001,
        help=(
            "Absolute validation tolerance used only to describe candidates as "
            "approximately non-inferior to G0 on all four metrics."
        ),
    )
    p.add_argument("--top-candidates", type=int, default=5)
    return p.parse_args()


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


def _finite_float(value, default: float = -math.inf) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def baseline_dict(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "accuracy": float(args.baseline_acc),
        "f1": float(args.baseline_f1),
        "auroc": float(args.baseline_auroc),
        "aupr": float(args.baseline_aupr),
    }


def trial_metrics(trial: optuna.trial.FrozenTrial) -> Dict[str, float]:
    return {
        name: _finite_float(trial.user_attrs.get(f"best_val_{name}"))
        for name in METRIC_NAMES
    }


def metric_deltas(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {name: float(metrics[name] - baseline[name]) for name in METRIC_NAMES}


def candidate_characteristics(
    trial: optuna.trial.FrozenTrial,
    baseline: Dict[str, float],
    tol: float,
) -> Dict:
    metrics = trial_metrics(trial)
    deltas = metric_deltas(metrics, baseline)
    vals = np.asarray([deltas[m] for m in METRIC_NAMES], dtype=float)
    strict_improvements = int(np.sum(vals > 0.0))
    dominates = bool(np.all(vals >= 0.0) and np.any(vals > 0.0))
    strictly_better_all_four = bool(np.all(vals > 0.0))
    approximately_noninferior = bool(np.all(vals >= -float(tol)))
    return {
        "trial": int(trial.number),
        "params": dict(trial.params),
        "metrics": metrics,
        "delta_vs_g0": deltas,
        "strict_improvements_count": strict_improvements,
        "dominates_g0": dominates,
        "strictly_better_all_four": strictly_better_all_four,
        "approximately_noninferior_all_four": approximately_noninferior,
        "worst_delta": float(np.min(vals)),
        "mean_delta": float(np.mean(vals)),
        "best_epoch": trial.user_attrs.get("best_epoch"),
        "threshold": trial.user_attrs.get("best_val_threshold"),
    }


def candidate_sort_key(item: Dict) -> Tuple:
    """Prefer true all-metric improvements, then robust maximin gains."""
    return (
        int(item["strictly_better_all_four"]),
        int(item["dominates_g0"]),
        int(item["approximately_noninferior_all_four"]),
        int(item["strict_improvements_count"]),
        float(item["worst_delta"]),
        float(item["mean_delta"]),
        float(item["metrics"]["f1"]),
        float(item["metrics"]["accuracy"]),
        float(item["metrics"]["aupr"]),
        float(item["metrics"]["auroc"]),
    )


def objective_factory(
    args: argparse.Namespace,
    train_ds: HERAHGTDataset,
    val_ds: HERAHGTDataset,
    feat_cfg: FeaturizerConfig,
    out: Path,
    train_pos_weight: Optional[float],
):
    device = torch.device(args.device)

    def objective(trial: optuna.Trial) -> Tuple[float, float, float, float]:
        # Narrow search around the frozen, already strong G0 architecture.
        lr = trial.suggest_categorical("lr", [5e-5, 1e-4, 2e-4])
        weight_decay = trial.suggest_categorical("weight_decay", [1e-4, 3e-4, 5e-4])
        dropout = trial.suggest_categorical("dropout", [0.10, 0.20, 0.30])
        layers = trial.suggest_categorical("layers", [3, 4])
        batch_size = trial.suggest_categorical("batch_size", [64, 128])

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
            fusion_type="fhgnn_adaptive",
        )
        # Explicitly freeze the architecture to the complete G0 model.
        _, model_cfg = apply_ablation("full", feat_cfg, model_cfg)
        model_cfg.validate()

        model = HERAHGT(model_cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        train_loader = make_loader(
            train_ds, int(batch_size), args.num_workers, True, trial_seed + 101
        )
        val_loader = make_loader(
            val_ds, int(batch_size), args.num_workers, False, trial_seed + 202
        )

        # Checkpoint selection remains AUPR, matching the historical G0 protocol.
        # Multi-objective selection happens ACROSS trials using ACC/F1/AUROC/AUPR.
        best_aupr = -float("inf")
        best_epoch = 0
        best_metrics: Optional[Dict[str, float]] = None
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
            val_pack = evaluate_model(model, val_loader, device, threshold=0.5)
            m05 = val_pack["metrics"]
            val_aupr = float(m05["aupr"])

            history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(m05["loss"]),
                    "val_accuracy_05": float(m05["accuracy"]),
                    "val_f1_05": float(m05["f1"]),
                    "val_auroc": float(m05["auroc"]),
                    "val_aupr": val_aupr,
                }
            )

            if np.isfinite(val_aupr) and val_aupr > best_aupr + 1e-8:
                best_aupr = val_aupr
                best_epoch = int(epoch)
                bad = 0

                threshold, _ = best_f1_threshold(val_pack["y_true"], val_pack["y_prob"])
                selected = classification_metrics(
                    val_pack["y_true"], val_pack["y_prob"], threshold
                )
                best_metrics = {**selected, "loss": float(m05["loss"])}
                best_metrics_05 = {
                    "threshold": 0.5,
                    "accuracy": float(m05["accuracy"]),
                    "f1": float(m05["f1"]),
                    "auroc": float(m05["auroc"]),
                    "aupr": float(m05["aupr"]),
                    "loss": float(m05["loss"]),
                }
            else:
                bad += 1

            if bad >= args.patience:
                break

        if best_metrics is None or best_metrics_05 is None:
            raise RuntimeError(f"Trial {trial.number} produced no finite validation checkpoint")

        for name in METRIC_NAMES:
            trial.set_user_attr(f"best_val_{name}", float(best_metrics[name]))
        trial.set_user_attr("best_val_threshold", float(best_metrics["threshold"]))
        trial.set_user_attr("best_epoch", int(best_epoch))
        trial.set_user_attr("trial_seed", int(trial_seed))
        trial.set_user_attr("epochs_ran", int(len(history)))
        trial.set_user_attr("fusion", "fhgnn_adaptive")
        trial.set_user_attr("ablation", "full")

        trial_dir = out / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "trial": int(trial.number),
                "params": dict(trial.params),
                "architecture": {
                    "fusion": "fhgnn_adaptive",
                    "ablation": "full",
                    "hidden_dim": int(args.hidden_dim),
                    "heads": int(args.heads),
                },
                "trial_seed": int(trial_seed),
                "best_epoch": int(best_epoch),
                "best_validation_metrics_selected_threshold": best_metrics,
                "best_validation_metrics_threshold_0_5": best_metrics_05,
                "checkpoint_selection_metric": "validation AUPR",
                "multi_objective_trial_metrics": list(METRIC_NAMES),
                "training_loss": args.loss,
                "test_accessed": False,
                "history": history,
            },
            trial_dir / "summary.json",
        )

        result = tuple(float(best_metrics[name]) for name in METRIC_NAMES)

        del model, optimizer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result  # accuracy, f1, auroc, aupr

    return objective


def save_trials_csv(
    study: optuna.Study,
    path: Path,
    baseline: Dict[str, float],
    tol: float,
) -> None:
    pareto_numbers = {int(t.number) for t in study.best_trials}
    rows: List[Dict] = []
    for t in study.trials:
        row: Dict = {
            "number": int(t.number),
            "state": t.state.name,
            "is_pareto": int(t.number) in pareto_numbers,
            "best_epoch": t.user_attrs.get("best_epoch"),
            "threshold": t.user_attrs.get("best_val_threshold"),
            "trial_seed": t.user_attrs.get("trial_seed"),
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None:
            item = candidate_characteristics(t, baseline, tol)
            for m in METRIC_NAMES:
                row[m] = item["metrics"][m]
                row[f"delta_{m}_vs_g0"] = item["delta_vs_g0"][m]
            row["strict_improvements_count"] = item["strict_improvements_count"]
            row["dominates_g0"] = item["dominates_g0"]
            row["strictly_better_all_four"] = item["strictly_better_all_four"]
            row["approximately_noninferior_all_four"] = item[
                "approximately_noninferior_all_four"
            ]
            row["worst_delta"] = item["worst_delta"]
            row["mean_delta"] = item["mean_delta"]
        rows.append(row)

    keys = sorted({k for r in rows for k in r}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if keys:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def build_validation_command(
    args: argparse.Namespace,
    trial_number: int,
    params: Dict,
    seed: int,
) -> str:
    return f'''python scripts/train.py \\
  --split-mode fixed_csv \\
  --train-csv {args.train_csv} \\
  --val-csv {args.val_csv} \\
  --validation-only \\
  --out tuning/g0_confirmation/trial_{trial_number:04d}/seed_{seed} \\
  --cache-dir .cache/hera_hgt_g0_confirmation \\
  --seed {seed} \\
  --epochs {args.epochs} \\
  --patience {args.patience} \\
  --batch-size {params['batch_size']} \\
  --lr {float(params['lr']):.12g} \\
  --weight-decay {float(params['weight_decay']):.12g} \\
  --hidden-dim {args.hidden_dim} \\
  --heads {args.heads} \\
  --layers {params['layers']} \\
  --dropout {params['dropout']} \\
  --loss {args.loss} \\
  --monitor aupr \\
  --ablation full \\
  --fusion fhgnn_adaptive
'''


def main() -> None:
    args = parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("--epochs and --patience must be > 0")
    if args.noninferiority_tolerance < 0:
        raise ValueError("--noninferiority-tolerance must be >= 0")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    issues = pharmacophore_registry_issues()
    if issues:
        raise RuntimeError("Invalid pharmacophore registry: " + "; ".join(issues))

    train_records, train_report = load_records_csv_with_report(
        args.train_csv, args.smiles_col, args.label_col
    )
    val_records, val_report = load_records_csv_with_report(
        args.val_csv, args.smiles_col, args.label_col
    )
    if not train_records or not val_records:
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
            "architecture": "G0 full HERA-HGT",
            "fusion": "fhgnn_adaptive",
            "ablation": "full",
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
    scaler = fit_descriptor_scaler(train_records)
    save_json(scaler.to_dict(), out / "descriptor_scaler_train_only.json")

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

    default_db = out / "study.db"
    if args.storage is None:
        if default_db.exists() and not args.resume:
            raise RuntimeError(
                f"{default_db} already exists. Use a new --out directory or --resume "
                "only for this same G0 multi-objective study."
            )
        storage = f"sqlite:///{default_db.resolve()}"
    else:
        storage = args.storage

    sampler = optuna.samplers.NSGAIISampler(seed=args.seed, population_size=12)
    study = optuna.create_study(
        study_name=args.study_name,
        directions=["maximize", "maximize", "maximize", "maximize"],
        sampler=sampler,
        storage=storage,
        load_if_exists=args.resume,
    )

    # Force the current champion configuration into the study as a reference.
    # This guarantees that tuning never "forgets" G0.
    baseline_params = {
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "dropout": 0.20,
        "layers": 4,
        "batch_size": 128,
    }
    if not args.resume or len(study.trials) == 0:
        study.enqueue_trial(baseline_params)

    objective = objective_factory(args, train_ds, val_ds, feat_cfg, out, train_pos_weight)
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    base = baseline_dict(args)
    pareto_items = [
        candidate_characteristics(t, base, args.noninferiority_tolerance)
        for t in study.best_trials
    ]
    pareto_items.sort(key=candidate_sort_key, reverse=True)

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    all_items = [
        candidate_characteristics(t, base, args.noninferiority_tolerance)
        for t in completed
    ]
    all_items.sort(key=candidate_sort_key, reverse=True)

    strict_all_four = [x for x in all_items if x["strictly_better_all_four"]]
    dominates = [x for x in all_items if x["dominates_g0"]]

    recommendation = {
        "baseline_g0": base,
        "test_accessed": False,
        "decision_rule": (
            "Do not replace G0 from one seed alone. Confirm the best Pareto candidates "
            "across multiple validation seeds. A candidate that strictly improves all four "
            "metrics is especially strong; otherwise prefer robust non-dominated trade-offs "
            "and keep G0 if confirmation does not show a stable global gain."
        ),
        "strictly_better_all_four_found": bool(strict_all_four),
        "number_strictly_better_all_four": len(strict_all_four),
        "number_dominating_g0": len(dominates),
        "recommended_for_multiseed_confirmation": pareto_items[: max(1, args.top_candidates)],
    }

    save_json(
        {
            "metric_order": list(METRIC_NAMES),
            "directions": ["maximize"] * 4,
            "baseline_g0": base,
            "search_space": {
                "lr": [5e-5, 1e-4, 2e-4],
                "weight_decay": [1e-4, 3e-4, 5e-4],
                "dropout": [0.10, 0.20, 0.30],
                "layers": [3, 4],
                "batch_size": [64, 128],
            },
            "fixed_architecture": {
                "fusion": "fhgnn_adaptive",
                "ablation": "full",
                "hidden_dim": args.hidden_dim,
                "heads": args.heads,
                "loss": args.loss,
                "checkpoint_selection": "validation AUPR",
            },
            "test_accessed": False,
        },
        out / "protocol.json",
    )
    save_json({"pareto_front": pareto_items}, out / "pareto_front.json")
    save_json(recommendation, out / "recommendation.json")
    save_trials_csv(study, out / "trials.csv", base, args.noninferiority_tolerance)

    confirmation_seeds = [42, 2026, 3407]
    commands: List[str] = []
    for item in pareto_items[: min(3, max(1, args.top_candidates))]:
        for seed in confirmation_seeds:
            commands.append(
                build_validation_command(
                    args,
                    trial_number=int(item["trial"]),
                    params=item["params"],
                    seed=seed,
                )
            )
    (out / "MULTISEED_CONFIRMATION_COMMANDS.txt").write_text(
        "\n".join(commands), encoding="utf-8"
    )

    print("\n=== G0 MULTI-OBJECTIVE TUNING COMPLETE ===")
    print(json.dumps(recommendation, indent=2))
    print("\nIMPORTANT: the TEST split was never read.")
    print(f"All trials:       {out / 'trials.csv'}")
    print(f"Pareto front:     {out / 'pareto_front.json'}")
    print(f"Recommendation:   {out / 'recommendation.json'}")
    print(f"Confirmation cmd: {out / 'MULTISEED_CONFIRMATION_COMMANDS.txt'}")


if __name__ == "__main__":
    main()
