#!/usr/bin/env python3
"""
Multi-seed validation-only confirmation for the three selected HERA-HGT V2 candidates.

Important experimental guarantees:
- Uses only the fixed TRAIN and VALIDATION CSV files.
- Never passes a test CSV to train.py.
- Keeps the candidate hyperparameters fixed; only the random seed changes.
- Reuses completed runs unless --force is supplied.
- Produces per-run and mean±std summaries for validation metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


CANDIDATES: dict[str, dict[str, Any]] = {
    "A_trial35_ranking": {
        "source_trial": 35,
        "lr": 2e-4,
        "weight_decay": 3e-4,
        "dropout": 0.2,
        "layers": 2,
        "batch_size": 128,
    },
    "B_trial21_balanced": {
        "source_trial": 21,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "layers": 2,
        "batch_size": 128,
    },
    "C_trial45_classification": {
        "source_trial": 45,
        "lr": 5e-5,
        "weight_decay": 3e-4,
        "dropout": 0.2,
        "layers": 4,
        "batch_size": 128,
    },
}

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "auroc",
    "aupr",
    "threshold",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Confirm the three selected HERA-HGT V2 candidates across multiple seeds using validation only."
    )
    p.add_argument("--train-csv", default="data/paper_split/hERGAT_train_df.csv")
    p.add_argument("--val-csv", default="data/paper_split/hERGAT_valid_df.csv")
    p.add_argument("--out", default="validation_multiseed/hera_hgt_v2_candidates")
    p.add_argument("--cache-dir", default=".cache/hera_hgt_deterministic_v2")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--monitor", choices=["aupr", "auroc", "f1"], default="aupr")
    p.add_argument("--loss", choices=["bce", "balanced_bce"], default="bce")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None, help="Optional explicit device, e.g. cuda or cuda:0")
    p.add_argument("--force", action="store_true", help="Re-run even if val_metrics.json already exists")
    return p.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_epoch_from_history(path: Path, monitor: str) -> int | None:
    if not path.exists():
        return None
    history = load_json(path)
    key = f"val_{monitor}"
    valid = []
    for row in history:
        value = row.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            valid.append((value, int(row["epoch"])))
    if not valid:
        return None
    valid.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return valid[0][1]


def run_one(
    project_root: Path,
    args: argparse.Namespace,
    candidate_name: str,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    run_dir = Path(args.out) / candidate_name / f"seed_{seed}"
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir
    metrics_path = run_dir / "val_metrics.json"

    if metrics_path.exists() and not args.force:
        print(f"\n[SKIP] {candidate_name} seed={seed}: existing {metrics_path}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        train_script = project_root / "scripts" / "train.py"

        cmd = [
            sys.executable,
            str(train_script),
            "--split-mode", "fixed_csv",
            "--train-csv", str((project_root / args.train_csv).resolve() if not Path(args.train_csv).is_absolute() else Path(args.train_csv)),
            "--val-csv", str((project_root / args.val_csv).resolve() if not Path(args.val_csv).is_absolute() else Path(args.val_csv)),
            "--validation-only",
            "--out", str(run_dir),
            "--cache-dir", str((project_root / args.cache_dir).resolve() if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir)),
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--batch-size", str(cfg["batch_size"]),
            "--lr", str(cfg["lr"]),
            "--weight-decay", str(cfg["weight_decay"]),
            "--hidden-dim", str(args.hidden_dim),
            "--heads", str(args.heads),
            "--layers", str(cfg["layers"]),
            "--dropout", str(cfg["dropout"]),
            "--loss", args.loss,
            "--monitor", args.monitor,
            "--num-workers", str(args.num_workers),
        ]
        if args.device:
            cmd.extend(["--device", args.device])

        print("\n" + "=" * 90)
        print(f"RUN {candidate_name} | source trial={cfg['source_trial']} | seed={seed}")
        print("=" * 90)
        print(" ".join(cmd))
        subprocess.run(cmd, cwd=project_root, check=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected result not found: {metrics_path}")

    metrics = load_json(metrics_path)
    row: dict[str, Any] = {
        "candidate": candidate_name,
        "source_trial": cfg["source_trial"],
        "seed": seed,
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "dropout": cfg["dropout"],
        "layers": cfg["layers"],
        "batch_size": cfg["batch_size"],
        "best_epoch": best_epoch_from_history(run_dir / "history.json", args.monitor),
    }
    for metric in METRICS:
        if metric == "threshold":
            row[metric] = float(metrics.get("threshold"))
        else:
            row[metric] = float(metrics.get(metric))
    return row


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for candidate_name, cfg in CANDIDATES.items():
        subset = [r for r in rows if r["candidate"] == candidate_name]
        if not subset:
            continue
        s: dict[str, Any] = {
            "candidate": candidate_name,
            "source_trial": cfg["source_trial"],
            "n_seeds": len(subset),
            "seeds": [r["seed"] for r in subset],
            "lr": cfg["lr"],
            "weight_decay": cfg["weight_decay"],
            "dropout": cfg["dropout"],
            "layers": cfg["layers"],
            "batch_size": cfg["batch_size"],
        }
        for metric in METRICS:
            vals = [float(r[metric]) for r in subset]
            s[f"{metric}_mean"] = mean(vals)
            s[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        epochs = [r["best_epoch"] for r in subset if r["best_epoch"] is not None]
        if epochs:
            s["best_epoch_mean"] = mean(epochs)
            s["best_epoch_std"] = stdev(epochs) if len(epochs) > 1 else 0.0
        summaries.append(s)

    # Ranking policy for confirmation:
    # 1) mean AUPR (primary), but allow candidates within 0.002 of the best mean AUPR;
    # 2) among near-best candidates, rank by mean F1, mean MCC, mean Accuracy, then mean AUROC;
    if summaries:
        best_aupr = max(s["aupr_mean"] for s in summaries)
        cutoff = best_aupr - 0.002
        for s in summaries:
            s["near_best_aupr"] = s["aupr_mean"] >= cutoff
            s["aupr_cutoff"] = cutoff
        summaries.sort(
            key=lambda s: (
                s["near_best_aupr"],
                s["f1_mean"] if s["near_best_aupr"] else s["aupr_mean"],
                s["mcc_mean"],
                s["accuracy_mean"],
                s["auroc_mean"],
            ),
            reverse=True,
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = project_root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Save exact experiment definition before execution.
    experiment = {
        "purpose": "multi-seed validation-only confirmation of selected HERA-HGT V2 candidates",
        "test_accessed": False,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "monitor": args.monitor,
        "loss": args.loss,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "candidates": CANDIDATES,
    }
    with (out_root / "experiment.json").open("w", encoding="utf-8") as f:
        json.dump(experiment, f, indent=2)

    rows: list[dict[str, Any]] = []
    for candidate_name, cfg in CANDIDATES.items():
        for seed in args.seeds:
            rows.append(run_one(project_root, args, candidate_name, cfg, seed))
            # Keep an always-current partial summary so an interrupted machine loses no analysis.
            write_csv(out_root / "runs.csv", rows)
            with (out_root / "runs.json").open("w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)

    summaries = summarize(rows)
    write_csv(out_root / "summary_candidates.csv", summaries)
    with (out_root / "summary_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + "=" * 90)
    print("MULTI-SEED VALIDATION COMPLETE — TEST NOT ACCESSED")
    print("=" * 90)
    for rank, s in enumerate(summaries, start=1):
        print(
            f"#{rank} {s['candidate']} | "
            f"AUPR {s['aupr_mean']:.6f} ± {s['aupr_std']:.6f} | "
            f"AUROC {s['auroc_mean']:.6f} ± {s['auroc_std']:.6f} | "
            f"F1 {s['f1_mean']:.6f} ± {s['f1_std']:.6f} | "
            f"ACC {s['accuracy_mean']:.6f} ± {s['accuracy_std']:.6f} | "
            f"MCC {s['mcc_mean']:.6f} ± {s['mcc_std']:.6f} | "
            f"threshold {s['threshold_mean']:.4f} ± {s['threshold_std']:.4f}"
        )

    if summaries:
        selected = summaries[0]
        with (out_root / "selected_candidate.json").open("w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        print("\nSelected validation candidate:")
        print(json.dumps(selected, indent=2))
        print(f"\nSaved summaries to: {out_root}")


if __name__ == "__main__":
    main()
