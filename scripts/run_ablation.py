#!/usr/bin/env python3
"""Controlled validation-only ablation study for the final HERA-HGT architecture.

Scientific guarantees
---------------------
1. TRAIN + VALIDATION only. This script never accepts or passes a test CSV.
2. One conceptual component is removed at a time.
3. The optimization recipe is fixed across variants unless the user explicitly
   changes the common command-line hyperparameters before starting the study.
4. The final selected fusion is FH-GNN-inspired adaptive fusion; concat and
   cross-attention are retained as explicit fusion baselines.
5. Both validation-selected-threshold and fixed-threshold-0.5 metrics are saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Variant:
    name: str
    ablation: str
    fusion: str
    description: str


VARIANTS: Dict[str, Variant] = {
    "full": Variant(
        "full", "full", "fhgnn_adaptive",
        "Complete HERA-HGT reference model.",
    ),
    "graph_only": Variant(
        "graph_only", "graph_only", "fhgnn_adaptive",
        "Hierarchical graph representation only.",
    ),
    "evidence_only": Variant(
        "evidence_only", "evidence_only", "fhgnn_adaptive",
        "Morgan + MACCS + AtomPair + descriptors only.",
    ),
    "no_brics": Variant(
        "no_brics", "no_brics", "fhgnn_adaptive",
        "Remove BRICS structural motifs.",
    ),
    "no_herg_motifs": Variant(
        "no_herg_motifs", "no_herg_motifs", "fhgnn_adaptive",
        "Remove explicit hERG pharmacophore motifs.",
    ),
    "no_relation_bias": Variant(
        "no_relation_bias", "no_relation_bias", "fhgnn_adaptive",
        "Disable relation-type attention bias only.",
    ),
    "no_distance_bias": Variant(
        "no_distance_bias", "no_distance_bias", "fhgnn_adaptive",
        "Disable topological-distance attention bias only.",
    ),
    "no_bond_bias": Variant(
        "no_bond_bias", "no_bond_bias", "fhgnn_adaptive",
        "Disable 10D bond-chemistry attention bias only.",
    ),
    "no_morgan": Variant(
        "no_morgan", "no_morgan", "fhgnn_adaptive",
        "Remove Morgan evidence.",
    ),
    "no_maccs": Variant(
        "no_maccs", "no_maccs", "fhgnn_adaptive",
        "Remove MACCS evidence.",
    ),
    "no_atompair": Variant(
        "no_atompair", "no_atompair", "fhgnn_adaptive",
        "Remove AtomPair evidence.",
    ),
    "no_descriptors": Variant(
        "no_descriptors", "no_descriptors", "fhgnn_adaptive",
        "Remove physicochemical descriptors.",
    ),
    "fusion_concat": Variant(
        "fusion_concat", "full", "concat",
        "Replace adaptive fusion by simple late concatenation.",
    ),
    "fusion_cross_attention": Variant(
        "fusion_cross_attention", "full", "cross_attention",
        "Replace adaptive fusion by the original cross-attention fusion.",
    ),
}

GROUPS: Dict[str, List[str]] = {
    "core": [
        "full",
        "graph_only",
        "evidence_only",
        "no_brics",
        "no_herg_motifs",
        "no_relation_bias",
        "no_distance_bias",
        "no_bond_bias",
    ],
    "evidence": [
        "full",
        "no_morgan",
        "no_maccs",
        "no_atompair",
        "no_descriptors",
    ],
    "fusion": [
        "full",
        "fusion_concat",
        "fusion_cross_attention",
    ],
}
GROUPS["all"] = list(dict.fromkeys(GROUPS["core"] + GROUPS["evidence"] + GROUPS["fusion"]))

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "auroc",
    "aupr",
    "loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-only controlled HERA-HGT ablation study")
    p.add_argument("--train-csv", default="data/paper_split/hERGAT_train_df.csv")
    p.add_argument("--val-csv", default="data/paper_split/hERGAT_valid_df.csv")
    p.add_argument("--out", default="ablation_study/hera_hgt_final")
    p.add_argument("--cache-dir", default=".cache/hera_hgt_ablation")
    p.add_argument("--group", choices=sorted(GROUPS), default="core")
    p.add_argument(
        "--variants",
        nargs="*",
        choices=sorted(VARIANTS),
        default=None,
        help="Optional explicit variants; overrides --group.",
    )

    # Frozen final-model recipe. Change these only before beginning a new study.
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=3e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--loss", choices=["bce", "balanced_bce"], default="bce")
    p.add_argument("--monitor", choices=["aupr", "auroc", "f1"], default="aupr")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--force", action="store_true", help="Delete/re-run an existing variant directory")
    return p.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=True)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _expected_signature(args: argparse.Namespace, v: Variant) -> Dict[str, Any]:
    return {
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "heads": args.heads,
        "layers": args.layers,
        "dropout": args.dropout,
        "loss": args.loss,
        "monitor": args.monitor,
        "fusion": v.fusion,
        "ablation": v.ablation,
        "validation_only": True,
    }


def _signature_matches(run_config: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    got = run_config.get("args", {})
    for key, value in expected.items():
        if got.get(key) != value:
            return False
    return True


def _run_variant(args: argparse.Namespace, v: Variant, root: Path) -> Path:
    run_dir = root / v.name / f"seed_{args.seed}"
    metrics_path = run_dir / "val_metrics.json"
    run_config_path = run_dir / "run_config.json"
    expected = _expected_signature(args, v)

    if metrics_path.exists() and run_config_path.exists() and not args.force:
        run_config = _load_json(run_config_path)
        if _signature_matches(run_config, expected):
            print(f"[REUSE] {v.name}: {metrics_path}")
            return run_dir
        raise RuntimeError(
            f"Existing run {run_dir} does not match the requested protocol. "
            "Use --force to replace it or choose a different --out directory."
        )

    if run_dir.exists() and args.force:
        import shutil
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--split-mode", "fixed_csv",
        "--train-csv", args.train_csv,
        "--val-csv", args.val_csv,
        "--validation-only",
        "--out", str(run_dir),
        "--cache-dir", args.cache_dir,
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--hidden-dim", str(args.hidden_dim),
        "--heads", str(args.heads),
        "--layers", str(args.layers),
        "--dropout", str(args.dropout),
        "--loss", args.loss,
        "--monitor", args.monitor,
        "--fusion", v.fusion,
        "--ablation", v.ablation,
        "--num-workers", str(args.num_workers),
    ]
    if args.device:
        cmd += ["--device", args.device]

    print("\n" + "=" * 92)
    print(f"RUN {v.name}: {v.description}")
    print(" ".join(cmd))
    print("=" * 92)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return run_dir


def _best_epoch(run_dir: Path) -> int | None:
    history_path = run_dir / "history.json"
    config_path = run_dir / "run_config.json"
    if not history_path.exists() or not config_path.exists():
        return None
    history = _load_json(history_path)
    cfg = _load_json(config_path)
    monitor = cfg.get("args", {}).get("monitor", "aupr")
    key = f"val_{monitor}"
    valid = [r for r in history if r.get(key) is not None]
    if not valid:
        return None
    best = max(valid, key=lambda r: float(r[key]))
    return int(best["epoch"])


def _collect_row(v: Variant, run_dir: Path, seed: int) -> Dict[str, Any]:
    selected = _load_json(run_dir / "val_metrics.json")
    fixed = _load_json(run_dir / "val_metrics_threshold_0_5.json")
    cfg = _load_json(run_dir / "run_config.json")

    row: Dict[str, Any] = {
        "variant": v.name,
        "ablation": v.ablation,
        "fusion": v.fusion,
        "seed": seed,
        "description": v.description,
        "best_epoch": _best_epoch(run_dir),
        "threshold": selected.get("threshold"),
        "prediction_mode": cfg.get("model_config", {}).get("prediction_mode"),
    }
    for m in METRICS:
        row[m] = selected.get(m)
        row[f"{m}_at_0_5"] = fixed.get(m)
    return row


def _attach_deltas(rows: List[Dict[str, Any]]) -> None:
    full = next((r for r in rows if r["variant"] == "full"), None)
    if full is None:
        return
    for row in rows:
        for m in ["accuracy", "f1", "mcc", "auroc", "aupr"]:
            a = row.get(m)
            b = full.get(m)
            row[f"delta_{m}_vs_full"] = (
                float(a) - float(b) if a is not None and b is not None else None
            )
            a05 = row.get(f"{m}_at_0_5")
            b05 = full.get(f"{m}_at_0_5")
            row[f"delta_{m}_at_0_5_vs_full"] = (
                float(a05) - float(b05)
                if a05 is not None and b05 is not None
                else None
            )


def main() -> None:
    args = parse_args()
    selected_names = args.variants if args.variants else GROUPS[args.group]
    variants = [VARIANTS[name] for name in selected_names]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    protocol = {
        "purpose": "Controlled validation-only HERA-HGT ablation study",
        "test_accessed": False,
        "selection": "Best checkpoint by validation monitor; threshold selected on validation F1; fixed 0.5 metrics also reported.",
        "common_hyperparameters": {
            "seed": args.seed,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "heads": args.heads,
            "layers": args.layers,
            "dropout": args.dropout,
            "loss": args.loss,
            "monitor": args.monitor,
        },
        "variants": [v.__dict__ for v in variants],
    }
    _write_json(out / "protocol.json", protocol)

    rows: List[Dict[str, Any]] = []
    for v in variants:
        run_dir = _run_variant(args, v, out)
        rows.append(_collect_row(v, run_dir, args.seed))
        _attach_deltas(rows)
        _write_csv(out / "ablation_summary.csv", rows)
        _write_json(out / "ablation_summary.json", rows)

    _attach_deltas(rows)
    _write_csv(out / "ablation_summary.csv", rows)
    _write_json(out / "ablation_summary.json", rows)

    print("\nABLATION STUDY COMPLETE")
    print(f"Summary CSV : {out / 'ablation_summary.csv'}")
    print(f"Summary JSON: {out / 'ablation_summary.json'}")
    print("TEST ACCESSED: False")


if __name__ == "__main__":
    main()
