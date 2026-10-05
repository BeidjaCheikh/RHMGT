#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hera_hgt.ablations import ABLATION_CHOICES, ablation_description, apply_ablation
from hera_hgt.checkpoint import load_checkpoint, save_checkpoint
from hera_hgt.chemistry import pharmacophore_registry_issues
from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import (
    HERAHGTDataset,
    clean_random_split,
    clean_scaffold_split,
    clean_unique_records,
    collate_hera_hgt,
    fit_descriptor_scaler,
    grouped_random_split,
    hergat_exact_split,
    load_records_csv_with_report,
    scaffold_split,
    split_audit,
    split_manifest,
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


def parse_args():
    p = argparse.ArgumentParser(description="Train HERA-HGT")
    p.add_argument("--csv", default="data/hERGAT_final_dataset.csv")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")
    p.add_argument(
        "--split-mode",
        choices=[
            "hergat_exact",
            "grouped_random",
            "scaffold",
            "clean_random",
            "clean_scaffold",
            "fixed_csv",
        ],
        default="hergat_exact",
        help=(
            "hergat_exact reproduces the supplied hERGAT paper_run1; "
            "clean_random canonicalizes/deduplicates/removes conflicting labels "
            "before a stratified 80/10/10 split; clean_scaffold applies a "
            "scaffold split to the same clean unique-compound dataset"
        ),
    )
    p.add_argument("--hergat-split-json", default="data/hergat_paper_run1_split_indices.json")
    p.add_argument("--out", default="runs/hera_hgt")
    p.add_argument("--cache-dir", default=".cache/hera_hgt")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--max-atom-dist", type=int, default=10)
    p.add_argument("--max-motif-dist", type=int, default=12)
    p.add_argument("--max-motifs", type=int, default=48)
    p.add_argument("--monitor", choices=["auroc", "aupr", "f1"], default="aupr")
    p.add_argument(
        "--fusion",
        choices=["cross_attention", "concat", "fhgnn_adaptive"],
        default="fhgnn_adaptive",
        help="Final multimodal fusion. cross_attention and concat are retained for ablation studies.",
    )
    p.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        default="full",
        help=(
            "Controlled architecture/component ablation. Exactly one conceptual "
            "component is removed per run; optimization hyperparameters are unchanged."
        ),
    )
    p.add_argument(
        "--loss",
        choices=["bce", "balanced_bce"],
        default="bce",
        help="Training loss. balanced_bce computes pos_weight=N_neg/N_pos from TRAIN only.",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic/reproducible PyTorch behavior (default: true).",
    )
    p.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Train/select on train+validation and never read/evaluate the test CSV. "
            "Use this while comparing training choices such as BCE vs balanced BCE."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _loader(records, feat_cfg, scaler, cache_dir, split, batch_size, workers, shuffle, seed):
    ds = HERAHGTDataset(
        records,
        feat_cfg,
        descriptor_scaler=scaler,
        cache_dir=Path(cache_dir) / split if cache_dir else None,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_hera_hgt,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=make_generator(seed),
    )


def _save_predictions(pack, path, threshold):
    arr = pack["cross_attention"]
    df = pd.DataFrame({
        "row_id": pack["row_id"],
        "smiles": pack["smiles"],
        "y_true": pack["y_true"].astype(int),
        "y_prob": pack["y_prob"],
        "y_pred": (pack["y_prob"] >= threshold).astype(int),
    })
    if arr.ndim == 2 and arr.shape[1] == 4:
        for i, name in enumerate(["Morgan", "MACCS", "AtomPair", "Descriptors"]):
            df[f"cross_attn_{name}"] = arr[:, i]

    fusion_weights = pack.get("fusion_weights")
    fusion_names = pack.get("fusion_weight_names", [])
    if (
        isinstance(fusion_weights, np.ndarray)
        and fusion_weights.ndim == 2
        and fusion_weights.shape[1] == len(fusion_names)
        and len(fusion_names) > 0
    ):
        for i, name in enumerate(fusion_names):
            df[f"fusion_weight_{name}"] = fusion_weights[:, i]
    df.to_csv(path, index=False)


def _resolve_splits(args):
    if args.split_mode == "fixed_csv":
        if not all([args.train_csv, args.val_csv]):
            raise ValueError("fixed_csv requires --train-csv and --val-csv")
        if not args.validation_only and not args.test_csv:
            raise ValueError("fixed_csv requires --test-csv unless --validation-only is used")

        tr, tr_report = load_records_csv_with_report(args.train_csv, args.smiles_col, args.label_col)
        va, va_report = load_records_csv_with_report(args.val_csv, args.smiles_col, args.label_col)
        if args.validation_only:
            te = []
            load_report = {
                "train": tr_report.to_dict(),
                "val": va_report.to_dict(),
                "test_accessed": False,
            }
        else:
            te, te_report = load_records_csv_with_report(args.test_csv, args.smiles_col, args.label_col)
            load_report = {
                "train": tr_report.to_dict(),
                "val": va_report.to_dict(),
                "test": te_report.to_dict(),
                "test_accessed": True,
            }
        return tr, va, te, load_report

    if any([args.train_csv, args.val_csv, args.test_csv]):
        raise ValueError("Use --split-mode fixed_csv when passing explicit train/val/test CSVs")

    records, report = load_records_csv_with_report(args.csv, args.smiles_col, args.label_col)
    cleaning_report = None

    if args.split_mode == "hergat_exact":
        tr, va, te = hergat_exact_split(records, args.hergat_split_json, validate_smiles=True)
    elif args.split_mode == "grouped_random":
        tr, va, te = grouped_random_split(records, seed=args.seed, frac_train=0.8, frac_val=0.1)
    elif args.split_mode == "scaffold":
        tr, va, te = scaffold_split(records, frac_train=0.8, frac_val=0.1)
    elif args.split_mode == "clean_random":
        clean_records, cleaning_report = clean_unique_records(records)
        tr, va, te = clean_random_split(clean_records, seed=args.seed, frac_train=0.8, frac_val=0.1)
    elif args.split_mode == "clean_scaffold":
        clean_records, cleaning_report = clean_unique_records(records)
        tr, va, te = clean_scaffold_split(clean_records, frac_train=0.8, frac_val=0.1)
    else:
        raise ValueError(args.split_mode)

    load_report = {"all": report.to_dict()}
    if cleaning_report is not None:
        load_report["cleaning"] = cleaning_report
    return tr, va, te, load_report


def main():
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    issues = pharmacophore_registry_issues()
    if issues:
        raise RuntimeError("Invalid pharmacophore registry: " + "; ".join(issues))

    feat_cfg = FeaturizerConfig(
        max_atom_dist=args.max_atom_dist,
        max_motif_dist=args.max_motif_dist,
        max_motifs=args.max_motifs,
    )
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
        relation_vocab=RELATION_VOCAB_SIZE,
        max_pair_dist=max(args.max_atom_dist, args.max_motif_dist),
        morgan_dim=feat_cfg.morgan_dim,
        maccs_dim=feat_cfg.maccs_dim,
        atom_pair_fp_dim=feat_cfg.atom_pair_fp_dim,
        fusion_type=args.fusion,
    )

    # Apply exactly one controlled ablation after constructing the canonical
    # full-model configs. Featurizer ablations therefore also get a distinct
    # cache hash, preventing stale full-model graph features from being reused.
    feat_cfg, model_cfg = apply_ablation(args.ablation, feat_cfg, model_cfg)
    model_cfg.validate()

    train_records, val_records, test_records, load_report = _resolve_splits(args)
    if len(train_records) == 0 or len(val_records) == 0:
        raise RuntimeError("Train or validation split is empty")
    if not args.validation_only and len(test_records) == 0:
        raise RuntimeError("Test split is empty")

    audit = split_audit(train_records, val_records, test_records)

    # Strict clean modes must be provably leakage-free before training begins.
    if args.split_mode in {"clean_random", "clean_scaffold"}:
        overlap = audit["canonical_smiles_overlap"]
        if any(int(v) != 0 for v in overlap.values()):
            raise RuntimeError(f"Leakage detected in clean split: {overlap}")
        if int(audit["canonical_smiles_with_conflicting_labels"]) != 0:
            raise RuntimeError("Conflicting canonical-SMILES labels remain after cleaning")
        for split_name in ("train", "val", "test"):
            part = audit["parts"][split_name]
            if int(part["rows"]) != int(part["unique_canonical_smiles"]):
                raise RuntimeError(f"{split_name}: duplicate molecules remain in clean split")

    manifest = split_manifest(train_records, val_records, test_records, args.split_mode)
    save_json(load_report, out / "preprocessing_report.json")
    save_json(manifest, out / "split_manifest.json")
    save_json(audit, out / "split_audit.json")

    # Critical: descriptors are standardized using TRAIN statistics only.
    desc_scaler = fit_descriptor_scaler(train_records)
    save_json(desc_scaler.to_dict(), out / "descriptor_scaler.json")

    # Independent deterministic DataLoader RNGs make the final train.py run
    # reproduce the minibatch order used by tune.py for the same seed/config.
    loaders = {
        "train": _loader(train_records, feat_cfg, desc_scaler, args.cache_dir, "train", args.batch_size, args.num_workers, True, args.seed + 101),
        "val": _loader(val_records, feat_cfg, desc_scaler, args.cache_dir, "val", args.batch_size, args.num_workers, False, args.seed + 202),
    }
    if not args.validation_only:
        loaders["test"] = _loader(
            test_records, feat_cfg, desc_scaler, args.cache_dir, "test",
            args.batch_size, args.num_workers, False, args.seed + 303
        )

    train_pos_weight = (
        class_balance_pos_weight(r.label for r in train_records)
        if args.loss == "balanced_bce"
        else None
    )

    device = torch.device(args.device)
    model = HERAHGT(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    summary = {
        "model_name": "HERA-HGT",
        "ablation": args.ablation,
        "ablation_description": ablation_description(args.ablation),
        "full_name": "hERG-Aware Relational Hierarchical Graph Transformer",
        "args": vars(args),
        "split_mode": args.split_mode,
        "sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "split_audit": audit,
        "model_config": model_cfg.to_dict(),
        "featurizer_config": feat_cfg.to_dict(),
        "descriptor_scaling": "mean/std fitted on train only",
        "training_loss": args.loss,
        "train_pos_weight": train_pos_weight,
        "deterministic": bool(args.deterministic),
        "selection_policy": (
            "validation-only: checkpoint and threshold selected on validation; test not read"
            if args.validation_only
            else "checkpoint and threshold selected only on validation; test evaluated after selection"
        ),
    }
    save_json(summary, out / "run_config.json")

    best = -float("inf")
    bad = 0
    history = []
    best_path = out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            device,
            loss_name=args.loss,
            pos_weight=train_pos_weight,
        )
        val = evaluate_model(model, loaders["val"], device, threshold=0.5)
        raw_score = float(val["metrics"][args.monitor])
        score = raw_score if np.isfinite(raw_score) else -float(val["metrics"]["loss"])
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val["metrics"].items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if (not best_path.exists()) or score > best + 1e-8:
            best = score
            bad = 0
            save_checkpoint(best_path, model, feat_cfg, desc_scaler, {
                "best_epoch": epoch,
                "best_val_score": best,
                "best_val_raw_monitor": raw_score,
                "monitor": args.monitor,
                "split_mode": args.split_mode,
                "training_loss": args.loss,
                "train_pos_weight": train_pos_weight,
                "deterministic": bool(args.deterministic),
            })
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    save_json(history, out / "history.json")

    # Reload the best VALIDATION checkpoint before touching the test set.
    model, loaded_feat_cfg, loaded_scaler, extra = load_checkpoint(best_path, map_location=device)
    model = model.to(device)

    val_pack = evaluate_model(model, loaders["val"], device, threshold=0.5)
    threshold, _ = best_f1_threshold(val_pack["y_true"], val_pack["y_prob"])
    val_metrics = classification_metrics(val_pack["y_true"], val_pack["y_prob"], threshold)
    val_metrics_05 = classification_metrics(val_pack["y_true"], val_pack["y_prob"], 0.5)

    # Keep the ordinary validation BCE in both reports for auditability.
    val_loss = float(val_pack["metrics"].get("loss", float("nan")))
    val_metrics["loss"] = val_loss
    val_metrics_05["loss"] = val_loss

    if args.validation_only:
        final_extra = dict(extra)
        final_extra.update({
            "selected_threshold": threshold,
            "threshold_selected_on": "validation",
            "val_metrics_selected_threshold": val_metrics,
            "val_metrics_threshold_0_5": val_metrics_05,
            "ablation": args.ablation,
            "test_accessed": False,
        })
        save_checkpoint(best_path, model, loaded_feat_cfg, loaded_scaler, final_extra)
        save_json(val_metrics, out / "val_metrics.json")
        save_json(val_metrics_05, out / "val_metrics_threshold_0_5.json")
        _save_predictions(val_pack, out / "val_predictions.csv", threshold)
        print("\nFINAL VALIDATION-ONLY")
        print(json.dumps({
            "ablation": args.ablation,
            "threshold": threshold,
            "val": val_metrics,
            "val_threshold_0_5": val_metrics_05,
            "test_accessed": False,
        }, indent=2))
        return

    test_pack = evaluate_model(model, loaders["test"], device, threshold=threshold)
    test_metrics = test_pack["metrics"]
    test_metrics_05 = classification_metrics(test_pack["y_true"], test_pack["y_prob"], 0.5)

    final_extra = dict(extra)
    final_extra.update({
        "selected_threshold": threshold,
        "threshold_selected_on": "validation",
        "val_metrics_selected_threshold": val_metrics,
        "val_metrics_threshold_0_5": val_metrics_05,
        "ablation": args.ablation,
        "test_metrics_selected_threshold": test_metrics,
        "test_metrics_threshold_0_5": test_metrics_05,
    })
    save_checkpoint(best_path, model, loaded_feat_cfg, loaded_scaler, final_extra)
    save_json(val_metrics, out / "val_metrics.json")
    save_json(val_metrics_05, out / "val_metrics_threshold_0_5.json")
    save_json(test_metrics, out / "test_metrics.json")
    save_json(test_metrics_05, out / "test_metrics_threshold_0_5.json")
    _save_predictions(val_pack, out / "val_predictions.csv", threshold)
    _save_predictions(test_pack, out / "test_predictions.csv", threshold)

    print("\nFINAL")
    print(json.dumps({"threshold": threshold, "val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
