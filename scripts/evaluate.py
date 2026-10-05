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

from hera_hgt.checkpoint import load_checkpoint
from hera_hgt.data import HERAHGTDataset, collate_hera_hgt, load_records_csv
from hera_hgt.metrics import classification_metrics
from hera_hgt.train_utils import evaluate_model


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained HERA-HGT checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-dir", default=".cache/hera_hgt_eval")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, feat_cfg, scaler, extra = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    threshold = float(args.threshold if args.threshold is not None else extra.get("selected_threshold", 0.5))
    records = load_records_csv(args.csv, args.smiles_col, args.label_col)
    ds = HERAHGTDataset(records, feat_cfg, scaler, args.cache_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_hera_hgt)
    pack = evaluate_model(model, loader, device, threshold=threshold)
    metrics = classification_metrics(pack["y_true"], pack["y_prob"], threshold)
    print(json.dumps(metrics, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "row_id": pack["row_id"], "smiles": pack["smiles"],
            "y_true": pack["y_true"].astype(int), "y_prob": pack["y_prob"],
            "y_pred": (pack["y_prob"] >= threshold).astype(int),
        })
        if pack["cross_attention"].ndim == 2 and pack["cross_attention"].shape[1] == 4:
            for i, name in enumerate(["Morgan", "MACCS", "AtomPair", "Descriptors"]):
                df[f"cross_attn_{name}"] = pack["cross_attention"][:, i]
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
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
