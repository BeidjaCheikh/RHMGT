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
import torch

from hera_hgt.checkpoint import load_checkpoint
from hera_hgt.data import collate_hera_hgt
from hera_hgt.hierarchy import featurize_hierarchical_smiles
from hera_hgt.train_utils import move_batch


def main():
    p = argparse.ArgumentParser(description="Predict hERG-blocker probability for one SMILES")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--smiles", required=True)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, feat_cfg, scaler, extra = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    item = featurize_hierarchical_smiles(args.smiles, feat_cfg)
    item["desc"] = scaler.transform(np.asarray(item["desc_raw"], dtype=np.float32))
    item["label"] = 0.0
    item["row_id"] = 0
    batch = move_batch(collate_hera_hgt([item]), device)
    with torch.no_grad():
        out = model(batch)
    p_block = float(out["prob"].item())
    threshold = float(args.threshold if args.threshold is not None else extra.get("selected_threshold", 0.5))
    attn = out["cross_attention"].mean(dim=1).squeeze(0).cpu().tolist()
    result = {
        "model": "HERA-HGT",
        "smiles": item["smiles"],
        "p_hERG_blocker": p_block,
        "threshold": threshold,
        "prediction": int(p_block >= threshold),
        "molecular_evidence_attention": dict(zip(["Morgan", "MACCS", "AtomPair", "Descriptors"], attn)),
        "motifs": item["motif_names"],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
