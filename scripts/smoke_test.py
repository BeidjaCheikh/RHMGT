#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import DescriptorScaler, collate_hera_hgt
from hera_hgt.hierarchy import RELATION_VOCAB_SIZE, featurize_hierarchical_smiles
from hera_hgt.model import HERAHGT


def main():
    smiles = [
        "CN(C)CCc1ccc(OC)c2ccccc12",
        "CCN1CCC(CC1)Oc1ccc(Cl)cc1",
    ]
    feat_cfg = FeaturizerConfig(max_motifs=24)
    items = []
    raw_desc = []
    for i, smi in enumerate(smiles):
        x = featurize_hierarchical_smiles(smi, feat_cfg)
        raw_desc.append(np.asarray(x["desc_raw"], dtype=np.float32))
        x["label"] = float(i % 2)
        x["row_id"] = i
        items.append(x)
    arr = np.stack(raw_desc)
    scale = arr.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    scaler = DescriptorScaler(arr.mean(axis=0).astype(np.float32), scale)
    for x in items:
        x["desc"] = scaler.transform(np.asarray(x["desc_raw"], dtype=np.float32))

    batch = collate_hera_hgt(items)
    cfg = ModelConfig(hidden_dim=64, n_heads=4, n_layers=2, fp_encoder_hidden=64, desc_encoder_hidden=32,
                      classifier_hidden=64, relation_vocab=RELATION_VOCAB_SIZE)
    model = HERAHGT(cfg)
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], batch["label"])
    loss.backward()
    print("SMOKE TEST OK")
    print("atom_dim:", batch["node_x"].shape[-1], "bond_dim:", batch["pair_bond10"].shape[-1])
    print("Morgan dim:", batch["fp_morgan"].shape[-1])
    print("cross_attention shape:", tuple(out["cross_attention"].shape))
    print("loss:", float(loss.detach()))


if __name__ == "__main__":
    main()
