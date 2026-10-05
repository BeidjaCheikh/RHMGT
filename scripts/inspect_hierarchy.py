#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter
import numpy as np

from hera_hgt.config import FeaturizerConfig
from hera_hgt.hierarchy import featurize_hierarchical_smiles


def main():
    p = argparse.ArgumentParser(description="Inspect HERA-HGT hierarchical graph construction")
    p.add_argument("--smiles", required=True)
    p.add_argument("--max-motifs", type=int, default=48)
    args = p.parse_args()
    item = featurize_hierarchical_smiles(args.smiles, FeaturizerConfig(max_motifs=args.max_motifs))
    print("Canonical SMILES:", item["smiles"])
    print("Atoms:", item["n_atoms"], "Motifs:", item["n_motifs"], "Global index:", item["global_index"])
    print("Motifs:")
    for i, (name, atoms, fam) in enumerate(zip(item["motif_names"], item["motif_atoms"], item["motif_families"])):
        print(f"  {i:02d} family={fam} {name:24s} atoms={atoms}")
    rel = item["relation"]
    mask = item["pair_mask"]
    counts = Counter(rel[mask].astype(int).tolist())
    print("Relation counts:", dict(sorted(counts.items())))
    nonzero_bond_pairs = int(np.any(item["pair_bond10"] != 0, axis=-1).sum())
    print("Pairs with non-zero 10D hERGAT chemical bond encoding:", nonzero_bond_pairs)


if __name__ == "__main__":
    main()
