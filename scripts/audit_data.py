#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

from hera_hgt.data import (
    clean_random_split,
    clean_scaffold_split,
    clean_unique_records,
    grouped_random_split,
    hergat_exact_split,
    load_records_csv_with_report,
    scaffold_split,
    split_audit,
)


def main():
    p = argparse.ArgumentParser(description="Audit HERA-HGT preprocessing and split leakage")
    p.add_argument("--csv", default="data/hERGAT_final_dataset.csv")
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
        ],
        default="hergat_exact",
    )
    p.add_argument("--hergat-split-json", default="data/hergat_paper_run1_split_indices.json")
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    records, report = load_records_csv_with_report(args.csv, args.smiles_col, args.label_col)
    cleaning_report = None

    if args.split_mode == "hergat_exact":
        tr, va, te = hergat_exact_split(records, args.hergat_split_json)
    elif args.split_mode == "grouped_random":
        tr, va, te = grouped_random_split(records, seed=args.seed)
    elif args.split_mode == "scaffold":
        tr, va, te = scaffold_split(records)
    elif args.split_mode == "clean_random":
        clean_records, cleaning_report = clean_unique_records(records)
        tr, va, te = clean_random_split(clean_records, seed=args.seed)
    elif args.split_mode == "clean_scaffold":
        clean_records, cleaning_report = clean_unique_records(records)
        tr, va, te = clean_scaffold_split(clean_records)
    else:
        raise ValueError(args.split_mode)

    payload = {
        "preprocessing": report.to_dict(),
        "cleaning": cleaning_report,
        "split": split_audit(tr, va, te),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
