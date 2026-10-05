#!/usr/bin/env python

from pathlib import Path
import json

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]

PAPER_DIR = ROOT / "data" / "paper_split"
PREP_DIR = ROOT / "data" / "external" / "ctoxpred" / "prepared"


EXTERNAL = {
    "herg60": PREP_DIR / "herg60_as_published.csv",
    "herg70": PREP_DIR / "herg70_as_published.csv",
}


TRAIN_CSV = PAPER_DIR / "hERGAT_train_df.csv"
VAL_CSV = PAPER_DIR / "hERGAT_valid_df.csv"


def canonicalize(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def connectivity_key(smiles):
    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles).strip())

    if mol is None:
        return None

    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None

    if not key:
        return None

    return key.split("-")[0]


def load_reference(path):
    df = pd.read_csv(path)

    if "SMILES" not in df.columns:
        raise ValueError(f"SMILES absent dans {path}")

    out = pd.DataFrame()

    out["canonical_smiles"] = df["SMILES"].map(canonicalize)
    out["connectivity_key"] = df["SMILES"].map(connectivity_key)

    return out


def main():

    train = load_reference(TRAIN_CSV)
    val = load_reference(VAL_CSV)

    train_canon = set(
        train["canonical_smiles"].dropna()
    )

    val_canon = set(
        val["canonical_smiles"].dropna()
    )

    development_canon = (
        train_canon | val_canon
    )

    train_conn = set(
        train["connectivity_key"].dropna()
    )

    val_conn = set(
        val["connectivity_key"].dropna()
    )

    development_conn = (
        train_conn | val_conn
    )

    for name, path in EXTERNAL.items():

        df = pd.read_csv(path)

        if "SMILES" not in df.columns:
            raise ValueError(
                f"SMILES absent dans {path}"
            )

        if "Class" not in df.columns:
            raise ValueError(
                f"Class absent dans {path}"
            )

        df["canonical_smiles"] = (
            df["SMILES"].map(canonicalize)
        )

        df["connectivity_key"] = (
            df["SMILES"].map(connectivity_key)
        )

        invalid = df[
            "canonical_smiles"
        ].isna()

        if invalid.any():
            raise ValueError(
                f"{name}: {invalid.sum()} invalid SMILES"
            )

        overlap_train = (
            df["canonical_smiles"]
            .isin(train_canon)
        )

        overlap_val = (
            df["canonical_smiles"]
            .isin(val_canon)
        )

        overlap_dev = (
            df["canonical_smiles"]
            .isin(development_canon)
        )

        connectivity_dev = (
            df["connectivity_key"]
            .isin(development_conn)
        )

        # IMPORTANT:
        # on retire uniquement les overlaps CANONIQUES
        # avec TRAIN + VAL.
        #
        # Le connectivity overlap est seulement reporté.
        devclean = df.loc[
            ~overlap_dev
        ].copy()

        output_csv = (
            PREP_DIR /
            f"{name}_no_train_val_overlap.csv"
        )

        devclean[
            ["SMILES", "Class"]
        ].to_csv(
            output_csv,
            index=False,
        )

        audit = {
            "dataset": name,

            "original_n": int(len(df)),

            "train_exact_overlap": int(
                overlap_train.sum()
            ),

            "val_exact_overlap": int(
                overlap_val.sum()
            ),

            "train_val_union_exact_overlap": int(
                overlap_dev.sum()
            ),

            "train_val_connectivity_overlap": int(
                connectivity_dev.sum()
            ),

            "development_clean_n": int(
                len(devclean)
            ),

            "development_clean_blockers": int(
                (devclean["Class"] == 1).sum()
            ),

            "development_clean_nonblockers": int(
                (devclean["Class"] == 0).sum()
            ),

            "filter_rule": (
                "Remove exact canonical-SMILES overlap "
                "with hERGAT TRAIN or VALIDATION only. "
                "hERGAT TEST overlap is not removed."
            ),
        }

        audit_path = (
            PREP_DIR /
            f"{name}_no_train_val_audit.json"
        )

        with open(
            audit_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                audit,
                f,
                indent=2,
            )

        print("\n" + "=" * 70)
        print(name.upper())
        print("=" * 70)
        print(json.dumps(audit, indent=2))
        print("CSV:", output_csv)


if __name__ == "__main__":
    main()
