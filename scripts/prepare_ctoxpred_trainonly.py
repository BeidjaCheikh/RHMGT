#!/usr/bin/env python

from pathlib import Path
import json
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]

PAPER_DIR = ROOT / "data" / "paper_split"
PREP_DIR = ROOT / "data" / "external" / "ctoxpred" / "prepared"

TRAIN_CSV = PAPER_DIR / "hERGAT_train_df.csv"

EXTERNAL = {
    "herg60": PREP_DIR / "herg60_as_published.csv",
    "herg70": PREP_DIR / "herg70_as_published.csv",
}


def canonicalize(smiles):
    if pd.isna(smiles):
        return None

    smiles = str(smiles).strip()

    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)

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
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None

    if not inchikey:
        return None

    return inchikey.split("-")[0]


def main():

    # ------------------------------------------------------
    # hERGAT TRAIN reference ONLY
    # ------------------------------------------------------

    train = pd.read_csv(TRAIN_CSV)

    if "SMILES" not in train.columns:
        raise ValueError("SMILES column missing from hERGAT TRAIN.")

    train["canonical_smiles"] = (
        train["SMILES"].map(canonicalize)
    )

    train["connectivity_key"] = (
        train["SMILES"].map(connectivity_key)
    )

    train_canonical = set(
        train["canonical_smiles"].dropna()
    )

    train_connectivity = set(
        train["connectivity_key"].dropna()
    )

    print("hERGAT TRAIN")
    print("rows:", len(train))
    print(
        "unique canonical:",
        len(train_canonical),
    )

    # ------------------------------------------------------
    # External sets
    # ------------------------------------------------------

    for name, path in EXTERNAL.items():

        print("\n" + "=" * 72)
        print(name.upper())
        print("=" * 72)

        df = pd.read_csv(path)

        if "SMILES" not in df.columns:
            raise ValueError(
                f"{path}: SMILES column missing."
            )

        if "Class" not in df.columns:
            raise ValueError(
                f"{path}: Class column missing."
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
                f"{name}: {int(invalid.sum())} invalid SMILES."
            )

        # Exact canonical overlap with TRAIN
        overlap_train_exact = (
            df["canonical_smiles"]
            .isin(train_canonical)
        )

        # Connectivity overlap is only audited,
        # NOT automatically removed.
        overlap_train_connectivity = (
            df["connectivity_key"]
            .isin(train_connectivity)
        )

        train_clean = df.loc[
            ~overlap_train_exact
        ].copy()

        # --------------------------------------------------
        # Save evaluation CSV
        # --------------------------------------------------

        output_csv = (
            PREP_DIR /
            f"{name}_no_train_overlap.csv"
        )

        train_clean[
            ["SMILES", "Class"]
        ].to_csv(
            output_csv,
            index=False,
        )

        # --------------------------------------------------
        # Save overlap rows for transparency
        # --------------------------------------------------

        overlap_csv = (
            PREP_DIR /
            f"{name}_train_overlap_removed.csv"
        )

        df.loc[
            overlap_train_exact
        ].to_csv(
            overlap_csv,
            index=False,
        )

        # --------------------------------------------------
        # Audit
        # --------------------------------------------------

        audit = {
            "dataset": name,

            "original_n": int(len(df)),

            "train_exact_overlap_removed": int(
                overlap_train_exact.sum()
            ),

            "train_connectivity_overlap": int(
                overlap_train_connectivity.sum()
            ),

            "train_clean_n": int(
                len(train_clean)
            ),

            "train_clean_blockers": int(
                (train_clean["Class"] == 1).sum()
            ),

            "train_clean_nonblockers": int(
                (train_clean["Class"] == 0).sum()
            ),

            "filter_rule": (
                "Remove exact canonical-SMILES overlap "
                "with hERGAT TRAIN only. "
                "Validation and test overlaps are retained."
            ),
        }

        audit_path = (
            PREP_DIR /
            f"{name}_no_train_overlap_audit.json"
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

        print(
            json.dumps(
                audit,
                indent=2,
            )
        )

        print("CSV:", output_csv)


if __name__ == "__main__":
    main()
