#!/usr/bin/env python

from pathlib import Path
import json

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "external" / "ctoxpred" / "raw"
OUT_DIR = ROOT / "data" / "external" / "ctoxpred" / "prepared"

PAPER_DIR = ROOT / "data" / "paper_split"

OUT_DIR.mkdir(parents=True, exist_ok=True)


EXTERNAL_FILES = {
    "herg60": RAW_DIR / "eval_set_herg_60.csv",
    "herg70": RAW_DIR / "eval_set_herg_70.csv",
}


HERGAT_FILES = {
    "train": PAPER_DIR / "hERGAT_train_df.csv",
    "val": PAPER_DIR / "hERGAT_valid_df.csv",
    "test": PAPER_DIR / "hERGAT_test_df.csv",
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
    """
    First block of InChIKey.
    Useful as an additional connectivity-level overlap audit.
    """

    if pd.isna(smiles):
        return None

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        return None

    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None

    if not key:
        return None

    return key.split("-")[0]


def prepare_hergat_reference():

    result = {}

    for split, path in HERGAT_FILES.items():

        df = pd.read_csv(path)

        if "SMILES" not in df.columns:
            raise ValueError(
                f"{path}: colonne SMILES absente"
            )

        tmp = pd.DataFrame()

        tmp["SMILES"] = df["SMILES"].astype(str)

        tmp["canonical_smiles"] = (
            tmp["SMILES"].map(canonicalize)
        )

        tmp["connectivity_key"] = (
            tmp["SMILES"].map(connectivity_key)
        )

        result[split] = tmp

    return result


def prepare_external(
    name,
    path,
    hergat,
):

    print()
    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    df = pd.read_csv(path)

    required = {
        "SMILES",
        "pIC50",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path}: colonnes manquantes: {missing}"
        )

    original_rows = len(df)

    # Keep original information
    df["pIC50"] = pd.to_numeric(
        df["pIC50"],
        errors="coerce",
    )

    missing_pic50 = int(
        df["pIC50"].isna().sum()
    )

    if missing_pic50:
        raise ValueError(
            f"{name}: {missing_pic50} valeurs pIC50 invalides"
        )

    # IMPORTANT:
    # CToxPred published definition:
    #
    # pIC50 >= 5 => blocker
    # pIC50 < 5  => non-blocker
    #
    df["Class"] = (
        df["pIC50"] >= 5.0
    ).astype(int)

    # Molecular audit
    df["canonical_smiles"] = (
        df["SMILES"].map(canonicalize)
    )

    df["connectivity_key"] = (
        df["SMILES"].map(connectivity_key)
    )

    invalid_mask = (
        df["canonical_smiles"].isna()
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    if invalid_count:

        invalid_path = (
            OUT_DIR /
            f"{name}_invalid_smiles.csv"
        )

        df.loc[invalid_mask].to_csv(
            invalid_path,
            index=False,
        )

        print(
            f"WARNING: {invalid_count} invalid SMILES"
        )

    valid = df.loc[
        ~invalid_mask
    ].copy()

    # -------------------------------------------------------
    # Internal duplicates
    # -------------------------------------------------------

    duplicate_mask = (
        valid.duplicated(
            subset=["canonical_smiles"],
            keep=False,
        )
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    # We do NOT silently delete them here.
    if duplicate_rows:

        valid.loc[
            duplicate_mask
        ].to_csv(
            OUT_DIR /
            f"{name}_internal_duplicates.csv",
            index=False,
        )

    # -------------------------------------------------------
    # Overlap with hERGAT
    # -------------------------------------------------------

    all_hergat_canonical = set()

    all_hergat_connectivity = set()

    overlap_report = {}

    for split, ref in hergat.items():

        canonical_set = set(
            ref["canonical_smiles"].dropna()
        )

        connectivity_set = set(
            ref["connectivity_key"].dropna()
        )

        exact_mask = (
            valid["canonical_smiles"]
            .isin(canonical_set)
        )

        connectivity_mask = (
            valid["connectivity_key"]
            .isin(connectivity_set)
        )

        overlap_report[split] = {
            "canonical_smiles_overlap": int(
                exact_mask.sum()
            ),
            "connectivity_overlap": int(
                connectivity_mask.sum()
            ),
        }

        all_hergat_canonical.update(
            canonical_set
        )

        all_hergat_connectivity.update(
            connectivity_set
        )

    any_exact_overlap = (
        valid["canonical_smiles"]
        .isin(all_hergat_canonical)
    )

    any_connectivity_overlap = (
        valid["connectivity_key"]
        .isin(all_hergat_connectivity)
    )

    valid["overlap_hergat_exact"] = (
        any_exact_overlap.astype(int)
    )

    valid["overlap_hergat_connectivity"] = (
        any_connectivity_overlap.astype(int)
    )

    # -------------------------------------------------------
    # AS-PUBLISHED version
    # -------------------------------------------------------

    as_published = valid[
        [
            "SMILES",
            "Class",
        ]
    ].copy()

    as_published.to_csv(
        OUT_DIR /
        f"{name}_as_published.csv",
        index=False,
    )

    # -------------------------------------------------------
    # STRICT version
    #
    # Remove exact canonical structures occurring anywhere
    # in hERGAT Train / Val / Test.
    # Connectivity overlap is reported but not automatically
    # deleted.
    # -------------------------------------------------------

    strict = valid.loc[
        ~any_exact_overlap
    ].copy()

    strict_out = strict[
        [
            "SMILES",
            "Class",
        ]
    ].copy()

    strict_out.to_csv(
        OUT_DIR /
        f"{name}_strict_no_hergat_overlap.csv",
        index=False,
    )

    # Complete audited table
    valid.to_csv(
        OUT_DIR /
        f"{name}_audit_full.csv",
        index=False,
    )

    report = {
        "dataset": name,

        "source_file": str(path),

        "published_rows": int(
            original_rows
        ),

        "valid_smiles": int(
            len(valid)
        ),

        "invalid_smiles": int(
            invalid_count
        ),

        "internal_duplicate_rows": int(
            duplicate_rows
        ),

        "published_class_distribution": {
            "blocker_1": int(
                (valid["Class"] == 1).sum()
            ),
            "non_blocker_0": int(
                (valid["Class"] == 0).sum()
            ),
        },

        "label_rule": (
            "Class=1 if pIC50 >= 5.0 "
            "(IC50 <= 10 uM), else Class=0"
        ),

        "overlap_by_split": (
            overlap_report
        ),

        "any_hergat_exact_overlap": int(
            any_exact_overlap.sum()
        ),

        "any_hergat_connectivity_overlap": int(
            any_connectivity_overlap.sum()
        ),

        "strict_external_rows": int(
            len(strict)
        ),

        "strict_class_distribution": {
            "blocker_1": int(
                (strict["Class"] == 1).sum()
            ),
            "non_blocker_0": int(
                (strict["Class"] == 0).sum()
            ),
        },
    }

    with open(
        OUT_DIR /
        f"{name}_audit.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )


def main():

    hergat = prepare_hergat_reference()

    for name, path in EXTERNAL_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        prepare_external(
            name,
            path,
            hergat,
        )


if __name__ == "__main__":
    main()
    