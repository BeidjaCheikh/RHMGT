#!/usr/bin/env python

from pathlib import Path
import json
import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]

HERGAT_REPO = ROOT.parent / "hERGAT"

SOURCE_FILES = {
    "cai": (
        HERGAT_REPO /
        "recovered_external" /
        "dataset_External_Ex1.csv"
    ),
    "karim": (
        HERGAT_REPO /
        "recovered_external" /
        "dataset_External_Ex3.csv"
    ),
}

TRAIN_FILE = (
    ROOT /
    "data" /
    "paper_split" /
    "hERGAT_train_df.csv"
)

OUT_DIR = (
    ROOT /
    "data" /
    "external" /
    "hergat_reconstructed"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


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


def load_train():

    df = pd.read_csv(TRAIN_FILE)

    if "SMILES" not in df.columns:
        raise ValueError(
            "SMILES missing from hERGAT TRAIN."
        )

    df["canonical_smiles"] = (
        df["SMILES"].map(canonicalize)
    )

    return set(
        df["canonical_smiles"].dropna()
    )


def process_dataset(name, path, train_set):

    print("\n" + "=" * 80)
    print(name.upper())
    print("=" * 80)

    df = pd.read_csv(path)

    required = {"SMILES", "Class"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path}: required columns={required}"
        )

    original_n = len(df)

    df["canonical_smiles"] = (
        df["SMILES"].map(canonicalize)
    )

    invalid_mask = (
        df["canonical_smiles"].isna()
    )

    invalid_n = int(
        invalid_mask.sum()
    )

    if invalid_n:
        df.loc[
            invalid_mask
        ].to_csv(
            OUT_DIR /
            f"{name}_invalid_smiles.csv",
            index=False,
        )

    df = df.loc[
        ~invalid_mask
    ].copy()

    # --------------------------------------------------
    # Analyse duplicate canonical structures
    # --------------------------------------------------

    grouped = (
        df.groupby("canonical_smiles")["Class"]
        .agg(["size", "nunique"])
        .reset_index()
    )

    duplicate_groups = grouped[
        grouped["size"] > 1
    ]

    conflicting_groups = grouped[
        grouped["nunique"] > 1
    ]

    conflicting_canonicals = set(
        conflicting_groups[
            "canonical_smiles"
        ]
    )

    # Save ambiguous rows
    ambiguous_df = df[
        df["canonical_smiles"].isin(
            conflicting_canonicals
        )
    ].copy()

    ambiguous_df.to_csv(
        OUT_DIR /
        f"{name}_conflicting_duplicates_removed.csv",
        index=False,
    )

    # --------------------------------------------------
    # Remove ALL structures with conflicting labels
    # --------------------------------------------------

    clean = df.loc[
        ~df["canonical_smiles"].isin(
            conflicting_canonicals
        )
    ].copy()

    # --------------------------------------------------
    # Same-label duplicates:
    # keep exactly one canonical structure
    # --------------------------------------------------

    clean = (
        clean
        .sort_values(
            ["canonical_smiles", "SMILES"]
        )
        .drop_duplicates(
            subset=["canonical_smiles"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    after_internal_dedup_n = len(clean)

    # --------------------------------------------------
    # HISTORICAL external label orientation
    #
    # Ex1 counts:
    # old Class 0 = published blocker count
    #
    # Ex3 counts:
    # old Class 0 = published blocker count
    #
    # Therefore:
    # old 0 -> blocker -> new 1
    # old 1 -> nonblocker -> new 0
    # --------------------------------------------------

    clean["historical_Class"] = (
        clean["Class"].astype(int)
    )

    clean["Class"] = (
        1 - clean["historical_Class"]
    ).astype(int)

    # --------------------------------------------------
    # TRAIN overlap ONLY
    # --------------------------------------------------

    overlap_train = (
        clean["canonical_smiles"]
        .isin(train_set)
    )

    overlap_df = clean.loc[
        overlap_train
    ].copy()

    overlap_df.to_csv(
        OUT_DIR /
        f"{name}_train_overlap_removed.csv",
        index=False,
    )

    final_df = clean.loc[
        ~overlap_train
    ].copy()

    # --------------------------------------------------
    # Final evaluation CSV
    # --------------------------------------------------

    final_csv = (
        OUT_DIR /
        f"{name}_hergat_style_train_clean.csv"
    )

    final_df[
        ["SMILES", "Class"]
    ].to_csv(
        final_csv,
        index=False,
    )

    # Rich audit table
    final_df.to_csv(
        OUT_DIR /
        f"{name}_hergat_style_train_clean_audit_rows.csv",
        index=False,
    )

    # --------------------------------------------------
    # Audit
    # --------------------------------------------------

    audit = {
        "dataset": name,

        "source_file": str(path),

        "original_rows": int(
            original_n
        ),

        "invalid_smiles": int(
            invalid_n
        ),

        "unique_canonical_before_conflict_removal": int(
            df["canonical_smiles"].nunique()
        ),

        "duplicate_canonical_groups": int(
            len(duplicate_groups)
        ),

        "conflicting_canonical_groups_removed": int(
            len(conflicting_groups)
        ),

        "rows_in_conflicting_groups_removed": int(
            len(ambiguous_df)
        ),

        "rows_after_internal_dedup_and_conflict_removal": int(
            after_internal_dedup_n
        ),

        "train_exact_overlap_removed": int(
            overlap_train.sum()
        ),

        "final_n": int(
            len(final_df)
        ),

        "final_blockers_Class1": int(
            (final_df["Class"] == 1).sum()
        ),

        "final_nonblockers_Class0": int(
            (final_df["Class"] == 0).sum()
        ),

        "label_mapping": (
            "historical Class 0 -> blocker -> new Class 1; "
            "historical Class 1 -> non-blocker -> new Class 0"
        ),

        "filter_rule": (
            "Canonicalize SMILES; remove ambiguous canonical "
            "structures with conflicting labels; collapse same-label "
            "canonical duplicates; remove exact canonical overlap "
            "with hERGAT TRAIN only. Validation and test overlaps retained."
        ),
    }

    with open(
        OUT_DIR /
        f"{name}_hergat_style_audit.json",
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

    print(
        "\nFinal CSV:",
        final_csv,
    )


def main():

    train_set = load_train()

    print(
        "hERGAT TRAIN unique canonical:",
        len(train_set),
    )

    for name, path in SOURCE_FILES.items():

        if not path.exists():
            raise FileNotFoundError(path)

        process_dataset(
            name,
            path,
            train_set,
        )


if __name__ == "__main__":
    main()
