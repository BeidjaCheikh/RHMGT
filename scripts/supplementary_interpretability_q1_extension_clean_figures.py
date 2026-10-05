#!/usr/bin/env python3
"""
Additional supplementary interpretability figures for RHMGT.

This script is an extension of the final RHMGT interpretability pipeline.
It DOES NOT retrain the model and DOES NOT redefine the primary explanation
methods. It reads the molecule-level CSVs already produced by:

    interpretability_q1_journal_final_Q1_complete_v2.py

and generates two non-redundant supplementary figures:

Figure S4
---------
Global atom-type signed Integrated-Gradients summary.

Why it is useful:
- the main paper already provides global motif, pharmacophore, and PIG-relation
  summaries, but no global atom-level summary;
- this figure closes that gap without changing the explanation method;
- aggregation is performed at the molecule level to avoid allowing molecules
  with many repeated atoms of one type to dominate the result.

Figure S5
---------
Molecule-level distributions of motif-family occlusion effects.

Why it is useful:
- the main paper reports mean family effects with bootstrap confidence
  intervals;
- this supplementary figure exposes the full molecule-to-molecule
  heterogeneity behind those means;
- no additional significance claim is introduced.

Expected existing files in --interpretability-dir
--------------------------------------------------
global_atom_ig_sample.csv
global_family_occlusion.csv

The first file is already produced by the final pipeline from the global
atom-IG sample (default: 30 TP + 30 TN, 24 IG steps unless changed).
The second file is produced by the global motif/family occlusion pass.

Outputs
-------
supplementary_extended/
    Fig_S4_Global_Atom_Type_Attribution.png
    Fig_S4_Global_Atom_Type_Attribution.pdf
    Fig_S4_Global_Atom_Type_Attribution.svg
    Fig_S5_Motif_Family_Effect_Distributions.png
    Fig_S5_Motif_Family_Effect_Distributions.pdf
    Fig_S5_Motif_Family_Effect_Distributions.svg
    S4_atom_type_summary.csv
    S4_atom_type_contrasts.csv
    S5_family_distribution_summary.csv
    metadata.txt

Usage
-----
python scripts/supplementary_interpretability_q1_extension.py \
  --interpretability-dir outputs/interpretability_q1 \
  --bootstrap-reps 2000 \
  --seed 2026 \
  --dpi 600

Scientific scope
----------------
These plots describe predictive sensitivity of the trained RHMGT model.
They must not be interpreted as experimental evidence of physical ligand-hERG
contacts or as proof of a causal biochemical mechanism.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FAMILY_ORDER = ["STR", "BN", "AR", "NHC", "PF"]
CASE_ORDER = ["TP", "TN"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate additional Q1 supplementary interpretability figures "
            "from cached RHMGT interpretability CSVs."
        )
    )
    p.add_argument(
        "--interpretability-dir",
        required=True,
        help="Output directory created by the final RHMGT interpretability script.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Destination directory. Default: "
            "<interpretability-dir>/supplementary_extended"
        ),
    )
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument(
        "--top-atom-types",
        type=int,
        default=8,
        help="Maximum number of globally supported atom types shown in Fig. S4.",
    )
    p.add_argument(
        "--min-molecules-per-case",
        type=int,
        default=10,
        help=(
            "Minimum number of distinct molecules required in both TP and TN "
            "for an atom type to be displayed in Fig. S4."
        ),
    )
    return p.parse_args()


def require_columns(df: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def bootstrap_mean_ci(
    values: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan

    mean = float(np.mean(values))
    if values.size == 1:
        return mean, mean, mean

    n = values.size
    boot = np.empty(reps, dtype=float)
    for b in range(reps):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(values[idx]))

    alpha = 1.0 - ci
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return mean, float(lo), float(hi)


def bootstrap_unpaired_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    reps: int,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Descriptive molecule-level TP-minus-TN mean difference.

    TP and TN contain different molecules, so this is an unpaired bootstrap.
    It is intentionally reported as a confidence interval, not a p-value.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return np.nan, np.nan, np.nan

    delta = float(np.mean(a) - np.mean(b))
    if a.size == 1 and b.size == 1:
        return delta, delta, delta

    boot = np.empty(reps, dtype=float)
    for i in range(reps):
        aa = a[rng.integers(0, a.size, size=a.size)]
        bb = b[rng.integers(0, b.size, size=b.size)]
        boot[i] = float(np.mean(aa) - np.mean(bb))

    alpha = 1.0 - ci
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return delta, float(lo), float(hi)


def save_figure(fig: plt.Figure, base_path: Path, dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")


# =============================================================================
# Figure S4: global atom-type attribution
# =============================================================================

def atom_type_label(symbol: str, is_aromatic: int) -> str:
    symbol = str(symbol)
    return f"{symbol} (arom.)" if int(is_aromatic) == 1 else symbol


def prepare_atom_type_molecule_level(
    atom_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert atom-level rows into one row per molecule x atom type.

    We average repeated atoms of the same type within a molecule first. This
    prevents a molecule containing many carbons from contributing more weight
    than a molecule containing only one atom of another type.
    """
    df = atom_df.copy()
    df = df[df["case"].isin(CASE_ORDER)].copy()
    df["is_aromatic"] = pd.to_numeric(df["is_aromatic"], errors="coerce").fillna(0).astype(int)
    df["ig_score_blocker_logit"] = pd.to_numeric(
        df["ig_score_blocker_logit"], errors="coerce"
    )
    df = df[np.isfinite(df["ig_score_blocker_logit"])].copy()

    df["atom_type"] = [
        atom_type_label(s, a)
        for s, a in zip(df["symbol"].astype(str), df["is_aromatic"].astype(int))
    ]
    df["abs_ig"] = df["ig_score_blocker_logit"].abs()

    mol = (
        df.groupby(["row_id", "case", "atom_type"], as_index=False)
        .agg(
            molecule_mean_signed_ig=("ig_score_blocker_logit", "mean"),
            molecule_mean_abs_ig=("abs_ig", "mean"),
            n_atoms_of_type=("atom_idx", "size"),
        )
    )
    return mol


def supported_atom_types(
    mol: pd.DataFrame,
    min_molecules_per_case: int,
    top_k: int,
) -> List[str]:
    counts = (
        mol.groupby(["atom_type", "case"])["row_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    for case in CASE_ORDER:
        if case not in counts.columns:
            counts[case] = 0

    supported = counts[
        (counts["TP"] >= min_molecules_per_case)
        & (counts["TN"] >= min_molecules_per_case)
    ].index.tolist()

    if not supported:
        raise RuntimeError(
            "No atom type satisfies the minimum support requirement in both TP and TN. "
            "Lower --min-molecules-per-case only if scientifically justified."
        )

    rank_df = mol[mol["atom_type"].isin(supported)].copy()
    ranking = (
        rank_df.groupby("atom_type")["molecule_mean_abs_ig"]
        .mean()
        .sort_values(ascending=False)
    )

    return ranking.head(top_k).index.tolist()


def summarize_atom_types(
    mol: pd.DataFrame,
    atom_types: Sequence[str],
    reps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict] = []
    contrast_rows: List[Dict] = []

    for ai, atom_type in enumerate(atom_types):
        per_case: Dict[str, np.ndarray] = {}

        for ci, case in enumerate(CASE_ORDER):
            g = mol[
                (mol["atom_type"] == atom_type)
                & (mol["case"] == case)
            ]["molecule_mean_signed_ig"].to_numpy(float)

            per_case[case] = g
            rng = np.random.default_rng(seed + 1009 * ai + 97 * ci)
            mean, lo, hi = bootstrap_mean_ci(g, reps, rng)

            summary_rows.append(
                {
                    "atom_type": atom_type,
                    "case": case,
                    "n_molecules": int(len(g)),
                    "mean_signed_ig": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

        rng = np.random.default_rng(seed + 70001 + 313 * ai)
        delta, lo, hi = bootstrap_unpaired_difference_ci(
            per_case["TP"], per_case["TN"], reps, rng
        )

        contrast_rows.append(
            {
                "atom_type": atom_type,
                "n_tp": int(len(per_case["TP"])),
                "n_tn": int(len(per_case["TN"])),
                "tp_minus_tn_mean_signed_ig": delta,
                "ci_low": lo,
                "ci_high": hi,
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows)


def draw_s4(
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    out_base: Path,
    dpi: int,
) -> None:
    """Draw Fig. S4 without an embedded figure title or caption text.

    Only panel identifiers, axes, legend, confidence intervals, and sample
    support are kept inside the artwork. The scientific explanation belongs
    in the manuscript caption.
    """
    atom_types = contrasts["atom_type"].tolist()
    y = np.arange(len(atom_types), dtype=float)

    # Sized close to the final supplementary-page width so labels remain
    # readable after LaTeX scaling.
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2))

    # Panel (a): TP/TN mean signed IG
    ax = axes[0]
    offset = 0.17

    for case, dy, label in [
        ("TP", -offset, "True blocker (TP)"),
        ("TN",  offset, "True non-blocker (TN)"),
    ]:
        g = (
            summary[summary["case"] == case]
            .set_index("atom_type")
            .loc[atom_types]
            .reset_index()
        )
        mean = g["mean_signed_ig"].to_numpy(float)
        lo = g["ci_low"].to_numpy(float)
        hi = g["ci_high"].to_numpy(float)
        xerr = np.vstack([
            np.maximum(0.0, mean - lo),
            np.maximum(0.0, hi - mean),
        ])
        ax.errorbar(
            mean,
            y + dy,
            xerr=xerr,
            fmt="o",
            capsize=4.0,
            markersize=5.5,
            linewidth=1.2,
            label=label,
        )

    ax.axvline(0.0, linewidth=0.9)
    ax.set_yticks(y, atom_types, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean signed atom IG (blocker logit)", fontsize=10.5)
    ax.set_title("(a)", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9.5)
    ax.legend(frameon=False, fontsize=9.2)

    # Panel (b): TP - TN contrast
    ax = axes[1]
    delta = contrasts["tp_minus_tn_mean_signed_ig"].to_numpy(float)
    lo = contrasts["ci_low"].to_numpy(float)
    hi = contrasts["ci_high"].to_numpy(float)
    xerr = np.vstack([
        np.maximum(0.0, delta - lo),
        np.maximum(0.0, hi - delta),
    ])

    ax.errorbar(
        delta, y, xerr=xerr, fmt="o", capsize=4.0,
        markersize=5.5, linewidth=1.2,
    )
    ax.axvline(0.0, linewidth=0.9)
    ax.set_yticks(y, atom_types, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("TP - TN mean signed IG", fontsize=10.5)
    ax.set_title("(b)", loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9.5)

    # Keep support counts as compact data annotations, not explanatory prose.
    for yi, row in contrasts.reset_index(drop=True).iterrows():
        ax.text(
            0.98,
            yi,
            f"nTP={int(row['n_tp'])}, nTN={int(row['n_tn'])}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="right",
            fontsize=8.5,
        )

    fig.tight_layout(pad=1.0, w_pad=1.8)
    save_figure(fig, out_base, dpi)
    plt.close(fig)


# =============================================================================
# Figure S5: distribution of motif-family effects
# =============================================================================

def prepare_family_data(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x = x[x["case"].isin(CASE_ORDER)].copy()
    x["logit_contribution"] = pd.to_numeric(
        x["logit_contribution"], errors="coerce"
    )
    x = x[np.isfinite(x["logit_contribution"])].copy()

    if "motif_family_short" not in x.columns:
        raise ValueError(
            "global_family_occlusion.csv must contain 'motif_family_short'."
        )

    # There should normally be one row per molecule/family. If not, aggregate
    # to molecule level so each molecule carries equal weight.
    x = (
        x.groupby(
            ["row_id", "case", "motif_family_short"],
            as_index=False,
        )
        .agg(logit_contribution=("logit_contribution", "mean"))
    )
    return x


def summarize_family_distributions(x: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for case in CASE_ORDER:
        for family in FAMILY_ORDER:
            v = x[
                (x["case"] == case)
                & (x["motif_family_short"] == family)
            ]["logit_contribution"].to_numpy(float)

            if v.size == 0:
                continue

            rows.append(
                {
                    "case": case,
                    "family": family,
                    "n_molecules": int(v.size),
                    "mean": float(np.mean(v)),
                    "median": float(np.median(v)),
                    "q25": float(np.quantile(v, 0.25)),
                    "q75": float(np.quantile(v, 0.75)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
            )
    return pd.DataFrame(rows)


def draw_s5(
    family_df: pd.DataFrame,
    out_base: Path,
    dpi: int,
) -> None:
    """Draw Fig. S5 with publication-essential labels only.

    The overall title and methodological explanation are intentionally omitted
    from the image and should be supplied in the supplementary caption.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), sharey=True)

    for panel_idx, (ax, case) in enumerate(zip(axes, CASE_ORDER)):
        data: List[np.ndarray] = []
        labels: List[str] = []

        for family in FAMILY_ORDER:
            vals = family_df[
                (family_df["case"] == case)
                & (family_df["motif_family_short"] == family)
            ]["logit_contribution"].to_numpy(float)

            if vals.size == 0:
                continue

            data.append(vals)
            labels.append(f"{family}\n(n={vals.size})")

        if not data:
            ax.text(0.5, 0.5, f"No {case} family data", ha="center", va="center", fontsize=10)
            ax.axis("off")
            continue

        ax.boxplot(
            data,
            tick_labels=labels,
            showfliers=False,
            whis=(5, 95),
        )
        ax.axhline(0.0, linewidth=0.9)
        ax.set_xlabel("Motif family", fontsize=10.5)
        panel = "(a) TP" if panel_idx == 0 else "(b) TN"
        ax.set_title(panel, loc="left", fontsize=12, fontweight="bold")
        ax.tick_params(axis="both", labelsize=9.5)

    axes[0].set_ylabel("Family-occlusion Δlogit", fontsize=10.5)

    fig.tight_layout(pad=1.0, w_pad=1.8)
    save_figure(fig, out_base, dpi)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    if args.bootstrap_reps < 500:
        raise ValueError("--bootstrap-reps should be at least 500; 2000 is recommended.")
    if args.top_atom_types < 1:
        raise ValueError("--top-atom-types must be >= 1.")
    if args.min_molecules_per_case < 1:
        raise ValueError("--min-molecules-per-case must be >= 1.")

    base = Path(args.interpretability_dir)
    if not base.exists():
        raise FileNotFoundError(f"Interpretability directory not found: {base}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else base / "supplementary_extended"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    atom_path = base / "global_atom_ig_sample.csv"
    family_path = base / "global_family_occlusion.csv"

    if not atom_path.exists():
        raise FileNotFoundError(
            f"Missing {atom_path}\n"
            "Run the final interpretability pipeline without --skip-global first."
        )
    if not family_path.exists():
        raise FileNotFoundError(
            f"Missing {family_path}\n"
            "Run the final interpretability pipeline without --skip-global first."
        )

    # -------------------------------------------------------------------------
    # S4
    # -------------------------------------------------------------------------
    atom_df = pd.read_csv(atom_path)
    require_columns(
        atom_df,
        [
            "row_id",
            "case",
            "atom_idx",
            "symbol",
            "is_aromatic",
            "ig_score_blocker_logit",
        ],
        atom_path,
    )

    atom_mol = prepare_atom_type_molecule_level(atom_df)
    selected_types = supported_atom_types(
        atom_mol,
        args.min_molecules_per_case,
        args.top_atom_types,
    )
    atom_summary, atom_contrasts = summarize_atom_types(
        atom_mol,
        selected_types,
        args.bootstrap_reps,
        args.seed,
    )

    atom_summary.to_csv(out_dir / "S4_atom_type_summary.csv", index=False)
    atom_contrasts.to_csv(out_dir / "S4_atom_type_contrasts.csv", index=False)

    draw_s4(
        atom_summary,
        atom_contrasts,
        out_dir / "Fig_S4_Global_Atom_Type_Attribution",
        args.dpi,
    )

    # -------------------------------------------------------------------------
    # S5
    # -------------------------------------------------------------------------
    family_df = pd.read_csv(family_path)
    require_columns(
        family_df,
        [
            "row_id",
            "case",
            "motif_family_short",
            "logit_contribution",
        ],
        family_path,
    )

    family_mol = prepare_family_data(family_df)
    family_summary = summarize_family_distributions(family_mol)
    family_summary.to_csv(
        out_dir / "S5_family_distribution_summary.csv",
        index=False,
    )

    draw_s5(
        family_mol,
        out_dir / "Fig_S5_Motif_Family_Effect_Distributions",
        args.dpi,
    )

    metadata = f"""RHMGT additional supplementary interpretability figures
================================================================
Input directory: {base}
Seed: {args.seed}
Bootstrap replicates: {args.bootstrap_reps}

Figure S4
---------
Source: global_atom_ig_sample.csv
Aggregation:
  1. atom-level signed IG;
  2. average repeated atoms within each molecule x atom type;
  3. molecule-level bootstrap 95% CI.
Atom types require >= {args.min_molecules_per_case} distinct TP molecules
and >= {args.min_molecules_per_case} distinct TN molecules.
Displayed maximum atom types: {args.top_atom_types}.
No hypothesis test is attached to the TP-TN contrast.

Figure S5
---------
Source: global_family_occlusion.csv
Aggregation: one molecule-level logit contribution per family.
Boxplot whiskers: 5th-95th percentiles.
The figure is descriptive and complements the mean family-effect figure
in the main manuscript.

Interpretation boundary
-----------------------
Both figures characterize predictive sensitivity of RHMGT.
They do not establish causal ligand-channel interactions.
"""
    (out_dir / "metadata.txt").write_text(metadata, encoding="utf-8")

    print("\nAdditional supplementary interpretability completed.")
    print(f"Output directory: {out_dir}")
    print("Generated:")
    for p in sorted(out_dir.glob("Fig_S*")):
        print(f"  {p}")
    print("CSV summaries:")
    for p in sorted(out_dir.glob("S*.csv")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
