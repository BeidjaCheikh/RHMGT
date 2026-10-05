#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Cai": {
        "hera": ROOT / "external_results/cai_train_clean/predictions.csv",
        "hergat": ROOT.parent / "hERGAT/external_results/cai_train_clean_original_labels/predictions.csv",
        "hera_threshold": 0.805,
        "hergat_threshold": 0.5,
    },

    "Karim": {
        "hera": ROOT / "external_results/karim_train_clean/predictions.csv",
        "hergat": ROOT.parent / "hERGAT/external_results/karim_train_clean_original_labels/predictions.csv",
        "hera_threshold": 0.805,
        "hergat_threshold": 0.5,
    },
}

OUT_DIR = ROOT / "external_results/statistical_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 5000
SEED = 42


def find_column(df, candidates):
    lower = {
        str(c).lower(): c
        for c in df.columns
    }

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def find_label(df):
    col = find_column(
        df,
        [
            "Class",
            "label",
            "y_true",
            "true_label",
            "target",
        ],
    )

    if col is None:
        raise ValueError(
            f"Cannot detect label column. Columns={list(df.columns)}"
        )

    return col


def find_smiles(df):
    return find_column(
        df,
        [
            "SMILES",
            "smiles",
            "canonical_smiles",
        ],
    )


def find_score(df):

    exact = find_column(
        df,
        [
            "prob_blocker",
            "blocker_probability",
            "probability",
            "prob",
            "score",
            "y_prob",
            "prob_1",
            "p1",
            "positive_probability",
        ],
    )

    if exact is not None:
        return exact

    candidates = []

    for c in df.columns:
        lc = str(c).lower()

        if (
            "prob" in lc
            or "score" in lc
        ):
            if (
                "nonblock" not in lc
                and "class_0" not in lc
                and "prob_0" not in lc
            ):
                if pd.api.types.is_numeric_dtype(df[c]):
                    candidates.append(c)

    if candidates:
        return candidates[0]

    raise ValueError(
        f"Cannot detect probability column. Columns={list(df.columns)}"
    )


def prepare(df, threshold, name):

    label_col = find_label(df)
    score_col = find_score(df)
    smiles_col = find_smiles(df)

    print("\n", name)
    print("label column :", label_col)
    print("score column :", score_col)
    print("smiles column:", smiles_col)

    out = pd.DataFrame()

    if smiles_col is not None:
        out["SMILES"] = df[smiles_col].astype(str)

    out["y"] = df[label_col].astype(int)
    out["score"] = df[score_col].astype(float)

    out["pred"] = (
        out["score"] >= threshold
    ).astype(int)

    return out


def metrics(y, score, pred):

    return {
        "ACC": accuracy_score(y, pred),
        "F1": f1_score(y, pred, zero_division=0),
        "AUROC": roc_auc_score(y, score),
        "AUPR": average_precision_score(y, score),
    }


def percentile_ci(values):

    arr = np.asarray(values, dtype=float)

    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def bootstrap_compare(
    y,
    score_hera,
    pred_hera,
    score_hergat,
    pred_hergat,
):

    rng = np.random.default_rng(SEED)

    n = len(y)

    boot_hera = {
        "ACC": [],
        "F1": [],
        "AUROC": [],
        "AUPR": [],
    }

    boot_hergat = {
        "ACC": [],
        "F1": [],
        "AUROC": [],
        "AUPR": [],
    }

    boot_delta = {
        "ACC": [],
        "F1": [],
        "AUROC": [],
        "AUPR": [],
    }

    successful = 0

    while successful < N_BOOT:

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yy = y[idx]

        # AUROC impossible si une seule classe
        if len(np.unique(yy)) < 2:
            continue

        mh = metrics(
            yy,
            score_hera[idx],
            pred_hera[idx],
        )

        mg = metrics(
            yy,
            score_hergat[idx],
            pred_hergat[idx],
        )

        for metric in boot_hera:

            boot_hera[metric].append(
                mh[metric]
            )

            boot_hergat[metric].append(
                mg[metric]
            )

            boot_delta[metric].append(
                mh[metric] - mg[metric]
            )

        successful += 1

    return (
        boot_hera,
        boot_hergat,
        boot_delta,
    )


summary_rows = []
delta_rows = []
mcnemar_rows = []


for dataset, cfg in DATASETS.items():

    print("\n" + "=" * 80)
    print(dataset.upper())
    print("=" * 80)

    hera_raw = pd.read_csv(
        cfg["hera"]
    )

    hergat_raw = pd.read_csv(
        cfg["hergat"]
    )

    hera = prepare(
        hera_raw,
        cfg["hera_threshold"],
        f"{dataset} HERA-HGT",
    )

    hergat = prepare(
        hergat_raw,
        cfg["hergat_threshold"],
        f"{dataset} hERGAT",
    )

    # --------------------------------------------------
    # Pair molecules
    # --------------------------------------------------

    if (
        "SMILES" in hera.columns
        and "SMILES" in hergat.columns
    ):

        merged = hera.merge(
            hergat,
            on="SMILES",
            suffixes=("_hera", "_hergat"),
            validate="one_to_one",
        )

        if len(merged) != len(hera):
            raise RuntimeError(
                f"{dataset}: common={len(merged)}, "
                f"HERA={len(hera)}, "
                f"hERGAT={len(hergat)}"
            )

        if not np.array_equal(
            merged["y_hera"].to_numpy(),
            merged["y_hergat"].to_numpy(),
        ):
            raise RuntimeError(
                f"{dataset}: label mismatch."
            )

        y = merged["y_hera"].to_numpy()

        score_hera = (
            merged["score_hera"].to_numpy()
        )

        pred_hera = (
            merged["pred_hera"].to_numpy()
        )

        score_hergat = (
            merged["score_hergat"].to_numpy()
        )

        pred_hergat = (
            merged["pred_hergat"].to_numpy()
        )

    else:

        if len(hera) != len(hergat):
            raise RuntimeError(
                f"{dataset}: different number of rows."
            )

        if not np.array_equal(
            hera["y"].to_numpy(),
            hergat["y"].to_numpy(),
        ):
            raise RuntimeError(
                f"{dataset}: row labels differ."
            )

        y = hera["y"].to_numpy()

        score_hera = hera["score"].to_numpy()
        pred_hera = hera["pred"].to_numpy()

        score_hergat = hergat["score"].to_numpy()
        pred_hergat = hergat["pred"].to_numpy()

    n = len(y)

    # --------------------------------------------------
    # Original metrics
    # --------------------------------------------------

    m_hera = metrics(
        y,
        score_hera,
        pred_hera,
    )

    m_hergat = metrics(
        y,
        score_hergat,
        pred_hergat,
    )

    # --------------------------------------------------
    # Bootstrap
    # --------------------------------------------------

    (
        boot_hera,
        boot_hergat,
        boot_delta,
    ) = bootstrap_compare(
        y,
        score_hera,
        pred_hera,
        score_hergat,
        pred_hergat,
    )

    for metric in [
        "ACC",
        "F1",
        "AUROC",
        "AUPR",
    ]:

        h_low, h_high = percentile_ci(
            boot_hera[metric]
        )

        g_low, g_high = percentile_ci(
            boot_hergat[metric]
        )

        d_low, d_high = percentile_ci(
            boot_delta[metric]
        )

        delta = (
            m_hera[metric]
            - m_hergat[metric]
        )

        summary_rows.append({
            "dataset": dataset,
            "n": n,
            "metric": metric,

            "HERA_HGT": m_hera[metric],
            "HERA_CI_low": h_low,
            "HERA_CI_high": h_high,

            "hERGAT": m_hergat[metric],
            "hERGAT_CI_low": g_low,
            "hERGAT_CI_high": g_high,
        })

        delta_rows.append({
            "dataset": dataset,
            "n": n,
            "metric": metric,
            "delta_HERA_minus_hERGAT": delta,
            "delta_CI_low": d_low,
            "delta_CI_high": d_high,
            "CI_excludes_zero": (
                d_low > 0 or d_high < 0
            ),
        })

    # --------------------------------------------------
    # McNemar exact test
    # --------------------------------------------------

    correct_hera = (
        pred_hera == y
    )

    correct_hergat = (
        pred_hergat == y
    )

    # HERA correct, hERGAT wrong
    b = int(
        np.sum(
            correct_hera
            & ~correct_hergat
        )
    )

    # HERA wrong, hERGAT correct
    c = int(
        np.sum(
            ~correct_hera
            & correct_hergat
        )
    )

    discordant = b + c

    if discordant > 0:

        pvalue = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue

    else:
        pvalue = 1.0

    mcnemar_rows.append({
        "dataset": dataset,
        "n": n,
        "HERA_correct_hERGAT_wrong": b,
        "HERA_wrong_hERGAT_correct": c,
        "discordant_pairs": discordant,
        "mcnemar_exact_p": pvalue,
    })

    print("\nRESULTS")

    print(
        f"HERA-HGT : "
        f"ACC={m_hera['ACC']:.6f} "
        f"F1={m_hera['F1']:.6f} "
        f"AUROC={m_hera['AUROC']:.6f} "
        f"AUPR={m_hera['AUPR']:.6f}"
    )

    print(
        f"hERGAT   : "
        f"ACC={m_hergat['ACC']:.6f} "
        f"F1={m_hergat['F1']:.6f} "
        f"AUROC={m_hergat['AUROC']:.6f} "
        f"AUPR={m_hergat['AUPR']:.6f}"
    )


summary_df = pd.DataFrame(
    summary_rows
)

delta_df = pd.DataFrame(
    delta_rows
)

mcnemar_df = pd.DataFrame(
    mcnemar_rows
)


summary_df.to_csv(
    OUT_DIR / "bootstrap_metrics.csv",
    index=False,
)

delta_df.to_csv(
    OUT_DIR / "bootstrap_differences.csv",
    index=False,
)

mcnemar_df.to_csv(
    OUT_DIR / "mcnemar.csv",
    index=False,
)


print("\n" + "=" * 80)
print("BOOTSTRAP METRICS")
print("=" * 80)
print(
    summary_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("HERA-HGT - hERGAT")
print("=" * 80)
print(
    delta_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("MCNEMAR")
print("=" * 80)
print(
    mcnemar_df.to_string(
        index=False
    )
)

print("\nSaved:")
print(
    OUT_DIR / "bootstrap_metrics.csv"
)
print(
    OUT_DIR / "bootstrap_differences.csv"
)
print(
    OUT_DIR / "mcnemar.csv"
)
