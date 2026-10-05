from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .chemistry import canonicalize_smiles, descriptors_raw_10, mol_from_smiles
from .config import FeaturizerConfig
from .hierarchy import featurize_hierarchical_smiles

FEATURE_SCHEMA_VERSION = "hera_hgt_v1_hergat39_bond10_morgan1024r3_fp3_desc10_train_scaled"


@dataclass(frozen=True)
class Record:
    smiles: str
    label: int
    row_id: int


@dataclass
class DescriptorScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return ((x - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: Dict) -> "DescriptorScaler":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float32),
            scale=np.asarray(d["scale"], dtype=np.float32),
        )


@dataclass
class LoadReport:
    path: str
    n_raw: int
    n_valid: int
    n_invalid_smiles: int
    n_invalid_labels: int
    invalid_row_ids: List[int]

    def to_dict(self) -> Dict:
        return asdict(self)


def load_records_csv_with_report(
    path: str | Path,
    smiles_col: str = "SMILES",
    label_col: str = "Class",
) -> Tuple[List[Record], LoadReport]:
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column {smiles_col!r}; columns={list(df.columns)[:20]}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column {label_col!r}")

    records: List[Record] = []
    invalid_smiles = 0
    invalid_labels = 0
    invalid_ids: List[int] = []
    canonical_col = "cano_smiles" if "cano_smiles" in df.columns else None
    for row_id, row in df.iterrows():
        smi = str(row[smiles_col]).strip()
        # The bundled hERGAT dataset already contains the exact canonical SMILES
        # used to create paper_run1. Prefer it to avoid RDKit-version-dependent
        # recanonicalization changes; still validate that RDKit can parse it.
        if canonical_col is not None and pd.notna(row[canonical_col]):
            candidate = str(row[canonical_col]).strip()
            can = candidate if mol_from_smiles(candidate) is not None else None
        else:
            can = canonicalize_smiles(smi)
        if can is None:
            invalid_smiles += 1
            invalid_ids.append(int(row_id))
            continue
        try:
            y = int(row[label_col])
        except Exception:
            invalid_labels += 1
            invalid_ids.append(int(row_id))
            continue
        if y not in (0, 1):
            invalid_labels += 1
            invalid_ids.append(int(row_id))
            continue
        records.append(Record(can, y, int(row_id)))

    if not records:
        raise ValueError(f"No valid records loaded from {path}")
    report = LoadReport(
        path=str(path),
        n_raw=int(len(df)),
        n_valid=int(len(records)),
        n_invalid_smiles=int(invalid_smiles),
        n_invalid_labels=int(invalid_labels),
        invalid_row_ids=invalid_ids,
    )
    return records, report


def load_records_csv(path: str | Path, smiles_col: str = "SMILES", label_col: str = "Class") -> List[Record]:
    return load_records_csv_with_report(path, smiles_col, label_col)[0]


def records_from_row_ids(records: Sequence[Record], row_ids: Sequence[int]) -> List[Record]:
    by_id = {int(r.row_id): r for r in records}
    missing = [int(i) for i in row_ids if int(i) not in by_id]
    if missing:
        raise ValueError(f"Split references {len(missing)} unavailable rows; first missing={missing[:5]}")
    return [by_id[int(i)] for i in row_ids]


def hergat_exact_split(
    records: Sequence[Record],
    split_json: str | Path,
    validate_smiles: bool = True,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Reuse the exact paper_run1 row split from the supplied hERGAT repository.

    This mode intentionally reproduces the hERGAT benchmark, including any
    canonical-SMILES overlap present in that original row-level split. Use the
    grouped/scaffold modes for leakage-controlled robustness experiments.
    """
    payload = json.loads(Path(split_json).read_text(encoding="utf-8"))
    out = []
    for name in ("train", "val", "test"):
        idx = payload[f"{name}_indices"]
        part = records_from_row_ids(records, idx)
        expected = payload.get(f"{name}_smiles")
        if validate_smiles and expected is not None:
            if len(expected) != len(part):
                raise ValueError(f"{name}: split SMILES length mismatch")
            mismatches = [i for i, (r, e) in enumerate(zip(part, expected)) if r.smiles != str(e)]
            if mismatches:
                i = mismatches[0]
                raise ValueError(
                    f"{name}: hERGAT split does not match this CSV/order at position {i}: "
                    f"dataset={part[i].smiles!r}, split={expected[i]!r}"
                )
        out.append(part)
    return tuple(out)  # type: ignore[return-value]


def grouped_random_split(
    records: Sequence[Record],
    seed: int = 42,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Random split at canonical-SMILES group level; duplicates cannot leak."""
    groups: Dict[str, List[Record]] = {}
    for r in records:
        groups.setdefault(r.smiles, []).append(r)
    keys = sorted(groups)
    # Majority label only drives stratification; original row labels are preserved.
    labels = np.asarray([
        int(sum(r.label for r in groups[k]) >= len(groups[k]) / 2.0) for k in keys
    ])
    test_frac = 1.0 - frac_train
    try:
        k_train, k_rest, y_train, y_rest = train_test_split(
            keys, labels, test_size=test_frac, random_state=seed, stratify=labels
        )
        relative_test = (1.0 - frac_train - frac_val) / test_frac
        k_val, k_test = train_test_split(
            k_rest, test_size=relative_test, random_state=seed, stratify=y_rest
        )
    except ValueError:
        # Small synthetic/smoke datasets may not support stratification.
        k_train, k_rest = train_test_split(keys, test_size=test_frac, random_state=seed)
        relative_test = (1.0 - frac_train - frac_val) / test_frac
        k_val, k_test = train_test_split(k_rest, test_size=relative_test, random_state=seed)

    def flatten(group_keys: Sequence[str]) -> List[Record]:
        return [r for k in group_keys for r in groups[k]]

    return flatten(k_train), flatten(k_val), flatten(k_test)


def clean_unique_records(
    records: Sequence[Record],
) -> Tuple[List[Record], Dict]:
    """Return one unambiguous record per canonical SMILES.

    Cleaning policy used for the strict HERA-HGT experiments:

    1. Group rows by canonical SMILES.
    2. If every row in a group has the same binary label, keep exactly one
       representative row (the smallest original ``row_id``).
    3. If a canonical SMILES is associated with both labels 0 and 1, remove
       the whole molecule from the strict benchmark.

    No majority voting is used. This makes the clean benchmark deterministic
    and prevents ambiguous compounds from being silently relabelled.
    """
    groups: Dict[str, List[Record]] = {}
    for r in records:
        groups.setdefault(r.smiles, []).append(r)

    clean: List[Record] = []
    conflicting_smiles: List[str] = []
    conflicting_rows_removed = 0
    duplicate_rows_removed = 0

    for smiles in sorted(groups):
        group = groups[smiles]
        labels = {int(r.label) for r in group}

        if len(labels) > 1:
            conflicting_smiles.append(smiles)
            conflicting_rows_removed += len(group)
            continue

        representative = min(group, key=lambda r: int(r.row_id))
        clean.append(
            Record(
                smiles=representative.smiles,
                label=int(representative.label),
                row_id=int(representative.row_id),
            )
        )
        duplicate_rows_removed += len(group) - 1

    clean.sort(key=lambda r: int(r.row_id))
    labels = [int(r.label) for r in clean]

    report = {
        "input_rows": int(len(records)),
        "input_unique_canonical_smiles": int(len(groups)),
        "conflicting_canonical_smiles_removed": int(len(conflicting_smiles)),
        "conflicting_rows_removed": int(conflicting_rows_removed),
        "duplicate_rows_removed_from_consistent_groups": int(duplicate_rows_removed),
        "final_unique_compounds": int(len(clean)),
        "final_positive": int(sum(labels)),
        "final_negative": int(len(labels) - sum(labels)),
        "final_positive_fraction": float(np.mean(labels)) if labels else float("nan"),
        "conflicting_examples": conflicting_smiles[:20],
    }
    return clean, report


def clean_random_split(
    records: Sequence[Record],
    seed: int = 2026,
    frac_train: float = 0.8,
    frac_val: float = 0.1,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Stratified 80/10/10 split for an already-clean unique dataset.

    The function validates that every canonical SMILES occurs exactly once and
    that labels are therefore unambiguous before splitting.
    """
    records = list(records)
    if not records:
        raise ValueError("clean_random_split received an empty dataset")

    test_frac = 1.0 - float(frac_train) - float(frac_val)
    if frac_train <= 0 or frac_val <= 0 or test_frac <= 0:
        raise ValueError("Split fractions must all be positive")
    if not np.isclose(frac_train + frac_val + test_frac, 1.0):
        raise ValueError("Split fractions must sum to 1")

    smiles = [r.smiles for r in records]
    if len(smiles) != len(set(smiles)):
        raise ValueError(
            "clean_random_split requires one row per canonical SMILES; "
            "call clean_unique_records() first"
        )

    indices = np.arange(len(records))
    labels = np.asarray([int(r.label) for r in records], dtype=np.int64)

    holdout_frac = 1.0 - float(frac_train)
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=holdout_frac,
        random_state=seed,
        stratify=labels,
    )

    holdout_labels = labels[holdout_idx]
    relative_test = test_frac / holdout_frac
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=relative_test,
        random_state=seed,
        stratify=holdout_labels,
    )

    return (
        [records[int(i)] for i in train_idx],
        [records[int(i)] for i in val_idx],
        [records[int(i)] for i in test_idx],
    )


def clean_scaffold_split(
    records: Sequence[Record],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Bemis-Murcko scaffold split for an already-clean unique dataset.

    Exact-molecule duplicates and conflicting labels must be removed before
    calling this function.
    """
    records = list(records)
    smiles = [r.smiles for r in records]
    if len(smiles) != len(set(smiles)):
        raise ValueError(
            "clean_scaffold_split requires one row per canonical SMILES; "
            "call clean_unique_records() first"
        )
    return scaffold_split(records, frac_train=frac_train, frac_val=frac_val)


def _scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        from rdkit import Chem
        return Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True)
    except Exception:
        return ""


def scaffold_split(
    records: Sequence[Record],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
) -> Tuple[List[Record], List[Record], List[Record]]:
    """Deterministic Bemis-Murcko scaffold split; duplicate molecules stay together."""
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(records):
        groups.setdefault(_scaffold(r.smiles), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    n = len(records)
    train_target = int(round(n * frac_train))
    val_target = int(round(n * frac_val))
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for group in ordered:
        if len(train_idx) + len(group) <= train_target:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
    return ([records[i] for i in train_idx], [records[i] for i in val_idx], [records[i] for i in test_idx])


def split_audit(train: Sequence[Record], val: Sequence[Record], test: Sequence[Record]) -> Dict:
    parts = {"train": list(train), "val": list(val), "test": list(test)}
    report: Dict = {"parts": {}}
    for name, rs in parts.items():
        smiles = [r.smiles for r in rs]
        labels = [r.label for r in rs]
        report["parts"][name] = {
            "rows": len(rs),
            "unique_canonical_smiles": len(set(smiles)),
            "positive": int(sum(labels)),
            "negative": int(len(labels) - sum(labels)),
            "positive_fraction": float(np.mean(labels)) if labels else float("nan"),
        }
    sets = {k: set(r.smiles for r in v) for k, v in parts.items()}
    report["canonical_smiles_overlap"] = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    all_records = [r for rs in parts.values() for r in rs]
    label_sets: Dict[str, set] = {}
    for r in all_records:
        label_sets.setdefault(r.smiles, set()).add(int(r.label))
    conflicts = [s for s, ys in label_sets.items() if len(ys) > 1]
    report["canonical_smiles_with_conflicting_labels"] = len(conflicts)
    report["conflicting_label_examples"] = conflicts[:20]
    return report


def split_manifest(train: Sequence[Record], val: Sequence[Record], test: Sequence[Record], mode: str) -> Dict:
    return {
        "mode": mode,
        "train_indices": [int(r.row_id) for r in train],
        "val_indices": [int(r.row_id) for r in val],
        "test_indices": [int(r.row_id) for r in test],
        "train_smiles": [r.smiles for r in train],
        "val_smiles": [r.smiles for r in val],
        "test_smiles": [r.smiles for r in test],
        "audit": split_audit(train, val, test),
    }


def fit_descriptor_scaler(records: Sequence[Record]) -> DescriptorScaler:
    rows = []
    for r in records:
        mol = mol_from_smiles(r.smiles)
        if mol is None:
            continue
        rows.append(descriptors_raw_10(mol))
    if not rows:
        raise ValueError("Cannot fit descriptor scaler: no valid training molecules")
    x = np.stack(rows).astype(np.float32)
    mean = x.mean(axis=0).astype(np.float32)
    scale = x.std(axis=0).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return DescriptorScaler(mean=mean, scale=scale)


class HERAHGTDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Record],
        feat_cfg: FeaturizerConfig,
        descriptor_scaler: Optional[DescriptorScaler] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.records = list(records)
        self.feat_cfg = feat_cfg
        self.descriptor_scaler = descriptor_scaler
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cfg_hash = hashlib.sha1(
            (FEATURE_SCHEMA_VERSION + json.dumps(feat_cfg.to_dict(), sort_keys=True)).encode()
        ).hexdigest()[:12]

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, smiles: str) -> Path:
        digest = hashlib.sha1(smiles.encode()).hexdigest()
        assert self.cache_dir is not None
        return self.cache_dir / f"{digest}_{self._cfg_hash}.pt"

    def __getitem__(self, idx: int) -> Dict:
        rec = self.records[idx]
        obj = None
        if self.cache_dir:
            p = self._cache_path(rec.smiles)
            if p.exists():
                try:
                    obj = torch.load(p, map_location="cpu", weights_only=False)
                except Exception:
                    obj = None
        if obj is None:
            obj = featurize_hierarchical_smiles(rec.smiles, self.feat_cfg)
            if self.cache_dir:
                torch.save(obj, self._cache_path(rec.smiles))
        out = dict(obj)
        raw_desc = np.asarray(out.get("desc_raw", out["desc"]), dtype=np.float32)
        out["desc"] = self.descriptor_scaler.transform(raw_desc) if self.descriptor_scaler else raw_desc
        out["label"] = float(rec.label)
        out["row_id"] = int(rec.row_id)
        return out


def collate_hera_hgt(batch: List[Dict]) -> Dict[str, torch.Tensor | List]:
    if not batch:
        raise ValueError("Empty batch")
    B = len(batch)
    max_nodes = max(int(x["node_x"].shape[0]) for x in batch)
    atom_dim = int(batch[0]["node_x"].shape[1])

    node_x = torch.zeros(B, max_nodes, atom_dim, dtype=torch.float32)
    node_level = torch.zeros(B, max_nodes, dtype=torch.long)
    motif_family = torch.zeros(B, max_nodes, dtype=torch.long)
    motif_size = torch.zeros(B, max_nodes, 1, dtype=torch.float32)
    relation = torch.zeros(B, max_nodes, max_nodes, dtype=torch.long)
    pair_dist = torch.zeros(B, max_nodes, max_nodes, dtype=torch.long)
    pair_bond10 = torch.zeros(B, max_nodes, max_nodes, 10, dtype=torch.float32)
    pair_mask = torch.zeros(B, max_nodes, max_nodes, dtype=torch.bool)
    node_mask = torch.zeros(B, max_nodes, dtype=torch.bool)
    global_index = torch.zeros(B, dtype=torch.long)

    for b, item in enumerate(batch):
        n = int(item["node_x"].shape[0])
        node_x[b, :n] = torch.as_tensor(item["node_x"], dtype=torch.float32)
        node_level[b, :n] = torch.as_tensor(item["node_level"], dtype=torch.long)
        motif_family[b, :n] = torch.as_tensor(item["motif_family"], dtype=torch.long)
        motif_size[b, :n] = torch.as_tensor(item["motif_size"], dtype=torch.float32)
        relation[b, :n, :n] = torch.as_tensor(item["relation"], dtype=torch.long)
        pair_dist[b, :n, :n] = torch.as_tensor(item["pair_dist"], dtype=torch.long)
        pair_bond10[b, :n, :n] = torch.as_tensor(item["pair_bond10"], dtype=torch.float32)
        pair_mask[b, :n, :n] = torch.as_tensor(item["pair_mask"], dtype=torch.bool)
        node_mask[b, :n] = True
        global_index[b] = int(item["global_index"])

    return {
        "node_x": node_x,
        "node_level": node_level,
        "motif_family": motif_family,
        "motif_size": motif_size,
        "relation": relation,
        "pair_dist": pair_dist,
        "pair_bond10": pair_bond10,
        "pair_mask": pair_mask,
        "node_mask": node_mask,
        "global_index": global_index,
        "fp_morgan": torch.stack([torch.as_tensor(x["fp_morgan"], dtype=torch.float32) for x in batch]),
        "fp_maccs": torch.stack([torch.as_tensor(x["fp_maccs"], dtype=torch.float32) for x in batch]),
        "fp_atompair": torch.stack([torch.as_tensor(x["fp_atompair"], dtype=torch.float32) for x in batch]),
        "desc": torch.stack([torch.as_tensor(x["desc"], dtype=torch.float32) for x in batch]),
        "label": torch.tensor([float(x["label"]) for x in batch], dtype=torch.float32),
        "row_id": torch.tensor([int(x["row_id"]) for x in batch], dtype=torch.long),
        "smiles": [str(x["smiles"]) for x in batch],
        "motif_names": [x["motif_names"] for x in batch],
        "motif_atoms": [x["motif_atoms"] for x in batch],
        "motif_families": [x["motif_families"] for x in batch],
    }
