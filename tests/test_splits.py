from pathlib import Path

import pytest

from hera_hgt.data import (
    Record,
    clean_random_split,
    clean_unique_records,
    grouped_random_split,
    hergat_exact_split,
    load_records_csv,
    split_audit,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundled_records():
    return load_records_csv(ROOT / "data" / "hERGAT_final_dataset.csv")


def test_bundled_hergat_split_matches_bundled_dataset(bundled_records):
    split = ROOT / "data" / "hergat_paper_run1_split_indices.json"
    tr, va, te = hergat_exact_split(bundled_records, split)
    assert (len(tr), len(va), len(te)) == (18704, 2338, 2339)


def test_grouped_random_has_no_canonical_overlap(bundled_records):
    tr, va, te = grouped_random_split(bundled_records[:1000], seed=7)
    a = split_audit(tr, va, te)["canonical_smiles_overlap"]
    assert a == {"train_val": 0, "train_test": 0, "val_test": 0}


def _synthetic_records_for_clean_tests():
    records = []
    row_id = 0
    # 120 unique, unambiguous compounds, balanced labels.
    for i in range(120):
        records.append(Record(smiles=f"mol_{i:03d}", label=i % 2, row_id=row_id))
        row_id += 1
    # Consistent duplicates: should collapse to one molecule each.
    for i in range(10):
        records.append(Record(smiles=f"mol_{i:03d}", label=i % 2, row_id=row_id))
        row_id += 1
    # Two conflicting canonical SMILES: both groups must be removed completely.
    records.extend([
        Record(smiles="conflict_a", label=0, row_id=row_id),
        Record(smiles="conflict_a", label=1, row_id=row_id + 1),
        Record(smiles="conflict_b", label=1, row_id=row_id + 2),
        Record(smiles="conflict_b", label=0, row_id=row_id + 3),
    ])
    return records


def test_clean_unique_records_are_unique_and_conflict_free():
    clean, report = clean_unique_records(_synthetic_records_for_clean_tests())

    smiles = [r.smiles for r in clean]
    assert len(smiles) == len(set(smiles))
    assert report["conflicting_canonical_smiles_removed"] == 2
    assert report["final_unique_compounds"] == 120

    label_sets = {}
    for r in clean:
        label_sets.setdefault(r.smiles, set()).add(int(r.label))
    assert all(len(labels) == 1 for labels in label_sets.values())


def test_clean_random_is_80_10_10_and_leakage_free():
    clean, _ = clean_unique_records(_synthetic_records_for_clean_tests())
    tr, va, te = clean_random_split(clean, seed=2026)
    audit = split_audit(tr, va, te)

    assert (len(tr), len(va), len(te)) == (96, 12, 12)
    assert audit["canonical_smiles_overlap"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert audit["canonical_smiles_with_conflicting_labels"] == 0

    for split_name in ("train", "val", "test"):
        part = audit["parts"][split_name]
        assert part["rows"] == part["unique_canonical_smiles"]
