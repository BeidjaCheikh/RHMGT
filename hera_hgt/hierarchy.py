from __future__ import annotations

from typing import Dict, List

import numpy as np

from .chemistry import (
    FAMILY_AROMATIC,
    FAMILY_BASIC_N,
    FAMILY_N_HETEROCYCLE,
    FAMILY_PERIPHERAL,
    FAMILY_STRUCTURAL,
    Motif,
    atom_bond_feature_matrix,
    atom_feature_dim,
    atom_features_hergat,
    atom_pair_fingerprint,
    descriptors_raw_10,
    extract_motifs,
    maccs_fingerprint,
    min_motif_distance,
    mol_from_smiles,
    motif_boundary_bond_features,
    morgan_fingerprint,
    shortest_path_matrix,
)
from .config import FeaturizerConfig


# Pair relation IDs. Relation identity already carries cross-level semantics, so
# the Transformer does not need a second pairwise "hierarchy bias" channel.
REL_NONE = 0
REL_SELF = 1
REL_ATOM_ATOM = 2
REL_ATOM_TO_MOTIF = 3
REL_MOTIF_TO_ATOM = 4
REL_MM_GENERIC = 5
REL_MM_BN_AR = 6
REL_MM_BN_NHC = 7
REL_MM_BN_PF = 8
REL_MM_AR_NHC = 9
REL_MM_AR_PF = 10
REL_MM_NHC_PF = 11
REL_MM_SAME_FAMILY = 12
REL_MOTIF_TO_GLOBAL = 13
REL_GLOBAL_TO_MOTIF = 14
RELATION_VOCAB_SIZE = 15

LEVEL_ATOM = 0
LEVEL_MOTIF = 1
LEVEL_GLOBAL = 2


def _motif_pair_relation(m1: Motif, m2: Motif) -> int:
    if m1.family == m2.family and m1.family != FAMILY_STRUCTURAL:
        return REL_MM_SAME_FAMILY
    pair = {m1.family, m2.family}
    if pair == {FAMILY_BASIC_N, FAMILY_AROMATIC}:
        return REL_MM_BN_AR
    if pair == {FAMILY_BASIC_N, FAMILY_N_HETEROCYCLE}:
        return REL_MM_BN_NHC
    if pair == {FAMILY_BASIC_N, FAMILY_PERIPHERAL}:
        return REL_MM_BN_PF
    if pair == {FAMILY_AROMATIC, FAMILY_N_HETEROCYCLE}:
        return REL_MM_AR_NHC
    if pair == {FAMILY_AROMATIC, FAMILY_PERIPHERAL}:
        return REL_MM_AR_PF
    if pair == {FAMILY_N_HETEROCYCLE, FAMILY_PERIPHERAL}:
        return REL_MM_NHC_PF
    return REL_MM_GENERIC


def featurize_hierarchical_smiles(smiles: str, cfg: FeaturizerConfig) -> Dict[str, np.ndarray | str | int | List]:
    mol = mol_from_smiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    n_atoms = mol.GetNumAtoms()
    atom_x = np.asarray([atom_features_hergat(a) for a in mol.GetAtoms()], dtype=np.float32)
    if atom_x.ndim != 2 or atom_x.shape[1] != atom_feature_dim():
        raise RuntimeError("Unexpected hERGAT atom feature dimension")

    max_dist = max(cfg.max_atom_dist, cfg.max_motif_dist)
    atom_dist = shortest_path_matrix(mol, max_dist)
    atom_bond10 = atom_bond_feature_matrix(mol)

    motifs = extract_motifs(mol, cfg)
    if not motifs:
        # Connectivity-only fallback. This is NOT a BRICS fragment and is used
        # only when an ablation/molecule leaves no explicit motif node. It keeps
        # the atom -> motif -> global topology valid without re-enabling BRICS.
        motifs = [Motif(tuple(range(n_atoms)), "connectivity_bridge", FAMILY_STRUCTURAL)]

    n_motifs = len(motifs)
    total = n_atoms + n_motifs + 1
    global_index = total - 1

    # Raw chemistry for atom and motif nodes. Motif nodes inherit the mean atom
    # chemistry of their members. The global node is a learned token in model.py.
    node_x = np.zeros((total, atom_feature_dim()), dtype=np.float32)
    node_x[:n_atoms] = atom_x
    motif_family = np.zeros((total,), dtype=np.int64)
    motif_size = np.zeros((total, 1), dtype=np.float32)
    node_level = np.full((total,), LEVEL_ATOM, dtype=np.int64)
    node_level[n_atoms:global_index] = LEVEL_MOTIF
    node_level[global_index] = LEVEL_GLOBAL

    for m_idx, motif in enumerate(motifs):
        idx = n_atoms + m_idx
        atom_ids = list(motif.atoms)
        node_x[idx] = atom_x[atom_ids].mean(axis=0)
        motif_family[idx] = int(motif.family)
        motif_size[idx, 0] = min(len(atom_ids), 20) / 20.0

    relation = np.full((total, total), REL_NONE, dtype=np.int64)
    pair_dist = np.full((total, total), max_dist + 1, dtype=np.int64)
    pair_bond10 = np.zeros((total, total, 10), dtype=np.float32)
    pair_mask = np.zeros((total, total), dtype=bool)

    # Self-attention.
    for i in range(total):
        relation[i, i] = REL_SELF
        pair_dist[i, i] = 0
        pair_mask[i, i] = True

    # Atom <-> atom context. Direct covalent neighbors carry the 10D bond vector;
    # nonbonded atom pairs within the topological radius carry zero bond vector
    # and are distinguished by distance > 1.
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            d = int(atom_dist[i, j])
            if d <= cfg.max_atom_dist:
                relation[i, j] = REL_ATOM_ATOM
                pair_dist[i, j] = d
                pair_bond10[i, j] = atom_bond10[i, j]
                pair_mask[i, j] = True

    # Atom <-> motif membership.
    for m_idx, motif in enumerate(motifs):
        mi = n_atoms + m_idx
        for a in motif.atoms:
            a = int(a)
            relation[a, mi] = REL_ATOM_TO_MOTIF
            relation[mi, a] = REL_MOTIF_TO_ATOM
            pair_dist[a, mi] = 1
            pair_dist[mi, a] = 1
            pair_mask[a, mi] = True
            pair_mask[mi, a] = True

    # Motif <-> motif interactions, informed by family pair, topological distance,
    # and (when present) chemistry of original covalent boundary bonds.
    for i, m1 in enumerate(motifs):
        ni = n_atoms + i
        for j, m2 in enumerate(motifs):
            if i == j:
                continue
            nj = n_atoms + j
            d = min_motif_distance(atom_dist, m1.atoms, m2.atoms, cfg.max_motif_dist)
            if d <= cfg.max_motif_dist:
                relation[ni, nj] = _motif_pair_relation(m1, m2)
                pair_dist[ni, nj] = d
                pair_bond10[ni, nj] = motif_boundary_bond_features(mol, m1.atoms, m2.atoms)
                pair_mask[ni, nj] = True

    # Motif <-> global; no direct atom-global shortcut. This preserves the
    # intended atom -> motif/pharmacophore -> molecule hierarchy.
    for m_idx in range(n_motifs):
        mi = n_atoms + m_idx
        relation[mi, global_index] = REL_MOTIF_TO_GLOBAL
        relation[global_index, mi] = REL_GLOBAL_TO_MOTIF
        pair_dist[mi, global_index] = 1
        pair_dist[global_index, mi] = 1
        pair_mask[mi, global_index] = True
        pair_mask[global_index, mi] = True

    return {
        "smiles": ChemCanonical(mol),
        "node_x": node_x,
        "node_level": node_level,
        "motif_family": motif_family,
        "motif_size": motif_size,
        "relation": relation,
        "pair_dist": pair_dist,
        "pair_bond10": pair_bond10,
        "pair_mask": pair_mask,
        "global_index": int(global_index),
        "n_atoms": int(n_atoms),
        "n_motifs": int(n_motifs),
        "motif_names": [m.name for m in motifs],
        "motif_families": [int(m.family) for m in motifs],
        "motif_atoms": [tuple(m.atoms) for m in motifs],
        "fp_morgan": morgan_fingerprint(mol, cfg.morgan_dim, cfg.morgan_radius),
        "fp_maccs": maccs_fingerprint(mol),
        "fp_atompair": atom_pair_fingerprint(mol, cfg.atom_pair_fp_dim),
        "desc_raw": descriptors_raw_10(mol),
        "desc": descriptors_raw_10(mol),
    }


def ChemCanonical(mol) -> str:
    from rdkit import Chem
    return Chem.MolToSmiles(mol, canonical=True)
