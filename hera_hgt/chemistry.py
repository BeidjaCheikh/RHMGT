from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, Lipinski, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

from .config import FeaturizerConfig


# -----------------------------------------------------------------------------
# Motif / pharmacophore families
# -----------------------------------------------------------------------------

FAMILY_STRUCTURAL = 0
FAMILY_BASIC_N = 1
FAMILY_AROMATIC = 2
FAMILY_N_HETEROCYCLE = 3
FAMILY_PERIPHERAL = 4

FAMILY_NAMES = {
    FAMILY_STRUCTURAL: "structural",
    FAMILY_BASIC_N: "basic_nitrogen",
    FAMILY_AROMATIC: "aromatic",
    FAMILY_N_HETEROCYCLE: "n_heterocycle",
    FAMILY_PERIPHERAL: "peripheral",
}

# Domain-guided hERG motif registry. Fine-grained names are kept only as
# interpretable metadata. The neural model receives the FAMILY embedding, not a
# separate embedding for every SMARTS pattern, which avoids unnecessary model
# complexity and duplicated semantic channels.
_HERG_SMARTS: Dict[str, Tuple[str, int]] = {
    "tertiary_amine": ("[NX3;H0;+0;!$(NC=O);!$(NS(=O));!$(N=*)]", FAMILY_BASIC_N),
    "secondary_amine": ("[NX3;H1;+0;!$(NC=O);!$(NS(=O));!$(N=*)]", FAMILY_BASIC_N),
    "piperidine": ("[NX3;H0,H1;R;!$(NC=O)]1CCCCC1", FAMILY_BASIC_N),
    "piperazine": ("[NX3;H0;R]1CC[NX3;H0;R]CC1", FAMILY_BASIC_N),
    "pyrrolidine_N": ("[NX3;H0,H1;R;!$(NC=O)]1CCCC1", FAMILY_BASIC_N),
    "phenyl": ("c1ccccc1", FAMILY_AROMATIC),
    "methoxyphenyl": ("COc1ccccc1", FAMILY_AROMATIC),
    "fluorophenyl": ("Fc1ccccc1", FAMILY_AROMATIC),
    "chlorophenyl": ("Clc1ccccc1", FAMILY_AROMATIC),
    "naphthalene": ("c1ccc2ccccc2c1", FAMILY_AROMATIC),
    "quinoxaline": ("c1cnc2ccccc2n1", FAMILY_N_HETEROCYCLE),
    "pyridine": ("c1ccncc1", FAMILY_N_HETEROCYCLE),
    "pyrimidine": ("c1cnccn1", FAMILY_N_HETEROCYCLE),
    "benzimidazole": ("c1ccc2[nH]cnc2c1", FAMILY_N_HETEROCYCLE),
    "indole": ("c1ccc2[nH]ccc2c1", FAMILY_N_HETEROCYCLE),
    "aryl_sulfonamide": ("c1ccccc1S(=O)(=O)[NX3]", FAMILY_PERIPHERAL),
    "carbonyl_N_aryl": ("[NX3]C(=O)c1ccccc1", FAMILY_PERIPHERAL),
    "methoxy_group": ("[CH3]Oc1ccccc1", FAMILY_PERIPHERAL),
    "basic_N_arene_link": ("[NX3;H0,H1][CH2][CH2]c1ccccc1", FAMILY_PERIPHERAL),
}
_COMPILED_HERG = {
    name: (Chem.MolFromSmarts(smarts), family)
    for name, (smarts, family) in _HERG_SMARTS.items()
}


@dataclass(frozen=True)
class Motif:
    atoms: Tuple[int, ...]
    name: str
    family: int


def pharmacophore_registry_issues() -> List[str]:
    issues: List[str] = []
    for name, (smarts, _) in _HERG_SMARTS.items():
        if Chem.MolFromSmarts(smarts) is None:
            issues.append(f"Invalid SMARTS: {name} -> {smarts}")
    return issues


# -----------------------------------------------------------------------------
# Molecule parsing
# -----------------------------------------------------------------------------

def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        return None
    return mol


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = mol_from_smiles(smiles)
    return None if mol is None else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


# -----------------------------------------------------------------------------
# hERGAT code-compatible atom and bond features
# -----------------------------------------------------------------------------

HERGAT_ATOM_SYMBOLS = [
    "B", "C", "N", "O", "F", "Si", "P", "S",
    "Cl", "As", "Se", "Br", "Te", "I", "At", "other",
]
HERGAT_DEGREES = [0, 1, 2, 3, 4, 5]
HERGAT_H_COUNTS = [0, 1, 2, 3, 4]
HERGAT_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    "other",
]
HERGAT_BOND_STEREO = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]


def _one_hot_unk(value, choices: Sequence) -> List[float]:
    choices = list(choices)
    mapped = value if value in choices else choices[-1]
    return [float(mapped == c) for c in choices]


def _one_hot_or_zero(value, choices: Sequence) -> List[float]:
    # hERGAT's training data uses degree 0..5. For an unexpected degree, keep
    # the fixed 6D schema without crashing inference.
    return [float(value == c) for c in choices]


def atom_features_hergat(atom: Chem.Atom) -> List[float]:
    """39D atom vector matching the ordering in the released hERGAT code.

    16 symbol + 6 degree + formal charge + radical electrons
    + 6 hybridization + aromaticity + 5 H count + 2 CIP + chirality-possible.
    """
    symbol = atom.GetSymbol()
    symbol_vec = _one_hot_unk(symbol if symbol in HERGAT_ATOM_SYMBOLS[:-1] else "other", HERGAT_ATOM_SYMBOLS)
    degree_vec = _one_hot_or_zero(int(atom.GetDegree()), HERGAT_DEGREES)
    hyb = atom.GetHybridization()
    hyb_vec = _one_hot_unk(hyb if hyb in HERGAT_HYBRIDIZATIONS[:-1] else "other", HERGAT_HYBRIDIZATIONS)
    h_vec = _one_hot_unk(int(atom.GetTotalNumHs()), HERGAT_H_COUNTS)
    try:
        cip = atom.GetProp("_CIPCode")
    except Exception:
        cip = ""
    cip_vec = _one_hot_unk(cip, ["R", "S"]) if cip in ("R", "S") else [0.0, 0.0]
    chirality_possible = float(atom.HasProp("_ChiralityPossible"))

    feats = (
        symbol_vec
        + degree_vec
        + [float(atom.GetFormalCharge()), float(atom.GetNumRadicalElectrons())]
        + hyb_vec
        + [float(atom.GetIsAromatic())]
        + h_vec
        + cip_vec
        + [chirality_possible]
    )
    if len(feats) != 39:
        raise RuntimeError(f"atom_features_hergat produced {len(feats)} dims, expected 39")
    return feats


def atom_feature_dim() -> int:
    return 39


def bond_features_10(bond: Chem.Bond) -> np.ndarray:
    """10D bond vector matching the released hERGAT featurizer.

    4 bond types + conjugation + ring + 4 stereo states
    (NONE, ANY, Z, E).
    """
    bt = bond.GetBondType()
    bond_type = [
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
    ]
    stereo = str(bond.GetStereo())
    stereo_vec = _one_hot_unk(stereo, HERGAT_BOND_STEREO)
    out = np.asarray(
        bond_type + [float(bond.GetIsConjugated()), float(bond.IsInRing())] + stereo_vec,
        dtype=np.float32,
    )
    if out.shape != (10,):
        raise RuntimeError("bond_features_10 must return shape (10,)")
    return out


def atom_bond_feature_matrix(mol: Chem.Mol) -> np.ndarray:
    n = mol.GetNumAtoms()
    out = np.zeros((n, n, 10), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        f = bond_features_10(bond)
        out[i, j] = f
        out[j, i] = f
    return out


def shortest_path_matrix(mol: Chem.Mol, max_dist: int) -> np.ndarray:
    if mol.GetNumAtoms() == 0:
        return np.zeros((0, 0), dtype=np.int64)
    raw = Chem.GetDistanceMatrix(mol)
    return np.minimum(raw.astype(np.int64), max_dist + 1)


# -----------------------------------------------------------------------------
# Molecular evidence: Morgan + MACCS + AtomPair + 10 raw descriptors
# -----------------------------------------------------------------------------

def _bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def morgan_fingerprint(mol: Chem.Mol, n_bits: int, radius: int) -> np.ndarray:
    try:
        fp = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits).GetFingerprint(mol)
    except Exception:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return _bitvect_to_array(fp, n_bits)


def maccs_fingerprint(mol: Chem.Mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    return _bitvect_to_array(fp, fp.GetNumBits())


def atom_pair_fingerprint(mol: Chem.Mol, n_bits: int) -> np.ndarray:
    try:
        fp = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits).GetFingerprint(mol)
    except Exception:
        fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
    return _bitvect_to_array(fp, n_bits)


def _count_basic_nitrogens(mol: Chem.Mol) -> int:
    """Simple, deterministic basic-N count used only as one global descriptor."""
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetIsAromatic() or atom.GetFormalCharge() < 0:
            continue
        is_amide_like = False
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() not in (6, 16):
                continue
            for b2 in nbr.GetBonds():
                other = b2.GetOtherAtom(nbr)
                if other.GetIdx() == atom.GetIdx():
                    continue
                if other.GetAtomicNum() in (8, 16) and b2.GetBondTypeAsDouble() >= 2.0:
                    is_amide_like = True
                    break
            if is_amide_like:
                break
        if not is_amide_like:
            count += 1
    return count


DESCRIPTOR_NAMES = [
    "MW",
    "LogP",
    "TPSA",
    "HBA",
    "HBD",
    "RotatableBonds",
    "AromaticRingCount",
    "FractionCSP3",
    "HeavyAtomCount",
    "BasicNitrogenCount",
]


def descriptors_raw_10(mol: Chem.Mol) -> np.ndarray:
    """Raw descriptor vector. Standardization is fit on TRAIN only in data.py."""
    values = np.asarray([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        float(_count_basic_nitrogens(mol)),
    ], dtype=np.float32)
    out = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if out.shape != (10,):
        raise RuntimeError("descriptors_raw_10 must return shape (10,)")
    return out


# -----------------------------------------------------------------------------
# Hierarchical motifs: BRICS structural motifs + hERG pharmacophores only.
# No separate ring nodes: ring/aromatic information is already present in the
# atom/bond chemistry and in hERG aromatic motifs.
# -----------------------------------------------------------------------------

def _brics_motifs(mol: Chem.Mol, min_atoms: int) -> List[Motif]:
    cut_pairs: List[Tuple[int, int]] = []
    for item in BRICS.FindBRICSBonds(mol):
        (a, b), _labels = item
        cut_pairs.append((int(a), int(b)))

    # Fallback structural node guarantees atom -> motif -> global connectivity
    # for molecules that have no BRICS-cuttable bond.
    if not cut_pairs:
        atoms = tuple(range(mol.GetNumAtoms()))
        return [Motif(atoms, "brics_whole_molecule", FAMILY_STRUCTURAL)] if len(atoms) >= min_atoms else []

    rw = Chem.RWMol(mol)
    for a, b in cut_pairs:
        if rw.GetBondBetweenAtoms(a, b) is not None:
            rw.RemoveBond(a, b)
    frag_mol = rw.GetMol()
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    motifs: List[Motif] = []
    for frag in frags:
        atoms = tuple(sorted(int(i) for i in frag))
        if len(atoms) >= min_atoms:
            motifs.append(Motif(atoms, "brics_fragment", FAMILY_STRUCTURAL))
    return motifs


def _pharmacophore_motifs(mol: Chem.Mol) -> List[Motif]:
    out: List[Motif] = []
    seen = set()
    for name, (pattern, family) in _COMPILED_HERG.items():
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern, uniquify=True):
            atoms = tuple(sorted(int(i) for i in match))
            # Since the model embeds only the family, exact same atom set and
            # family is one semantic node even if multiple SMARTS match it.
            key = (atoms, family)
            if atoms and key not in seen:
                seen.add(key)
                out.append(Motif(atoms, name, family))
    return out


def extract_motifs(mol: Chem.Mol, cfg: FeaturizerConfig) -> List[Motif]:
    structural = _brics_motifs(mol, cfg.min_brics_atoms) if cfg.include_brics else []
    pharmacophores = _pharmacophore_motifs(mol) if cfg.include_pharmacophores else []

    # Keep BRICS coverage first, then add domain-specific nodes. This guarantees
    # a structural path from atoms to the global node whenever BRICS is enabled.
    unique: List[Motif] = []
    seen = set()
    for motif in structural + pharmacophores:
        key = (motif.atoms, motif.family)
        if key not in seen:
            seen.add(key)
            unique.append(motif)

    max_motifs = max(1, int(cfg.max_motifs))
    if len(unique) <= max_motifs:
        return unique

    # If truncation is necessary, preserve structural coverage first, then the
    # largest pharmacophore regions.
    structural_u = [m for m in unique if m.family == FAMILY_STRUCTURAL]
    pharma_u = [m for m in unique if m.family != FAMILY_STRUCTURAL]
    if len(structural_u) >= max_motifs:
        return structural_u[:max_motifs]
    pharma_u.sort(key=lambda m: (-len(m.atoms), m.family, m.atoms))
    return structural_u + pharma_u[: max_motifs - len(structural_u)]


def motif_boundary_bond_features(mol: Chem.Mol, atoms_i: Iterable[int], atoms_j: Iterable[int]) -> np.ndarray:
    """Mean 10D hERGAT chemistry of covalent bonds crossing two motif regions."""
    set_i, set_j = set(map(int, atoms_i)), set(map(int, atoms_j))
    only_i, only_j = set_i - set_j, set_j - set_i
    features: List[np.ndarray] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (a in only_i and b in only_j) or (a in only_j and b in only_i):
            features.append(bond_features_10(bond))
    if not features:
        return np.zeros((10,), dtype=np.float32)
    return np.mean(np.stack(features, axis=0), axis=0).astype(np.float32)


def min_motif_distance(atom_dist: np.ndarray, atoms_i: Iterable[int], atoms_j: Iterable[int], max_dist: int) -> int:
    ai, aj = list(map(int, atoms_i)), list(map(int, atoms_j))
    if not ai or not aj:
        return max_dist + 1
    d = int(np.min(atom_dist[np.ix_(ai, aj)]))
    return min(d, max_dist + 1)
