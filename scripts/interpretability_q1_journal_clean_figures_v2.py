"""Publication-grade hierarchical interpretability pipeline for RHMGT.

The pipeline uses one predefined, structurally diverse showcase panel across the
local explanation levels so the reader can follow the same compounds from atoms
to motifs, explicit pharmacophores, and PIG relations. It performs signed raw-input
Integrated Gradients, BRICS fragment feature occlusion, motif/pharmacophore-node
occlusion, PIG edge deletion + relation neutralization, quantitative screening,
global aggregation with molecule-level bootstrap confidence intervals, primary explanation faithfulness tests, attention-ranking diagnostics, and adaptive-fusion gate diagnostics.

Primary article figures are generated separately for atom-level attribution, hierarchical motif nodes, explicit pharmacophore occurrences, PIG relations, global test-set patterns, and quantitative primary-explanation faithfulness. BRICS-only, attention-ranking faithfulness, and adaptive-fusion gate views are supplementary. Attention is
exported only as an auxiliary diagnostic; direct perturbation is the primary
importance measure for motif and relation explanations.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch.utils.data import DataLoader

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "hera_hgt").exists() and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CWD = Path.cwd().resolve()
if (CWD / "hera_hgt").exists() and str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

from hera_hgt.checkpoint import load_checkpoint
from hera_hgt.chemistry import FAMILY_NAMES, canonicalize_smiles, mol_from_smiles
from hera_hgt.data import HERAHGTDataset, Record, collate_hera_hgt, hergat_exact_split, load_records_csv
from hera_hgt.hierarchy import featurize_hierarchical_smiles
from hera_hgt.train_utils import move_batch


CASE_ORDER = ("TP", "TN", "FP", "FN")
ARTICLE10_COUNTS = {"TP": 3, "TN": 3, "FP": 2, "FN": 2}
FAMILY_SHORT = {0: "STR", 1: "BN", 2: "AR", 3: "NHC", 4: "PF"}
REL_NAMES = {
    0: "none",
    1: "self",
    2: "atom_atom",
    3: "atom_to_motif",
    4: "motif_to_atom",
    5: "mm_generic",
    6: "mm_bn_ar",
    7: "mm_bn_nhc",
    8: "mm_bn_pf",
    9: "mm_ar_nhc",
    10: "mm_ar_pf",
    11: "mm_nhc_pf",
    12: "mm_same_family",
    13: "motif_to_global",
    14: "global_to_motif",
}


@dataclass
class MoleculeSummary:
    row_id: int
    smiles: str
    label: Optional[int]
    prediction: int
    case: str
    threshold: float
    p_blocker: float
    blocker_logit: float
    baseline_p_blocker: float
    baseline_blocker_logit: float
    ig_sum: float
    ig_completeness_delta: float
    ig_relative_error: float
    forward_consistency_abs_error: float
    n_atoms: int
    n_motifs: int
    top_atom_idx: int
    top_atom_score: float
    top_fragment: str
    top_fragment_contribution: float
    top_motif: str
    top_motif_contribution: float
    top_family: str
    top_family_contribution: float
    top_relation: str
    top_relation_contribution: float
    atom_figure: str
    fragment_figure: str
    motif_figure: str
    pig_figure: str
    case_study_figure: str


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full RHMGT molecular interpretability: atoms -> fragments -> motifs -> PIG -> global"
    )
    p.add_argument("--checkpoint", required=True, help="Path to RHMGT best.pt")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", help="Explain one SMILES")
    src.add_argument("--csv", help="CSV dataset for selection/analysis")

    p.add_argument("--label", type=int, choices=[0, 1], default=None)
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")
    p.add_argument(
        "--split-json",
        default=None,
        help=(
            "Optional hERGAT paper_run1 split JSON. When provided with --csv, "
            "the script reconstructs the exact official train/val/test row split."
        ),
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which exact hERGAT partition to analyze when --split-json is provided (default: test).",
    )
    p.add_argument("--row-ids", nargs="*", type=int, default=None)
    p.add_argument(
        "--selection",
        choices=["article10", "tp", "tn", "fp", "fn", "all"],
        default="article10",
        help="Detailed local-analysis selection. article10 = 3 TP + 3 TN + 2 FP + 2 FN.",
    )
    p.add_argument("--max-molecules", type=int, default=10)
    p.add_argument("--selection-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cache-dir", default=".cache/hera_hgt_interpretability_full")

    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--ig-steps", type=int, default=64)
    p.add_argument("--consistency-tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="interpretability/full")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--edge-top-k", type=int, default=40, help="Max strongest PIG edges shown per molecule")
    p.add_argument("--figure-width", type=int, default=1800)
    p.add_argument("--figure-height", type=int, default=1200)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--show-atom-scores", action="store_true")
    p.add_argument(
        "--global-all",
        action="store_true",
        help=(
            "Additionally compute scalable global motif/family/relation summaries over the full CSV. "
            "Detailed IG/figures remain on the selected local subset."
        ),
    )
    p.add_argument(
        "--global-max-molecules",
        type=int,
        default=0,
        help="Optional cap for --global-all; 0 means all valid rows.",
    )
    return p.parse_args()


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")
    return text[:140] if text else "molecule"


def _case_name(label: Optional[int], pred: int) -> str:
    if label is None:
        return "NA"
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def _family_name(fid: int) -> str:
    return FAMILY_NAMES.get(int(fid), f"family_{int(fid)}")


def _family_short(fid: int) -> str:
    return FAMILY_SHORT.get(int(fid), f"F{int(fid)}")


def _normalize_signed(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scale = float(np.max(np.abs(scores))) if scores.size else 0.0
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(scores, dtype=np.float64)
    return np.clip(scores / scale, -1.0, 1.0)


def _build_single_item(smiles: str, label: Optional[int], row_id: int, feat_cfg, scaler):
    can = canonicalize_smiles(smiles)
    if can is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    item = featurize_hierarchical_smiles(can, feat_cfg)
    item["desc"] = scaler.transform(np.asarray(item["desc_raw"], dtype=np.float32))
    item["label"] = float(0 if label is None else label)
    item["row_id"] = int(row_id)
    return item, can


def _prob_and_logit(model, batch: Dict) -> Tuple[float, float, Dict]:
    with torch.no_grad():
        out = model(batch)
    return float(out["prob"][0].item()), float(out["logits"][0].item()), out


def _records_by_row_ids(records: Sequence[Record], row_ids: Sequence[int]) -> List[Record]:
    by_id = {int(r.row_id): r for r in records}
    missing = [int(rid) for rid in row_ids if int(rid) not in by_id]
    if missing:
        raise ValueError(f"Unknown CSV row IDs: {missing[:20]}")
    return [by_id[int(rid)] for rid in row_ids]


def _predict_records(
    model,
    records: Sequence[Record],
    feat_cfg,
    scaler,
    device: torch.device,
    threshold: float,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
) -> pd.DataFrame:
    ds = HERAHGTDataset(records, feat_cfg, scaler, cache_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_hera_hgt,
    )
    rows: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            probs = out["prob"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().astype(int)
            row_ids = batch["row_id"].detach().cpu().numpy().astype(int)
            for rid, smi, y, p in zip(row_ids, batch["smiles"], labels, probs):
                pred = int(float(p) >= threshold)
                rows.append({
                    "row_id": int(rid),
                    "smiles": str(smi),
                    "label": int(y),
                    "p_blocker": float(p),
                    "prediction": pred,
                    "case": _case_name(int(y), pred),
                })
    return pd.DataFrame(rows)


def _interior_quantile_pick(group: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or group.empty:
        return group.iloc[0:0].copy()
    g = group.sort_values(["p_blocker", "row_id"]).reset_index(drop=True)
    if len(g) <= n:
        return g
    positions = np.linspace(0, len(g) - 1, n + 2)[1:-1]
    idx = np.unique(np.rint(positions).astype(int))
    chosen = list(idx.tolist())
    if len(chosen) < n:
        for candidate in range(len(g)):
            if candidate not in chosen:
                chosen.append(candidate)
            if len(chosen) == n:
                break
    return g.iloc[sorted(chosen[:n])].copy()


def _select_from_predictions(pred_df: pd.DataFrame, selection: str, max_molecules: int) -> pd.DataFrame:
    if max_molecules < 1:
        raise ValueError("--max-molecules must be >= 1")
    if selection == "article10":
        pieces = []
        for case in CASE_ORDER:
            pieces.append(_interior_quantile_pick(pred_df[pred_df["case"] == case], ARTICLE10_COUNTS[case]))
        return pd.concat(pieces, ignore_index=True) if pieces else pred_df.iloc[0:0]
    if selection in {"tp", "tn", "fp", "fn"}:
        return _interior_quantile_pick(pred_df[pred_df["case"] == selection.upper()], max_molecules)
    return _interior_quantile_pick(pred_df, max_molecules)


# -----------------------------------------------------------------------------
# Exact hierarchy-aware forward for atom IG / fragment feature occlusion
# -----------------------------------------------------------------------------
def _hierarchical_forward_from_atom_embeddings(
    model,
    batch: Dict,
    atom_chem: torch.Tensor,
    motif_atoms: Sequence[Sequence[int]],
) -> Dict[str, torch.Tensor]:
    if atom_chem.ndim != 2:
        raise ValueError("atom_chem must have shape [n_atoms, hidden_dim]")
    if batch["node_x"].shape[0] != 1:
        raise ValueError("Attribution forward expects batch size 1")

    enc = model.graph_encoder
    device = atom_chem.device
    dtype = atom_chem.dtype
    n_atoms = atom_chem.shape[0]
    global_index = int(batch["global_index"][0].item())
    n_nodes = global_index + 1
    expected_motifs = global_index - n_atoms
    if expected_motifs != len(motif_atoms):
        raise RuntimeError(
            f"Motif mapping mismatch: graph has {expected_motifs} motif nodes, metadata has {len(motif_atoms)}"
        )

    if motif_atoms:
        motif_rows = []
        for atoms in motif_atoms:
            idx = torch.as_tensor(list(atoms), device=device, dtype=torch.long)
            if idx.numel() == 0:
                raise RuntimeError("Encountered empty motif")
            motif_rows.append(atom_chem.index_select(0, idx).mean(dim=0, keepdim=True))
        motif_chem = torch.cat(motif_rows, dim=0)
    else:
        motif_chem = atom_chem.new_zeros((0, atom_chem.shape[1]))

    global_chem = atom_chem.new_zeros((1, atom_chem.shape[1]))
    h = torch.cat([atom_chem, motif_chem, global_chem], dim=0).unsqueeze(0)
    if h.shape[1] != n_nodes:
        raise RuntimeError(f"Rebuilt node count {h.shape[1]} != expected {n_nodes}")

    node_level = batch["node_level"][:, :n_nodes].clamp(0, 2)
    h = h + enc.level_embed(node_level)
    motif_mask = (node_level == 1).unsqueeze(-1).to(dtype)
    family = enc.motif_family_embed(
        batch["motif_family"][:, :n_nodes].clamp(0, enc.motif_family_embed.num_embeddings - 1)
    )
    h = h + motif_mask * (family + enc.size_proj(batch["motif_size"][:, :n_nodes]))

    global_mask = (node_level == 2).unsqueeze(-1)
    global_init = enc.global_token.expand(1, n_nodes, -1) + enc.level_embed(node_level)
    h = torch.where(global_mask, global_init, h)

    node_mask = batch["node_mask"][:, :n_nodes]
    h = enc.input_norm(h)
    h = h * node_mask.unsqueeze(-1).to(dtype)

    relation = batch["relation"][:, :n_nodes, :n_nodes]
    pair_dist = batch["pair_dist"][:, :n_nodes, :n_nodes]
    pair_bond10 = batch["pair_bond10"][:, :n_nodes, :n_nodes]
    pair_mask = batch["pair_mask"][:, :n_nodes, :n_nodes]

    for layer in enc.layers:
        h = layer(h, relation, pair_dist, pair_bond10, pair_mask, node_mask)
    h = enc.final_norm(h)
    global_h = h[:, global_index]

    mode = model.cfg.prediction_mode
    if mode == "evidence_only":
        raise ValueError("Graph interpretability is undefined for evidence_only mode")
    if mode == "graph_only":
        fused = global_h
        evidence = atom_chem.new_empty((1, 0, model.cfg.hidden_dim))
        fusion_weights = atom_chem.new_empty((1, 0))
    else:
        evidence = model.evidence_encoder(batch)
        if model.cfg.fusion_type == "cross_attention":
            fused, _ = model.fusion(global_h, evidence)
            fusion_weights = atom_chem.new_empty((1, 0))
        else:
            fused, _, fusion_weights = model.fusion(global_h, evidence)

    logits = model.classifier(fused).squeeze(-1)
    return {
        "logits": logits,
        "prob": torch.sigmoid(logits),
        "global_repr": global_h,
        "node_embeddings": h,
        "evidence_tokens": evidence,
        "fusion_weights": fusion_weights,
    }


# -----------------------------------------------------------------------------
# A) Atom-level IG
# -----------------------------------------------------------------------------
def integrated_gradients_atom_embeddings(
    model,
    batch: Dict,
    item: Dict,
    steps: int,
    consistency_tol: float,
) -> Dict:
    if steps < 8:
        raise ValueError("--ig-steps must be >= 8")
    n_atoms = int(item["n_atoms"])
    motif_atoms = item["motif_atoms"]

    with torch.no_grad():
        full_atom_chem = model.graph_encoder.node_proj(batch["node_x"][:, :n_atoms]).squeeze(0)
        normal_out = model(batch)
        normal_logit = float(normal_out["logits"][0].item())
        normal_prob = float(normal_out["prob"][0].item())
        exact_full = _hierarchical_forward_from_atom_embeddings(model, batch, full_atom_chem, motif_atoms)
        exact_full_logit = float(exact_full["logits"][0].item())
        forward_error = abs(normal_logit - exact_full_logit)
        if forward_error > consistency_tol:
            raise RuntimeError(
                f"IG forward mismatch |Δlogit|={forward_error:.6g} > {consistency_tol:.6g}"
            )
        baseline_atom_chem = torch.zeros_like(full_atom_chem)
        baseline_out = _hierarchical_forward_from_atom_embeddings(model, batch, baseline_atom_chem, motif_atoms)
        baseline_logit = float(baseline_out["logits"][0].item())
        baseline_prob = float(baseline_out["prob"][0].item())

    diff = full_atom_chem - baseline_atom_chem
    grad_sum = torch.zeros_like(full_atom_chem)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=diff.device, dtype=diff.dtype)
    for i, alpha in enumerate(alphas):
        x = (baseline_atom_chem + alpha * diff).detach().requires_grad_(True)
        out = _hierarchical_forward_from_atom_embeddings(model, batch, x, motif_atoms)
        target_logit = out["logits"][0]
        grad = torch.autograd.grad(target_logit, x, retain_graph=False, create_graph=False)[0]
        weight = 0.5 if i == 0 or i == steps else 1.0
        grad_sum += weight * grad.detach()

    ig = diff * (grad_sum / float(steps))
    atom_scores = ig.sum(dim=-1)
    ig_sum = float(atom_scores.sum().item())
    output_delta = normal_logit - baseline_logit
    completeness_delta = ig_sum - output_delta
    rel_error = abs(completeness_delta) / max(abs(output_delta), 1e-8)

    return {
        "embedding_attributions": ig.detach().cpu().numpy().astype(np.float32),
        "atom_scores": atom_scores.detach().cpu().numpy().astype(np.float32),
        "full_atom_chem": full_atom_chem.detach(),
        "normal_logit": normal_logit,
        "normal_prob": normal_prob,
        "baseline_logit": baseline_logit,
        "baseline_prob": baseline_prob,
        "ig_sum": ig_sum,
        "completeness_delta": completeness_delta,
        "relative_error": rel_error,
        "forward_consistency_abs_error": forward_error,
    }


# -----------------------------------------------------------------------------
# Drawing molecules
# -----------------------------------------------------------------------------
def draw_signed_atom_heatmap(
    smiles: str,
    atom_scores: np.ndarray,
    out_path: Path,
    legend: str,
    width: int,
    height: int,
    dpi: int,
    show_atom_scores: bool = False,
) -> None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot draw invalid SMILES: {smiles!r}")
    if mol.GetNumAtoms() != len(atom_scores):
        raise RuntimeError(f"Atom count {mol.GetNumAtoms()} != score count {len(atom_scores)}")

    norm = _normalize_signed(atom_scores)
    mol = Chem.Mol(mol)
    if show_atom_scores:
        for i, value in enumerate(norm):
            mol.GetAtomWithIdx(i).SetProp("atomNote", f"{value:+.2f}")

    highlight_atoms: List[int] = []
    highlight_colors: Dict[int, Tuple[float, float, float]] = {}
    highlight_radii: Dict[int, float] = {}
    for i, value in enumerate(norm):
        mag = abs(float(value))
        if mag < 1e-12:
            continue
        strength = 0.18 + 0.82 * mag
        color = (1.0, 1.0 - strength, 1.0 - strength) if value > 0 else (1.0 - strength, 1.0 - strength, 1.0)
        highlight_atoms.append(i)
        highlight_colors[i] = color
        highlight_radii[i] = 0.25 + 0.25 * mag

    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.continuousHighlight = True
    opts.highlightBondWidthMultiplier = 8
    opts.legendFontSize = max(24, int(height * 0.025))
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        legend=legend,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    image.save(out_path, format="PNG", dpi=(int(dpi), int(dpi)), optimize=True)


# -----------------------------------------------------------------------------
# B) Fragment / motif / family occlusion
# -----------------------------------------------------------------------------
def _brics_fragments(mol: Chem.Mol) -> List[Dict]:
    cut_pairs: List[Tuple[int, int]] = []
    for item in BRICS.FindBRICSBonds(mol):
        (a, b), _labels = item
        cut_pairs.append((int(a), int(b)))
    if not cut_pairs:
        atoms = tuple(range(mol.GetNumAtoms()))
        return [{
            "fragment_id": 0,
            "atoms": atoms,
            "fragment_name": "whole_molecule",
            "fragment_smiles": Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True),
        }]

    rw = Chem.RWMol(mol)
    for a, b in cut_pairs:
        if rw.GetBondBetweenAtoms(a, b) is not None:
            rw.RemoveBond(a, b)
    frag_mol = rw.GetMol()
    frags = Chem.GetMolFrags(frag_mol, asMols=False, sanitizeFrags=False)
    out: List[Dict] = []
    for i, frag in enumerate(frags):
        atoms = tuple(sorted(int(x) for x in frag))
        out.append({
            "fragment_id": i,
            "atoms": atoms,
            "fragment_name": f"brics_fragment_{i}",
            "fragment_smiles": Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), canonical=True),
        })
    return out


def _mask_atom_embeddings(full_atom_chem: torch.Tensor, atom_ids: Iterable[int]) -> torch.Tensor:
    x = full_atom_chem.clone()
    ids = sorted(set(int(i) for i in atom_ids))
    if ids:
        x[torch.as_tensor(ids, device=x.device, dtype=torch.long)] = 0.0
    return x


def _clone_batch(batch: Dict) -> Dict:
    out: Dict = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.clone()
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _mask_motif_nodes(batch: Dict, motif_node_ids: Sequence[int]) -> Dict:
    out = _clone_batch(batch)
    if not motif_node_ids:
        return out
    ids = torch.as_tensor(sorted(set(int(i) for i in motif_node_ids)), device=out["node_mask"].device, dtype=torch.long)
    out["node_mask"][0, ids] = False
    out["pair_mask"][0, ids, :] = False
    out["pair_mask"][0, :, ids] = False
    out["relation"][0, ids, :] = 0
    out["relation"][0, :, ids] = 0
    max_pd = int(out["pair_dist"].max().item()) if out["pair_dist"].numel() else 0
    out["pair_dist"][0, ids, :] = max_pd
    out["pair_dist"][0, :, ids] = max_pd
    out["pair_bond10"][0, ids, :, :] = 0.0
    out["pair_bond10"][0, :, ids, :] = 0.0
    return out


def _mask_pair_edge(batch: Dict, i: int, j: int) -> Dict:
    """Remove one motif-motif relation in both directions while preserving nodes."""
    out = _clone_batch(batch)
    for a, b in ((i, j), (j, i)):
        out["pair_mask"][0, a, b] = False
        out["relation"][0, a, b] = 0
        out["pair_bond10"][0, a, b, :] = 0.0
        out["pair_dist"][0, a, b] = int(out["pair_dist"].max().item())
    return out


def _fragment_occlusion(model, batch: Dict, item: Dict, full_prob: float, full_logit: float, full_atom_chem: torch.Tensor) -> List[Dict]:
    mol = mol_from_smiles(item["smiles"])
    if mol is None:
        raise ValueError(f"Invalid SMILES: {item['smiles']}")
    rows: List[Dict] = []
    for frag in _brics_fragments(mol):
        x = _mask_atom_embeddings(full_atom_chem, frag["atoms"])
        out = _hierarchical_forward_from_atom_embeddings(model, batch, x, item["motif_atoms"])
        masked_prob = float(out["prob"][0].item())
        masked_logit = float(out["logits"][0].item())
        contribution = full_prob - masked_prob
        rows.append({
            "fragment_id": int(frag["fragment_id"]),
            "fragment_name": str(frag["fragment_name"]),
            "fragment_smiles": str(frag["fragment_smiles"]),
            "atoms": " ".join(map(str, frag["atoms"])),
            "n_atoms": len(frag["atoms"]),
            "p_blocker_full": full_prob,
            "p_blocker_masked": masked_prob,
            "probability_contribution": contribution,
            "logit_contribution": full_logit - masked_logit,
            "direction": "blocker" if contribution > 0 else "non_blocker" if contribution < 0 else "neutral",
        })
    rows.sort(key=lambda r: (-abs(float(r["probability_contribution"])), int(r["fragment_id"])))
    return rows


def _motif_and_family_occlusion(model, batch: Dict, item: Dict, full_prob: float, full_logit: float) -> Tuple[List[Dict], List[Dict]]:
    n_atoms = int(item["n_atoms"])
    n_motifs = int(item["n_motifs"])
    motif_rows: List[Dict] = []
    family_nodes: Dict[int, List[int]] = {}

    for m_idx in range(n_motifs):
        node_id = n_atoms + m_idx
        family = int(item["motif_families"][m_idx])
        atoms = tuple(int(a) for a in item["motif_atoms"][m_idx])
        masked_prob, masked_logit, _ = _prob_and_logit(model, _mask_motif_nodes(batch, [node_id]))
        contribution = full_prob - masked_prob
        motif_rows.append({
            "motif_id": m_idx,
            "motif_node_id": node_id,
            "motif_name": str(item["motif_names"][m_idx]),
            "motif_family_id": family,
            "motif_family": _family_name(family),
            "motif_family_short": _family_short(family),
            "atoms": " ".join(map(str, atoms)),
            "n_atoms": len(atoms),
            "p_blocker_full": full_prob,
            "p_blocker_masked": masked_prob,
            "probability_contribution": contribution,
            "logit_contribution": full_logit - masked_logit,
            "direction": "blocker" if contribution > 0 else "non_blocker" if contribution < 0 else "neutral",
        })
        family_nodes.setdefault(family, []).append(node_id)

    family_rows: List[Dict] = []
    for family, node_ids in sorted(family_nodes.items()):
        masked_prob, masked_logit, _ = _prob_and_logit(model, _mask_motif_nodes(batch, node_ids))
        contribution = full_prob - masked_prob
        family_rows.append({
            "motif_family_id": family,
            "motif_family": _family_name(family),
            "motif_family_short": _family_short(family),
            "n_nodes_masked": len(node_ids),
            "p_blocker_full": full_prob,
            "p_blocker_masked": masked_prob,
            "probability_contribution": contribution,
            "logit_contribution": full_logit - masked_logit,
            "direction": "blocker" if contribution > 0 else "non_blocker" if contribution < 0 else "neutral",
        })

    motif_rows.sort(key=lambda r: (-abs(float(r["probability_contribution"])), int(r["motif_id"])))
    family_rows.sort(key=lambda r: (-abs(float(r["probability_contribution"])), int(r["motif_family_id"])))
    return motif_rows, family_rows


def _project_group_scores_to_atoms(groups: List[Dict], n_atoms: int, score_key: str) -> np.ndarray:
    scores = np.zeros((n_atoms,), dtype=np.float32)
    hits = np.zeros((n_atoms,), dtype=np.float32)
    for row in groups:
        atom_ids = [int(x) for x in str(row["atoms"]).split() if str(x).strip()]
        for a in atom_ids:
            if 0 <= a < n_atoms:
                scores[a] += float(row[score_key])
                hits[a] += 1.0
    hits[hits == 0] = 1.0
    return scores / hits


# -----------------------------------------------------------------------------
# C) PIG relation explanation: direct edge occlusion + auxiliary attention
# -----------------------------------------------------------------------------
def _relation_occlusion(
    model,
    batch: Dict,
    item: Dict,
    full_prob: float,
    full_logit: float,
    full_out: Dict,
) -> List[Dict]:
    n_atoms = int(item["n_atoms"])
    n_motifs = int(item["n_motifs"])
    relation = batch["relation"][0].detach().cpu().numpy()
    pair_mask = batch["pair_mask"][0].detach().cpu().numpy().astype(bool)

    attention = full_out.get("hierarchical_attention")
    if attention is not None and torch.is_tensor(attention) and attention.numel() > 0:
        attn_mean = attention[0].mean(dim=0).detach().cpu().numpy()
    else:
        attn_mean = None

    rows: List[Dict] = []
    for i in range(n_motifs):
        ni = n_atoms + i
        for j in range(i + 1, n_motifs):
            nj = n_atoms + j
            if not pair_mask[ni, nj] and not pair_mask[nj, ni]:
                continue
            masked_prob, masked_logit, _ = _prob_and_logit(model, _mask_pair_edge(batch, ni, nj))
            contribution = full_prob - masked_prob
            aux_attn = 0.0
            if attn_mean is not None:
                aux_attn = 0.5 * (float(attn_mean[ni, nj]) + float(attn_mean[nj, ni]))
            rows.append({
                "motif_i": i,
                "motif_j": j,
                "node_i": ni,
                "node_j": nj,
                "label_i": str(item["motif_names"][i]),
                "label_j": str(item["motif_names"][j]),
                "family_i": _family_short(int(item["motif_families"][i])),
                "family_j": _family_short(int(item["motif_families"][j])),
                "relation_id": int(relation[ni, nj]),
                "relation_name": REL_NAMES.get(int(relation[ni, nj]), f"rel_{int(relation[ni, nj])}"),
                "motif_topological_distance": int(batch["pair_dist"][0, ni, nj].item()),
                "p_blocker_full": full_prob,
                "p_blocker_masked": masked_prob,
                "probability_contribution": contribution,
                "logit_contribution": full_logit - masked_logit,
                "aux_last_layer_attention": aux_attn,
                "direction": "blocker" if contribution > 0 else "non_blocker" if contribution < 0 else "neutral",
            })
    rows.sort(key=lambda r: (-abs(float(r["probability_contribution"])), int(r["motif_i"]), int(r["motif_j"])))
    return rows


def draw_pig_graph(
    motif_rows: List[Dict],
    relation_rows: List[Dict],
    out_path: Path,
    title: str,
    dpi: int,
    edge_top_k: int,
) -> None:
    G = nx.Graph()
    for row in motif_rows:
        mid = int(row["motif_id"])
        G.add_node(
            mid,
            score=float(row["probability_contribution"]),
            label=f"{row['motif_name']}\n{row['motif_family_short']}",
        )
    for row in relation_rows[:edge_top_k]:
        G.add_edge(
            int(row["motif_i"]),
            int(row["motif_j"]),
            score=float(row["probability_contribution"]),
            attention=float(row["aux_last_layer_attention"]),
        )

    fig = plt.figure(figsize=(10, 8))
    if len(G) == 0:
        plt.text(0.5, 0.5, "No motif graph available", ha="center", va="center")
        plt.axis("off")
    else:
        pos = nx.kamada_kawai_layout(G)
        node_scores = np.asarray([G.nodes[n]["score"] for n in G.nodes()], dtype=float)
        max_abs_node = max(float(np.max(np.abs(node_scores))) if node_scores.size else 0.0, 1e-8)
        node_colors = []
        node_sizes = []
        for n in G.nodes():
            s = float(G.nodes[n]["score"])
            mag = min(1.0, abs(s) / max_abs_node)
            strength = 0.18 + 0.82 * mag
            if s > 0:
                c = (1.0, 1.0 - strength, 1.0 - strength)
            elif s < 0:
                c = (1.0 - strength, 1.0 - strength, 1.0)
            else:
                c = (0.92, 0.92, 0.92)
            node_colors.append(c)
            node_sizes.append(1200 + 3000 * mag)

        edge_scores = np.asarray([G.edges[e]["score"] for e in G.edges()], dtype=float) if G.number_of_edges() else np.zeros((0,))
        max_abs_edge = max(float(np.max(np.abs(edge_scores))) if edge_scores.size else 0.0, 1e-8)
        edge_widths, edge_colors = [], []
        for e in G.edges():
            s = float(G.edges[e]["score"])
            mag = min(1.0, abs(s) / max_abs_edge)
            edge_widths.append(1.0 + 7.0 * mag)
            edge_colors.append((0.8, 0.1, 0.1) if s > 0 else (0.1, 0.2, 0.8) if s < 0 else (0.7, 0.7, 0.7))

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, linewidths=1.4, edgecolors="black")
        nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=8)
        plt.title(title)
        plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Export rows
# -----------------------------------------------------------------------------
def _atom_rows(row_id: int, smiles: str, label: Optional[int], pred: int, case: str, p_blocker: float, atom_scores: np.ndarray) -> List[Dict]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES during atom export: {smiles!r}")
    norm = _normalize_signed(atom_scores)
    abs_total = float(np.abs(atom_scores).sum())
    rows: List[Dict] = []
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        raw = float(atom_scores[i])
        rows.append({
            "row_id": int(row_id),
            "smiles": smiles,
            "label": np.nan if label is None else int(label),
            "prediction": pred,
            "case": case,
            "p_blocker": p_blocker,
            "atom_idx": i,
            "symbol": atom.GetSymbol(),
            "atomic_num": atom.GetAtomicNum(),
            "formal_charge": atom.GetFormalCharge(),
            "is_aromatic": int(atom.GetIsAromatic()),
            "degree": atom.GetDegree(),
            "ig_score_blocker_logit": raw,
            "ig_score_normalized": float(norm[i]),
            "abs_attribution_fraction": abs(raw) / abs_total if abs_total > 0 else 0.0,
            "direction": "blocker" if raw > 0 else "non_blocker" if raw < 0 else "neutral",
        })
    return rows


def _annotate_rows(rows: List[Dict], row_id: int, smiles: str, label: Optional[int], pred: int, case: str) -> None:
    for row in rows:
        row.update({
            "row_id": int(row_id),
            "smiles": smiles,
            "label": np.nan if label is None else int(label),
            "prediction": int(pred),
            "case": case,
        })


# -----------------------------------------------------------------------------
# Article panels
# -----------------------------------------------------------------------------
def _combine_4panel(paths: Sequence[Path], titles: Sequence[str], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (ax, path, _title) in enumerate(zip(axes.flat, paths, titles)):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(panel_labels[idx], fontsize=12, fontweight="bold", loc="left")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _combine_side_by_side(left: Path, right: Path, left_title: str, right_title: str, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, path, panel in [(axes[0], left, "(a)"), (axes[1], right, "(b)")]:
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(panel, fontsize=12, fontweight="bold", loc="left")
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# One-molecule full pipeline
# -----------------------------------------------------------------------------
def explain_one(
    model,
    feat_cfg,
    scaler,
    device: torch.device,
    smiles: str,
    label: Optional[int],
    row_id: int,
    threshold: float,
    ig_steps: int,
    consistency_tol: float,
    out_dir: Path,
    figure_width: int,
    figure_height: int,
    dpi: int,
    edge_top_k: int,
    show_atom_scores: bool,
) -> Tuple[MoleculeSummary, List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    item, can = _build_single_item(smiles, label, row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    full_prob, full_logit, full_out = _prob_and_logit(model, batch)
    pred = int(full_prob >= threshold)
    case = _case_name(label, pred)

    # A) atom IG
    ig = integrated_gradients_atom_embeddings(model, batch, item, ig_steps, consistency_tol)
    atom_scores = np.asarray(ig["atom_scores"], dtype=np.float32)
    atom_rows = _atom_rows(row_id, can, label, pred, case, full_prob, atom_scores)

    # B) fragments, motifs, families
    fragment_rows = _fragment_occlusion(model, batch, item, full_prob, full_logit, ig["full_atom_chem"])
    motif_rows, family_rows = _motif_and_family_occlusion(model, batch, item, full_prob, full_logit)

    # C) relations - direct edge occlusion, attention only auxiliary
    relation_rows = _relation_occlusion(model, batch, item, full_prob, full_logit, full_out)

    for rows in (fragment_rows, motif_rows, family_rows, relation_rows):
        _annotate_rows(rows, row_id, can, label, pred, case)

    n_atoms = int(item["n_atoms"])
    fragment_atom_scores = _project_group_scores_to_atoms(fragment_rows, n_atoms, "probability_contribution")
    motif_atom_scores = _project_group_scores_to_atoms(motif_rows, n_atoms, "probability_contribution")

    fig_dir = out_dir / "figures"
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    atom_fig = fig_dir / _safe_name(f"row_{row_id}_{case}_A_atom_ig.png")
    frag_fig = fig_dir / _safe_name(f"row_{row_id}_{case}_B_fragment_occlusion.png")
    motif_fig = fig_dir / _safe_name(f"row_{row_id}_{case}_C_motif_occlusion.png")
    pig_fig = fig_dir / _safe_name(f"row_{row_id}_{case}_D_pig_relations.png")
    case_fig = fig_dir / _safe_name(f"row_{row_id}_{case}_E_case_study_4panel.png")

    draw_signed_atom_heatmap(
        can, atom_scores, atom_fig,
        f"Atom IG | p(blocker)={full_prob:.3f} | {case} | red=blocker, blue=non-blocker",
        figure_width, figure_height, dpi, show_atom_scores,
    )
    draw_signed_atom_heatmap(
        can, fragment_atom_scores, frag_fig,
        f"BRICS fragment feature occlusion | p(blocker)={full_prob:.3f} | {case}",
        figure_width, figure_height, dpi, False,
    )
    draw_signed_atom_heatmap(
        can, motif_atom_scores, motif_fig,
        f"Motif/pharmacophore node occlusion | p(blocker)={full_prob:.3f} | {case}",
        figure_width, figure_height, dpi, False,
    )
    draw_pig_graph(
        motif_rows, relation_rows, pig_fig,
        f"PIG direct edge occlusion | p(blocker)={full_prob:.3f} | {case}",
        dpi, edge_top_k,
    )
    _combine_4panel(
        [atom_fig, frag_fig, motif_fig, pig_fig],
        ["A. Atom Integrated Gradients", "B. BRICS fragment occlusion", "C. Motif/pharmacophore occlusion", "D. PIG relation occlusion"],
        case_fig,
        dpi,
    )

    np.savez_compressed(
        raw_dir / _safe_name(f"row_{row_id}_{case}_atom_ig.npz"),
        atom_embedding_ig=np.asarray(ig["embedding_attributions"], dtype=np.float32),
        atom_scores=atom_scores,
        smiles=np.asarray(can),
        row_id=np.asarray(row_id),
        blocker_logit=np.asarray(full_logit, dtype=np.float32),
        baseline_blocker_logit=np.asarray(float(ig["baseline_logit"]), dtype=np.float32),
    )

    top_atom_idx = int(np.argmax(np.abs(atom_scores))) if atom_scores.size else -1
    top_atom_score = float(atom_scores[top_atom_idx]) if top_atom_idx >= 0 else 0.0
    top_fragment = fragment_rows[0] if fragment_rows else {"fragment_smiles": "NA", "probability_contribution": 0.0}
    top_motif = motif_rows[0] if motif_rows else {"motif_name": "NA", "probability_contribution": 0.0}
    top_family = family_rows[0] if family_rows else {"motif_family": "NA", "probability_contribution": 0.0}
    top_relation = relation_rows[0] if relation_rows else {"label_i": "NA", "label_j": "NA", "probability_contribution": 0.0}

    summary = MoleculeSummary(
        row_id=int(row_id),
        smiles=can,
        label=None if label is None else int(label),
        prediction=pred,
        case=case,
        threshold=float(threshold),
        p_blocker=full_prob,
        blocker_logit=full_logit,
        baseline_p_blocker=float(ig["baseline_prob"]),
        baseline_blocker_logit=float(ig["baseline_logit"]),
        ig_sum=float(ig["ig_sum"]),
        ig_completeness_delta=float(ig["completeness_delta"]),
        ig_relative_error=float(ig["relative_error"]),
        forward_consistency_abs_error=float(ig["forward_consistency_abs_error"]),
        n_atoms=int(item["n_atoms"]),
        n_motifs=int(item["n_motifs"]),
        top_atom_idx=top_atom_idx,
        top_atom_score=top_atom_score,
        top_fragment=str(top_fragment.get("fragment_smiles", "NA")),
        top_fragment_contribution=float(top_fragment.get("probability_contribution", 0.0)),
        top_motif=str(top_motif.get("motif_name", "NA")),
        top_motif_contribution=float(top_motif.get("probability_contribution", 0.0)),
        top_family=str(top_family.get("motif_family", "NA")),
        top_family_contribution=float(top_family.get("probability_contribution", 0.0)),
        top_relation=f"{top_relation.get('label_i', 'NA')} <-> {top_relation.get('label_j', 'NA')}",
        top_relation_contribution=float(top_relation.get("probability_contribution", 0.0)),
        atom_figure=str(atom_fig),
        fragment_figure=str(frag_fig),
        motif_figure=str(motif_fig),
        pig_figure=str(pig_fig),
        case_study_figure=str(case_fig),
    )
    return summary, atom_rows, fragment_rows, motif_rows, family_rows, relation_rows


# -----------------------------------------------------------------------------
# D) Scalable global analysis over many molecules
# -----------------------------------------------------------------------------
def global_occlusion_one(model, feat_cfg, scaler, device, record: Record, threshold: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Scalable global pass: motifs + families + direct PIG relations (no IG)."""
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    full_prob, full_logit, full_out = _prob_and_logit(model, batch)
    pred = int(full_prob >= threshold)
    case = _case_name(record.label, pred)
    motif_rows, family_rows = _motif_and_family_occlusion(model, batch, item, full_prob, full_logit)
    relation_rows = _relation_occlusion(model, batch, item, full_prob, full_logit, full_out)
    for rows in (motif_rows, family_rows, relation_rows):
        _annotate_rows(rows, record.row_id, can, record.label, pred, case)
    return motif_rows, family_rows, relation_rows


def _atom_type_summary(atom_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if atom_df.empty:
        return pd.DataFrame()
    df = atom_df.copy()
    df["abs_ig"] = df["ig_score_blocker_logit"].abs()
    out = (
        df.groupby(["case", "symbol"], dropna=False)
        .agg(
            n_atoms=("symbol", "size"),
            mean_ig=("ig_score_blocker_logit", "mean"),
            mean_abs_ig=("abs_ig", "mean"),
            median_abs_ig=("abs_ig", "median"),
            blocker_fraction=("direction", lambda s: float((s == "blocker").mean())),
        )
        .reset_index()
        .sort_values(["case", "mean_abs_ig", "n_atoms"], ascending=[True, False, False])
    )
    return out.groupby("case", group_keys=False).head(top_k).reset_index(drop=True)


def _fragment_summary(fragment_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if fragment_df.empty:
        return pd.DataFrame()
    df = fragment_df.copy()
    df["abs_contribution"] = df["probability_contribution"].abs()
    out = (
        df.groupby(["case", "fragment_smiles"], dropna=False)
        .agg(
            n_occurrences=("fragment_smiles", "size"),
            mean_contribution=("probability_contribution", "mean"),
            mean_abs_contribution=("abs_contribution", "mean"),
            median_abs_contribution=("abs_contribution", "median"),
        )
        .reset_index()
        .sort_values(["case", "mean_abs_contribution", "n_occurrences"], ascending=[True, False, False])
    )
    return out.groupby("case", group_keys=False).head(top_k).reset_index(drop=True)


def _family_summary(family_df: pd.DataFrame) -> pd.DataFrame:
    if family_df.empty:
        return pd.DataFrame()
    df = family_df.copy()
    df["abs_contribution"] = df["probability_contribution"].abs()
    return (
        df.groupby(["case", "motif_family_short", "motif_family"], dropna=False)
        .agg(
            n_rows=("motif_family", "size"),
            mean_contribution=("probability_contribution", "mean"),
            mean_abs_contribution=("abs_contribution", "mean"),
            median_abs_contribution=("abs_contribution", "median"),
        )
        .reset_index()
        .sort_values(["case", "mean_abs_contribution"], ascending=[True, False])
    )


def _family_enrichment(family_df: pd.DataFrame) -> pd.DataFrame:
    if family_df.empty:
        return pd.DataFrame()
    pivot = (
        family_df.groupby(["case", "motif_family_short", "motif_family"], dropna=False)["probability_contribution"]
        .mean().reset_index()
        .pivot_table(index=["motif_family_short", "motif_family"], columns="case", values="probability_contribution", fill_value=0.0)
        .reset_index()
    )
    for col in ["TP", "TN", "FP", "FN"]:
        if col not in pivot.columns:
            pivot[col] = 0.0
    pivot["tp_minus_tn"] = pivot["TP"] - pivot["TN"]
    pivot["tp_minus_fn"] = pivot["TP"] - pivot["FN"]
    pivot["blocker_support_rank"] = pivot["tp_minus_tn"].abs()
    return pivot.sort_values("blocker_support_rank", ascending=False)


def _relation_summary(relation_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if relation_df.empty:
        return pd.DataFrame()
    df = relation_df.copy()
    df["abs_contribution"] = df["probability_contribution"].abs()
    df["family_pair"] = df.apply(lambda r: "-".join(sorted([str(r["family_i"]), str(r["family_j"])])), axis=1)
    out = (
        df.groupby(["case", "relation_name", "family_pair"], dropna=False)
        .agg(
            n_edges=("relation_name", "size"),
            mean_contribution=("probability_contribution", "mean"),
            mean_abs_contribution=("abs_contribution", "mean"),
            mean_aux_attention=("aux_last_layer_attention", "mean"),
        )
        .reset_index()
        .sort_values(["case", "mean_abs_contribution", "n_edges"], ascending=[True, False, False])
    )
    return out.groupby("case", group_keys=False).head(top_k).reset_index(drop=True)


def _write_global_report(out_dir: Path, atom_summary: pd.DataFrame, frag_summary: pd.DataFrame, fam_enrich: pd.DataFrame, rel_summary: pd.DataFrame) -> None:
    lines: List[str] = ["# RHMGT Global Interpretability Report", ""]
    lines += ["## Atom-level highlights"]
    if atom_summary.empty:
        lines.append("- No atom IG aggregate available.")
    else:
        for case in sorted(atom_summary["case"].dropna().unique()):
            sub = atom_summary[atom_summary["case"] == case].head(5)
            items = ", ".join(f"{r.symbol} (mean|IG|={r.mean_abs_ig:.4f})" for r in sub.itertuples(index=False))
            lines.append(f"- {case}: {items}")
    lines += ["", "## Fragment-level highlights"]
    if frag_summary.empty:
        lines.append("- No fragment aggregate available.")
    else:
        for case in sorted(frag_summary["case"].dropna().unique()):
            sub = frag_summary[frag_summary["case"] == case].head(5)
            items = ", ".join(f"{r.fragment_smiles} (Δp={r.mean_contribution:+.4f})" for r in sub.itertuples(index=False))
            lines.append(f"- {case}: {items}")
    lines += ["", "## Pharmacophore-family enrichment"]
    if fam_enrich.empty:
        lines.append("- No family aggregate available.")
    else:
        for r in fam_enrich.head(8).itertuples(index=False):
            lines.append(f"- {r.motif_family_short} / {r.motif_family}: TP-TN={r.tp_minus_tn:+.4f}; TP-FN={r.tp_minus_fn:+.4f}")
    lines += ["", "## PIG relation patterns"]
    if rel_summary.empty:
        lines.append("- No relation aggregate available.")
    else:
        for case in sorted(rel_summary["case"].dropna().unique()):
            sub = rel_summary[rel_summary["case"] == case].head(5)
            items = ", ".join(f"{r.relation_name}:{r.family_pair} (Δp={r.mean_contribution:+.4f})" for r in sub.itertuples(index=False))
            lines.append(f"- {case}: {items}")
    lines += [
        "",
        "## Interpretation hierarchy used",
        "1. Atom Integrated Gradients (signed, completeness checked)",
        "2. BRICS fragment feature occlusion",
        "3. Motif/pharmacophore node occlusion",
        "4. Pharmacophore-family occlusion",
        "5. Direct motif-motif edge occlusion (attention is auxiliary only)",
        "6. TP/TN/FP/FN global aggregation",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_global_summary_figure(family_enrich: pd.DataFrame, relation_summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if family_enrich.empty:
        axes[0].text(0.5, 0.5, "No family enrichment", ha="center", va="center")
        axes[0].axis("off")
    else:
        sub = family_enrich.head(8).copy()
        axes[0].barh(range(len(sub)), sub["tp_minus_tn"].values)
        axes[0].set_yticks(range(len(sub)))
        axes[0].set_yticklabels(sub["motif_family_short"].astype(str).tolist())
        axes[0].axvline(0.0, linewidth=1)
        axes[0].set_xlabel("Mean family contribution: TP - TN")
        axes[0].set_title("(a)", fontweight="bold", loc="left")
        axes[0].invert_yaxis()

    if relation_summary.empty:
        axes[1].text(0.5, 0.5, "No relation summary", ha="center", va="center")
        axes[1].axis("off")
    else:
        rel = relation_summary.copy()
        rel["label"] = rel["relation_name"].astype(str) + " | " + rel["family_pair"].astype(str)
        rel = rel.sort_values("mean_abs_contribution", ascending=False).head(10)
        axes[1].barh(range(len(rel)), rel["mean_contribution"].values)
        axes[1].set_yticks(range(len(rel)))
        axes[1].set_yticklabels(rel["label"].tolist(), fontsize=8)
        axes[1].axvline(0.0, linewidth=1)
        axes[1].set_xlabel("Mean direct edge-occlusion contribution Δp")
        axes[1].set_title("(b)", fontweight="bold", loc="left")
        axes[1].invert_yaxis()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
# =============================================================================
# Q1-grade branch-specific screening / selection / figures
# =============================================================================

def parse_args_q1() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Publication-grade hierarchical RHMGT interpretability: shared-molecule hierarchical screening "
            "and figures for atoms, BRICS fragments, hierarchical motif nodes, explicit pharmacophores, PIG "
            "relations, global summaries, and faithfulness."
        )
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-csv", "--test-csv", dest="eval_csv", default=None,
                   help="Frozen test/evaluation CSV, e.g. data/paper_split/hERGAT_test_df.csv")
    p.add_argument("--train-csv", default=None,
                   help="Training CSV used only to exclude exact train-overlap showcase molecules")
    p.add_argument("--valid-csv", default=None,
                   help="Validation CSV. Loaded only to document/verify the frozen split; it is NOT used to select interpretation examples or tune thresholds.")
    p.add_argument("--csv", default=None, help="Full dataset CSV; used only with --split-json")
    p.add_argument("--split-json", default=None, help="Optional row-index split JSON for an explicitly reconstructed split")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--smiles-col", default="SMILES")
    p.add_argument("--label-col", default="Class")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cache-dir", default=".cache/hera_hgt_interpretability_q1")
    p.add_argument("--out-dir", default="interpretability/q1")
    p.add_argument("--finalize-existing", action="store_true",
                   help="Reuse existing screening/selection/results in --out-dir and only regenerate report/metadata; avoids rerunning expensive interpretation passes after a late reporting error.")
    p.add_argument("--render-existing", action="store_true",
                   help="Regenerate the publication figures from existing CSV/NPZ outputs in --out-dir without rerunning the model. Useful after changing figure design.")
    p.add_argument("--render-global-only", action="store_true",
                   help="Regenerate only Fig. 5A/5B/5C from cached global CSVs in --out-dir. Does not require showcase row IDs and does not rerun the model.")
    p.add_argument("--selected-row-ids", nargs="*", type=int, default=None,
                   help="Optional explicit showcase row IDs used by --render-existing/repair mode. Example: --selected-row-ids 556 1345 1157 511")
    p.add_argument("--repair-selected", action="store_true",
                   help="With --render-existing, recompute only the selected showcase molecules when cached final_* CSV/NPZ files are missing. This does NOT rerun the full 1,424-molecule screening/global pass.")

    # Screening and final atom IG are intentionally separated. A modest number
    # of steps is sufficient for numerical screening, while the publication
    # maps are recomputed more accurately after molecule selection.
    p.add_argument("--screen-per-class", type=int, default=120)
    p.add_argument("--screen-ig-steps", type=int, default=12)
    p.add_argument("--final-ig-steps", type=int, default=96)
    p.add_argument("--branch-per-class", type=int, default=2,
                   help="2 gives 4 molecules/branch: 2 TP + 2 TN")
    p.add_argument("--max-tanimoto", type=float, default=0.70)
    p.add_argument("--min-atoms", type=int, default=8)
    p.add_argument("--max-atoms", type=int, default=65)
    p.add_argument("--allow-train-overlap", action="store_true",
                   help="By default showcase candidates with exact train overlap are excluded")

    # Global quantitative pass. Relation effects are measured only for the top
    # attention-screened explicit pairs per molecule, then validated by direct
    # perturbation; attention itself is not reported as importance.
    p.add_argument("--skip-global", action="store_true")
    p.add_argument("--global-max-molecules", type=int, default=0,
                   help="0 = all molecules in the selected official partition")
    p.add_argument("--global-relation-candidates", type=int, default=5)
    p.add_argument("--global-atom-per-class", type=int, default=30)
    p.add_argument("--global-atom-ig-steps", type=int, default=24)

    # Attention-ranking faithfulness (supplementary diagnostic).
    p.add_argument("--skip-faithfulness", action="store_true")
    p.add_argument("--faithfulness-molecules", type=int, default=80)
    p.add_argument("--faithfulness-controls", type=int, default=20)

    # Primary explanation faithfulness (main-paper quantitative validation).
    # Ranking comes from the primary explanation method at each level:
    # signed atom IG, direct motif-node occlusion, and direct PIG-edge deletion.
    # The validation outcome is cumulative top-k deletion versus matched random
    # perturbation, measured in predicted-class logit support.
    p.add_argument("--skip-primary-faithfulness", action="store_true")
    p.add_argument("--primary-faithfulness-molecules", type=int, default=80)
    p.add_argument("--primary-faithfulness-controls", type=int, default=20)
    p.add_argument("--primary-faithfulness-ig-steps", type=int, default=32)

    # Adaptive fusion diagnostics (supplementary).
    p.add_argument("--skip-fusion-gates", action="store_true")

    p.add_argument("--bootstrap-reps", type=int, default=2000)

    # Visual output.
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--top-fragments", type=int, default=4)
    p.add_argument("--top-pharmacophores", type=int, default=4)
    p.add_argument("--top-relations", type=int, default=3)
    return p.parse_args()


def _raw_atom_forward(model, batch: Dict, atom_raw: torch.Tensor, motif_atoms: Sequence[Sequence[int]]) -> Dict[str, torch.Tensor]:
    """Run the exact model after replacing raw atom features.

    Motif raw chemistry rows are recomputed as the mean of the perturbed member
    atoms. This makes the IG path consistent with RHMGT's atom->motif
    initialization and makes a zero baseline a genuine zero *raw atom-feature*
    baseline instead of a zero projected-embedding baseline.
    """
    if atom_raw.ndim != 2:
        raise ValueError("atom_raw must be [n_atoms, atom_dim]")
    if batch["node_x"].shape[0] != 1:
        raise ValueError("raw-atom attribution expects batch size 1")
    n_atoms = int(atom_raw.shape[0])
    global_index = int(batch["global_index"][0].item())
    expected_motifs = global_index - n_atoms
    if expected_motifs != len(motif_atoms):
        raise RuntimeError(
            f"Motif mapping mismatch: graph={expected_motifs}, metadata={len(motif_atoms)}"
        )

    motif_rows: List[torch.Tensor] = []
    for atoms in motif_atoms:
        ids = torch.as_tensor(list(atoms), dtype=torch.long, device=atom_raw.device)
        if ids.numel() == 0:
            raise RuntimeError("Empty motif")
        motif_rows.append(atom_raw.index_select(0, ids).mean(dim=0, keepdim=True))
    motif_raw = torch.cat(motif_rows, dim=0) if motif_rows else atom_raw.new_zeros((0, atom_raw.shape[1]))
    global_raw = atom_raw.new_zeros((1, atom_raw.shape[1]))
    node_x = torch.cat([atom_raw, motif_raw, global_raw], dim=0).unsqueeze(0)

    b = dict(batch)
    b["node_x"] = node_x
    return model(b)


def integrated_gradients_raw_atoms(
    model,
    batch: Dict,
    item: Dict,
    steps: int,
    consistency_tol: float = 1e-4,
) -> Dict[str, object]:
    if steps < 4:
        raise ValueError("IG steps must be >= 4")
    n_atoms = int(item["n_atoms"])
    motif_atoms = item["motif_atoms"]
    with torch.no_grad():
        full_raw = batch["node_x"][0, :n_atoms].detach().clone()
        normal = model(batch)
        normal_logit = float(normal["logits"][0].item())
        normal_prob = float(normal["prob"][0].item())
        exact = _raw_atom_forward(model, batch, full_raw, motif_atoms)
        exact_logit = float(exact["logits"][0].item())
        forward_err = abs(exact_logit - normal_logit)
        if forward_err > consistency_tol:
            raise RuntimeError(
                f"raw-input attribution forward mismatch: {forward_err:.6g} > {consistency_tol:.6g}"
            )
        baseline = torch.zeros_like(full_raw)
        base_out = _raw_atom_forward(model, batch, baseline, motif_atoms)
        base_logit = float(base_out["logits"][0].item())
        base_prob = float(base_out["prob"][0].item())

    diff = full_raw
    grad_sum = torch.zeros_like(full_raw)
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=full_raw.device, dtype=full_raw.dtype)
    for i, alpha in enumerate(alphas):
        x = (alpha * full_raw).detach().requires_grad_(True)
        out = _raw_atom_forward(model, batch, x, motif_atoms)
        grad = torch.autograd.grad(out["logits"][0], x, retain_graph=False, create_graph=False)[0]
        grad_sum += (0.5 if i in (0, steps) else 1.0) * grad.detach()
    ig = diff * (grad_sum / float(steps))
    atom_scores = ig.sum(dim=-1)
    ig_sum = float(atom_scores.sum().item())
    output_delta = normal_logit - base_logit
    completeness_delta = ig_sum - output_delta
    rel_error = abs(completeness_delta) / max(abs(output_delta), 1e-8)
    return {
        "feature_attributions": ig.detach().cpu().numpy().astype(np.float32),
        "atom_scores": atom_scores.detach().cpu().numpy().astype(np.float32),
        "normal_logit": normal_logit,
        "normal_prob": normal_prob,
        "baseline_logit": base_logit,
        "baseline_prob": base_prob,
        "ig_sum": ig_sum,
        "output_delta": output_delta,
        "completeness_delta": completeness_delta,
        "relative_error": rel_error,
        "forward_consistency_abs_error": forward_err,
    }


def _neutralize_pair_relation(batch: Dict, i: int, j: int) -> Dict:
    """Keep connectivity/distance but replace the pair's typed relation by generic."""
    out = _clone_batch(batch)
    generic_rel = 5  # REL_MM_GENERIC in hera_hgt.hierarchy
    for a, b in ((i, j), (j, i)):
        out["relation"][0, a, b] = generic_rel
        # Boundary-bond chemistry can encode pair-specific covalent contact.
        # For a relation-type sensitivity test it is retained; distance is also
        # retained. Only the discrete typed relation is neutralized.
    return out


def _assign_persistent_pharmacophore_ids(motif_rows: Sequence[Dict]) -> List[Dict]:
    """Assign deterministic P1, P2, ... identifiers from the model motif_id.

    The identifier is intentionally independent of attribution rank.  This makes
    the same explicit occurrence keep the same P label in Fig. 3, Fig. 4, CSVs,
    and any later manuscript discussion.  Attribution rank may change; identity
    must not.
    """
    rows = [dict(r) for r in motif_rows]
    ordered_ids = sorted({int(r["motif_id"]) for r in rows if "motif_id" in r})
    label_by_id = {mid: f"P{i+1}" for i, mid in enumerate(ordered_ids)}
    for r in rows:
        if "motif_id" not in r:
            continue
        mid = int(r["motif_id"])
        r["pharmacophore_index"] = int(ordered_ids.index(mid) + 1)
        r["pharmacophore_label"] = label_by_id[mid]
    return rows


def _explicit_motif_rows(motif_rows: Sequence[Dict]) -> List[Dict]:
    explicit = [dict(r) for r in motif_rows if int(r.get("motif_family_id", 0)) in (1, 2, 3, 4)]
    return _assign_persistent_pharmacophore_ids(explicit)


def _explicit_relation_rows(rows: Sequence[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        if str(r.get("family_i", "STR")) != "STR" and str(r.get("family_j", "STR")) != "STR":
            out.append(dict(r))
    return out


def _add_motif_attention(motif_rows: List[Dict], full_out: Dict, item: Dict) -> None:
    attn = full_out.get("hierarchical_attention")
    if not torch.is_tensor(attn) or attn.numel() == 0:
        for r in motif_rows:
            r["aux_global_to_motif_attention"] = 0.0
        return
    a = attn[0].mean(dim=0).detach().cpu().numpy()
    g = int(item["n_atoms"] + item["n_motifs"])
    n_atoms = int(item["n_atoms"])
    for r in motif_rows:
        node_id = n_atoms + int(r["motif_id"])
        r["aux_global_to_motif_attention"] = float(a[g, node_id])


def _add_relation_neutralization(model, batch: Dict, relation_rows: List[Dict], full_logit: float) -> None:
    for r in relation_rows:
        ni, nj = int(r["node_i"]), int(r["node_j"])
        _, z, _ = _prob_and_logit(model, _neutralize_pair_relation(batch, ni, nj))
        r["relation_type_neutralization_logit"] = float(full_logit - z)


def _directional_mass(values: Sequence[float], prediction: int) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    denom = float(np.abs(arr).sum())
    if denom <= 0:
        return 0.0
    mask = arr > 0 if int(prediction) == 1 else arr < 0
    return float(np.abs(arr[mask]).sum() / denom)


def _top_mass_fraction(values: Sequence[float], fraction: float = 0.20) -> float:
    arr = np.abs(np.asarray(values, dtype=float))
    if arr.size == 0 or float(arr.sum()) <= 0:
        return 0.0
    k = max(1, int(math.ceil(arr.size * fraction)))
    return float(np.sort(arr)[-k:].sum() / arr.sum())


def _screen_one(model, feat_cfg, scaler, device: torch.device, record: Record, threshold: float, ig_steps: int, consistency_tol: float = 1e-4) -> Dict:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    full_prob, full_logit, full_out = _prob_and_logit(model, batch)
    pred = int(full_prob >= threshold)
    case = _case_name(record.label, pred)

    ig = integrated_gradients_raw_atoms(model, batch, item, ig_steps, consistency_tol)
    atom_scores = np.asarray(ig["atom_scores"], dtype=float)

    with torch.no_grad():
        projected_atom = model.graph_encoder.node_proj(batch["node_x"][:, : int(item["n_atoms"])]).squeeze(0)
    frag = _fragment_occlusion(model, batch, item, full_prob, full_logit, projected_atom)
    motifs, families = _motif_and_family_occlusion(model, batch, item, full_prob, full_logit)
    _add_motif_attention(motifs, full_out, item)
    rel = _relation_occlusion(model, batch, item, full_prob, full_logit, full_out)

    # Re-sort by logit effect; probability can saturate around 0/1 and hide a
    # real model effect.
    frag.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    motifs.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    families.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    rel.sort(key=lambda r: -abs(float(r["logit_contribution"])))

    explicit_m = _explicit_motif_rows(motifs)
    explicit_r = _explicit_relation_rows(rel)
    fams = sorted(set(int(r["motif_family_id"]) for r in explicit_m))

    frag_vals = [float(r["logit_contribution"]) for r in frag if str(r.get("fragment_name")) != "whole_molecule"]
    motif_vals = [float(r["logit_contribution"]) for r in explicit_m]
    rel_vals = [float(r["logit_contribution"]) for r in explicit_r]

    abs_atom = np.abs(atom_scores)
    out = {
        "row_id": int(record.row_id),
        "smiles": can,
        "label": int(record.label),
        "prediction": pred,
        "case": case,
        "p_blocker": full_prob,
        "blocker_logit": full_logit,
        "confidence_from_threshold": abs(full_prob - threshold),
        "n_atoms": int(item["n_atoms"]),
        "n_motifs": int(item["n_motifs"]),
        "n_fragments": len(frag_vals),
        "n_explicit_motifs": len(explicit_m),
        "n_explicit_families": len(fams),
        "n_explicit_relations": len(explicit_r),
        "atom_mean_abs": float(abs_atom.mean()) if abs_atom.size else 0.0,
        "atom_max_abs": float(abs_atom.max()) if abs_atom.size else 0.0,
        "atom_focus_top20": _top_mass_fraction(atom_scores, 0.20),
        "atom_directional_mass": _directional_mass(atom_scores, pred),
        "atom_ig_relative_error": float(ig["relative_error"]),
        "atom_forward_error": float(ig["forward_consistency_abs_error"]),
        "fragment_top_abs_logit": max([abs(v) for v in frag_vals], default=0.0),
        "fragment_top2_abs_logit": float(sum(sorted([abs(v) for v in frag_vals], reverse=True)[:2])),
        "fragment_directional_mass": _directional_mass(frag_vals, pred),
        "motif_top_abs_logit": max([abs(v) for v in motif_vals], default=0.0),
        "motif_top3_abs_logit": float(sum(sorted([abs(v) for v in motif_vals], reverse=True)[:3])),
        "motif_directional_mass": _directional_mass(motif_vals, pred),
        "motif_family_diversity": float(len(fams) / 4.0),
        "relation_top_abs_logit": max([abs(v) for v in rel_vals], default=0.0),
        "relation_top3_abs_logit": float(sum(sorted([abs(v) for v in rel_vals], reverse=True)[:3])),
        "relation_directional_mass": _directional_mass(rel_vals, pred),
        "relation_count_score": float(min(len(explicit_r), 8) / 8.0),
    }
    return out


def _rank_pct_by_case(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("case")[col].rank(method="average", pct=True)


def _score_screening(df: pd.DataFrame, min_atoms: int, max_atoms: int) -> pd.DataFrame:
    d = df.copy()
    for col in [
        "atom_mean_abs", "atom_focus_top20", "atom_directional_mass", "confidence_from_threshold",
        "fragment_top_abs_logit", "fragment_top2_abs_logit", "fragment_directional_mass",
        "motif_top_abs_logit", "motif_top3_abs_logit", "motif_directional_mass", "motif_family_diversity",
        "relation_top_abs_logit", "relation_top3_abs_logit", "relation_directional_mass", "relation_count_score",
    ]:
        d[col + "_rank"] = _rank_pct_by_case(d, col)

    d["atom_quality"] = (
        0.30 * d["atom_mean_abs_rank"]
        + 0.20 * d["atom_focus_top20_rank"]
        + 0.35 * d["atom_directional_mass_rank"]
        + 0.15 * d["confidence_from_threshold_rank"]
    ) * np.exp(-5.0 * np.clip(d["atom_ig_relative_error"], 0, 1))

    frag_count_quality = np.where((d["n_fragments"] >= 2) & (d["n_fragments"] <= 10), 1.0, 0.5)
    d["fragment_quality"] = (
        0.35 * d["fragment_top_abs_logit_rank"]
        + 0.25 * d["fragment_top2_abs_logit_rank"]
        + 0.30 * d["fragment_directional_mass_rank"]
        + 0.10 * d["confidence_from_threshold_rank"]
    ) * frag_count_quality

    d["motif_quality"] = (
        0.30 * d["motif_top_abs_logit_rank"]
        + 0.25 * d["motif_top3_abs_logit_rank"]
        + 0.20 * d["motif_family_diversity_rank"]
        + 0.20 * d["motif_directional_mass_rank"]
        + 0.05 * d["confidence_from_threshold_rank"]
    )

    d["relation_quality"] = (
        0.35 * d["relation_top_abs_logit_rank"]
        + 0.25 * d["relation_top3_abs_logit_rank"]
        + 0.15 * d["relation_directional_mass_rank"]
        + 0.15 * d["relation_count_score_rank"]
        + 0.10 * d["motif_family_diversity_rank"]
    )

    readable = (d["n_atoms"] >= min_atoms) & (d["n_atoms"] <= max_atoms)
    d["eligible_atom"] = readable & (d["atom_ig_relative_error"] <= 0.10)
    d["eligible_fragment"] = readable & (d["n_fragments"] >= 2)
    d["eligible_motif"] = readable & (d["n_explicit_motifs"] >= 3) & (d["n_explicit_families"] >= 2)
    d["eligible_relation"] = d["eligible_motif"] & (d["n_explicit_relations"] >= 2)
    return d


def _morgan_fp(smiles: str):
    from rdkit.Chem import rdFingerprintGenerator
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


def _greedy_diverse_select(df: pd.DataFrame, score_col: str, eligible_col: str, n_per_class: int, max_sim: float) -> pd.DataFrame:
    from rdkit import DataStructs
    selected_rows = []
    selected_fps = []
    # round-robin between activity classes avoids letting one class monopolize
    # structural diversity space.
    class_lists = {}
    for case in ("TP", "TN"):
        g = df[(df["case"] == case) & (df[eligible_col])].sort_values([score_col, "row_id"], ascending=[False, True]).copy()
        class_lists[case] = list(g.to_dict("records"))

    thresholds = [max_sim, min(0.80, max_sim + 0.05), min(0.90, max_sim + 0.15), 1.0]
    chosen_ids = set()
    for sim_thr in thresholds:
        progress = True
        while progress:
            progress = False
            for case in ("TP", "TN"):
                already = sum(1 for r in selected_rows if r["case"] == case)
                if already >= n_per_class:
                    continue
                for r in class_lists[case]:
                    if int(r["row_id"]) in chosen_ids:
                        continue
                    fp = _morgan_fp(str(r["smiles"]))
                    if fp is None:
                        continue
                    sims = [DataStructs.TanimotoSimilarity(fp, s) for s in selected_fps]
                    if all(s <= sim_thr for s in sims):
                        rr = dict(r)
                        rr["selection_max_similarity_to_previous"] = max(sims) if sims else 0.0
                        rr["selection_similarity_threshold"] = sim_thr
                        selected_rows.append(rr)
                        selected_fps.append(fp)
                        chosen_ids.add(int(r["row_id"]))
                        progress = True
                        break
            if all(sum(1 for r in selected_rows if r["case"] == c) >= n_per_class for c in ("TP", "TN")):
                break
        if all(sum(1 for r in selected_rows if r["case"] == c) >= n_per_class for c in ("TP", "TN")):
            break
    return pd.DataFrame(selected_rows)


def _select_shared_hierarchy_panel(df: pd.DataFrame, n_per_class: int, max_sim: float) -> pd.DataFrame:
    """Select a publication showcase panel by a prespecified quantitative rule.

    The panel is chosen automatically before visual inspection. It balances:
    (i) readable molecular size, (ii) explicit motif/family richness,
    (iii) non-saturated correct predictions, (iv) atom/motif/relation explanation
    strength from the numerical screening pass, and (v) structural diversity.
    The global analysis still covers the complete test partition, so claims do not
    depend on these qualitative showcase examples.
    """
    d = df.copy()
    tp_band = (d["case"] == "TP") & d["p_blocker"].between(0.60, 0.985)
    tn_band = (d["case"] == "TN") & d["p_blocker"].between(0.015, 0.40)
    d["eligible_shared"] = d.get("eligible_relation", False) & (tp_band | tn_band)

    motif_count = np.clip(d["n_explicit_motifs"].astype(float), 0, 8) / 8.0
    rel_count = np.clip(d["n_explicit_relations"].astype(float), 0, 10) / 10.0
    fam_div = np.clip(d["n_explicit_families"].astype(float), 0, 4) / 4.0
    structural_richness = 0.35 * motif_count + 0.35 * rel_count + 0.30 * fam_div

    # Explanation-aware terms are percentile ranks calculated within each case;
    # they are used only for qualitative showcase selection and never for model
    # training, threshold tuning, or global quantitative claims.
    atom_q = d.get("atom_quality", pd.Series(0.0, index=d.index)).astype(float)
    motif_q = d.get("motif_quality", pd.Series(0.0, index=d.index)).astype(float)
    rel_q = d.get("relation_quality", pd.Series(0.0, index=d.index)).astype(float)
    d["shared_hierarchy_quality"] = (
        0.20 * structural_richness
        + 0.20 * atom_q
        + 0.30 * motif_q
        + 0.30 * rel_q
    )
    selected = _greedy_diverse_select(d, "shared_hierarchy_quality", "eligible_shared", n_per_class, max_sim)
    if len(selected) < 2 * n_per_class:
        d["eligible_shared"] = d.get("eligible_relation", False)
        selected = _greedy_diverse_select(d, "shared_hierarchy_quality", "eligible_shared", n_per_class, max_sim)
    return selected


def _draw_signed_atom_percentile(smiles: str, scores: np.ndarray, out_path: Path, legend: str, dpi: int, percentile: float = 95.0) -> None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    scores = np.asarray(scores, dtype=float)
    scale = float(np.percentile(np.abs(scores), percentile)) if scores.size else 0.0
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(np.abs(scores))) if scores.size else 1.0
    scale = max(scale, 1e-12)
    norm = np.clip(scores / scale, -1.0, 1.0)
    colors, radii, atoms = {}, {}, []
    for i, v in enumerate(norm):
        if abs(v) < 1e-8:
            continue
        mag = abs(float(v))
        strength = 0.20 + 0.80 * mag
        colors[i] = (1.0, 1.0 - strength, 1.0 - strength) if v > 0 else (1.0 - strength, 1.0 - strength, 1.0)
        radii[i] = 0.24 + 0.24 * mag
        atoms.append(i)
    drawer = rdMolDraw2D.MolDraw2DCairo(1700, 1050)
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.continuousHighlight = True
    opts.highlightBondWidthMultiplier = 7
    opts.legendFontSize = 28
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, Chem.Mol(mol), legend=legend,
        highlightAtoms=atoms, highlightAtomColors=colors, highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB").save(out_path, dpi=(dpi, dpi))


FRAGMENT_COLORS = [
    (0.90, 0.35, 0.25), (0.25, 0.55, 0.90), (0.25, 0.70, 0.45), (0.75, 0.45, 0.85),
    (0.95, 0.70, 0.20), (0.20, 0.75, 0.75), (0.65, 0.60, 0.25), (0.55, 0.55, 0.55),
]
FAMILY_COLORS = {
    "BN": (0.91, 0.35, 0.22),
    "AR": (0.95, 0.64, 0.10),
    "NHC": (0.20, 0.48, 0.84),
    "PF": (0.20, 0.65, 0.35),
    "STR": (0.58, 0.58, 0.58),
}
MOTIF_NODE_COLORS = [
    (0.48, 0.36, 0.82),  # M1: violet
    (0.14, 0.63, 0.68),  # M2: teal
    (0.95, 0.58, 0.14),  # M3: amber
    (0.36, 0.47, 0.62),  # M4: slate
]
POS_COLOR = "#D62728"
NEG_COLOR = "#2563EB"
GRID_COLOR = "#D9DEE7"
PANEL_BG = "#FFFFFF"
CASE_TP_BG = "#FCE8E6"
CASE_TN_BG = "#E7F0FF"

FAMILY_LONG = {
    "BN": "basic nitrogen",
    "AR": "aromatic",
    "NHC": "N-containing heterocycle",
    "PF": "peripheral",
    "STR": "structural/BRICS",
}

def _set_publication_style() -> None:
    """Consistent journal-style typography and clean white background."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11.5,
        "axes.titlesize": 13,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save_publication_figure(fig, out_path: Path, dpi: int) -> None:
    """Save PNG plus vector PDF/SVG companions for journal submission."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    stem = out_path.with_suffix("")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)


def _panel_border(ax, case: str, lw: float = 1.35) -> None:
    color = "#E97C7C" if str(case) == "TP" else "#72A9E8" if str(case) == "TN" else "#A8A8A8"
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(color)
        sp.set_linewidth(lw)


def _case_banner_text(case: str, p: float) -> str:
    if str(case) == "TP":
        return f"TP  |  p(blocker) = {p:.3f}  (true blocker)"
    if str(case) == "TN":
        return f"TN  |  p(blocker) = {p:.3f}  (true non-blocker)"
    return _case_display(case, p)

def _case_display(case: str, p_blocker: float) -> str:
    if str(case) == "TP":
        return f"TP | p(blocker)={p_blocker:.3f} | true blocker"
    if str(case) == "TN":
        return f"TN | p(blocker)={p_blocker:.3f} | true non-blocker"
    if str(case) == "FP":
        return f"FP | p(blocker)={p_blocker:.3f}"
    if str(case) == "FN":
        return f"FN | p(blocker)={p_blocker:.3f}"
    return f"{case} | p(blocker)={p_blocker:.3f}"

def _case_bg(case: str) -> str:
    return CASE_TP_BG if str(case) == "TP" else CASE_TN_BG if str(case) == "TN" else "#F2F2F2"

def _clean_relation_name(name: str) -> str:
    x = str(name)
    repl = {
        "mm_generic": "generic motif–motif",
        "mm_bn_ar": "BN–AR",
        "mm_bn_nhc": "BN–NHC",
        "mm_bn_pf": "BN–PF",
        "mm_ar_nhc": "AR–NHC",
        "mm_ar_pf": "AR–PF",
        "mm_nhc_pf": "NHC–PF",
        "mm_same_family": "same-family",
    }
    return repl.get(x, x.replace("_", "–"))


def _compact_relation_display(relation_name: str, family_pair: str) -> str:
    clean = _clean_relation_name(relation_name)
    pair = str(family_pair)
    # Typed cross-family relations often repeat the family pair verbatim
    # (e.g. ``BN–NHC | BN–NHC``).  Keep one label only.
    if clean == pair:
        return pair
    if clean == "same-family":
        return f"same-family | {pair}"
    if clean == "generic motif–motif":
        return f"generic | {pair}"
    return f"{clean} | {pair}"

def _short_text(text: str, n: int = 22) -> str:
    t = str(text)
    return t if len(t) <= n else t[: n - 1] + "…"


def _structural_motif_descriptor(smiles: str, row: Dict, max_len: int = 22) -> str:
    """Return a compact chemically meaningful descriptor for a structural node.

    Structural nodes are frequently named simply ``brics_fragment`` by the model
    metadata.  For the publication figure that generic name is uninformative, so
    the actual atom subset is converted to a canonical fragment SMILES when
    possible.  This changes only the display, never the model or attribution.
    """
    try:
        mol = mol_from_smiles(smiles)
        atom_ids = [int(x) for x in str(row.get("atoms", "")).split() if str(x).strip()]
        if mol is not None and atom_ids:
            frag = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ids, canonical=True)
            if frag:
                return _short_text(frag, max_len)
    except Exception:
        pass
    name = str(row.get("motif_name", "structural motif"))
    return _short_text(name.replace("brics_fragment", "BRICS motif"), max_len)


def _rgb255(color) -> tuple[int, int, int]:
    """Convert an RDKit/matplotlib RGB tuple in [0,1] or a hex string to PIL RGB."""
    if isinstance(color, str):
        s = color.lstrip("#")
        if len(s) == 6:
            return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
        return (100, 100, 100)
    vals = list(color)
    if max(vals) <= 1.0:
        vals = [round(255 * float(v)) for v in vals]
    return tuple(int(max(0, min(255, v))) for v in vals[:3])


def _pil_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for fp in candidates:
        try:
            return ImageFont.truetype(fp, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _boxes_overlap(a, b, pad: float = 8.0) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (
        ax1 + pad < bx0 or bx1 + pad < ax0 or
        ay1 + pad < by0 or by1 + pad < ay0
    )


def _box_intersects_molecule(box, mol_bbox, pad: float = 34.0) -> bool:
    """True when a label box enters the reserved molecular drawing region."""
    x0, y0, x1, y1 = box
    mx0, my0, mx1, my1 = mol_bbox
    expanded = (mx0 - pad, my0 - pad, mx1 + pad, my1 + pad)
    return _boxes_overlap(box, expanded, pad=0.0)


def _leader_target_on_box(anchor_xy, box):
    """Point on the nearest box edge, so leader lines stop at the cartouche."""
    ax, ay = map(float, anchor_xy)
    x0, y0, x1, y1 = map(float, box)
    candidates = [
        (min(max(ax, x0), x1), y0),
        (min(max(ax, x0), x1), y1),
        (x0, min(max(ay, y0), y1)),
        (x1, min(max(ay, y0), y1)),
    ]
    return min(candidates, key=lambda q: (q[0] - ax) ** 2 + (q[1] - ay) ** 2)


def _choose_external_label_box(anchor_xy, mol_bbox, box_wh, canvas_wh, occupied):
    """Place a label STRICTLY outside the molecular bounding box.

    The previous implementation merely moved labels radially away from a motif.
    For long molecules this still left M/P cartouches inside the molecular drawing.
    This version reserves four external lanes (top, bottom, left, right) and never
    accepts a candidate that intersects the molecular bounding box.
    """
    ax, ay = map(float, anchor_xy)
    mx0, my0, mx1, my1 = map(float, mol_bbox)
    bw, bh = map(float, box_wh)
    W, H = map(float, canvas_wh)
    margin = 20.0
    gap = 46.0

    # Prefer the molecular edge that is geometrically closest to the motif.
    side_distance = {
        'left': abs(ax - mx0),
        'right': abs(mx1 - ax),
        'top': abs(ay - my0),
        'bottom': abs(my1 - ay),
    }
    preferred = sorted(side_distance, key=side_distance.get)

    shifts = (0, 70, -70, 135, -135, 205, -205, 275, -275)
    candidates = []

    for side_rank, side in enumerate(preferred):
        for lane in (0, 1):
            lane_gap = gap + lane * (bh + 24)
            for shift in shifts:
                if side in ('top', 'bottom'):
                    cx = min(max(ax + shift, margin + bw / 2), W - margin - bw / 2)
                    if side == 'top':
                        y1 = my0 - lane_gap
                        y0 = y1 - bh
                    else:
                        y0 = my1 + lane_gap
                        y1 = y0 + bh
                    x0, x1 = cx - bw / 2, cx + bw / 2
                else:
                    cy = min(max(ay + shift, margin + bh / 2), H - margin - bh / 2)
                    if side == 'left':
                        x1 = mx0 - lane_gap
                        x0 = x1 - bw
                    else:
                        x0 = mx1 + lane_gap
                        x1 = x0 + bw
                    y0, y1 = cy - bh / 2, cy + bh / 2

                box = (x0, y0, x1, y1)
                if x0 < margin or y0 < margin or x1 > W - margin or y1 > H - margin:
                    continue
                if _box_intersects_molecule(box, mol_bbox, pad=30.0):
                    continue
                if any(_boxes_overlap(box, old, pad=14.0) for old in occupied):
                    continue

                bx, by = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                leader_len = math.hypot(bx - ax, by - ay)
                score = side_rank * 1200.0 + lane * 350.0 + abs(shift) * 1.4 + leader_len
                candidates.append((score, box))

    if candidates:
        return min(candidates, key=lambda z: z[0])[1]

    # Defensive fallback: still keep the label outside the molecule, even if it
    # means using the nearest free canvas edge.
    fallback_sides = ['top', 'bottom', 'left', 'right']
    for side in fallback_sides:
        if side == 'top':
            box = (margin, margin, margin + bw, margin + bh)
        elif side == 'bottom':
            box = (margin, H - margin - bh, margin + bw, H - margin)
        elif side == 'left':
            box = (margin, max(margin, min(H - margin - bh, ay - bh / 2)), margin + bw,
                   max(margin, min(H - margin - bh, ay - bh / 2)) + bh)
        else:
            box = (W - margin - bw, max(margin, min(H - margin - bh, ay - bh / 2)), W - margin,
                   max(margin, min(H - margin - bh, ay - bh / 2)) + bh)
        if not _box_intersects_molecule(box, mol_bbox, pad=20.0) and not any(
            _boxes_overlap(box, old, pad=10.0) for old in occupied
        ):
            return box
    return (margin, margin, margin + bw, margin + bh)


def _draw_external_label_overlay(img: Image.Image, anchors, mol_bbox, canvas_wh):
    """Draw old-paper-style external cartouches with leader lines.

    Cartouches are white with a family/motif-colored border and are guaranteed to
    remain outside the molecular bounding box.  This reproduces the visual logic
    of the reference figure: label outside, leader line inside.
    """
    draw = ImageDraw.Draw(img, 'RGBA')
    font = _pil_font(36, bold=True)
    occupied = []

    # Place motifs nearest to the outer molecular boundary first; this gives the
    # most natural short leader lines and stabilizes the layout.
    mx0, my0, mx1, my1 = mol_bbox
    def edge_distance(item):
        _, (x, y), _ = item
        return min(abs(x - mx0), abs(mx1 - x), abs(y - my0), abs(my1 - y))

    for tag, anchor, color in sorted(anchors, key=edge_distance):
        bbox = draw.textbbox((0, 0), tag, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box = _choose_external_label_box(
            anchor, mol_bbox, (tw + 36, th + 24), canvas_wh, occupied
        )
        occupied.append(box)
        x0, y0, x1, y1 = box
        bx, by = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rgb = _rgb255(color)
        outline = tuple(max(0, c - 45) for c in rgb)
        target = _leader_target_on_box(anchor, box)

        # A thin leader line, like the reference figure; no line crosses the text.
        draw.line([anchor, target], fill=outline + (255,), width=5)
        ax, ay = anchor
        draw.ellipse([ax - 7, ay - 7, ax + 7, ay + 7], fill=outline + (255,),
                     outline=(255, 255, 255, 255), width=2)

        # White cartouche with colored outline, not a solid colored box.  This is
        # much closer to the old manuscript figure and remains readable in print.
        draw.rounded_rectangle(
            box, radius=12, fill=(255, 255, 255, 246), outline=outline + (255,), width=4
        )
        draw.text((bx, by), tag, fill=outline + (255,), font=font, anchor='mm')
    return img

def _draw_group_structure(smiles: str, groups: Sequence[Dict], out_path: Path, mode: str, top_n: int, dpi: int) -> List[Dict]:
    """RDKit structure with highlighted groups and EXTERNAL M/P/F cartouches.

    M/P labels are never drawn inside the molecule.  RDKit is asked to reserve a
    generous peripheral margin, then each cartouche is placed outside the atom
    bounding box with a leader line to the corresponding group centroid.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    ranked = sorted(groups, key=lambda r: -abs(float(r.get('logit_contribution', 0.0))))[:top_n]
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(m)

    highlight_atoms, atom_colors, radii = [], {}, {}
    group_meta = []
    for rank, row in enumerate(ranked, start=1):
        ids = [int(x) for x in str(row.get('atoms', '')).split() if str(x).strip()]
        if not ids:
            continue
        if mode == 'fragment':
            color = FRAGMENT_COLORS[(rank - 1) % len(FRAGMENT_COLORS)]
            tag = f'F{rank}'
        elif mode == 'motif_nodes':
            color = MOTIF_NODE_COLORS[(rank - 1) % len(MOTIF_NODE_COLORS)]
            tag = f'M{rank} · STR'
        else:
            fam = str(row.get('motif_family_short', 'STR'))
            color = FAMILY_COLORS.get(fam, FAMILY_COLORS['STR'])
            pid = str(row.get('pharmacophore_label', f'P{rank}'))
            tag = f'{pid} · {fam}'
        group_meta.append((tag, ids, color))
        for a in ids:
            if a not in atom_colors:
                atom_colors[a] = color
            if a not in highlight_atoms:
                highlight_atoms.append(a)
            radii[a] = 0.34

    # Extra canvas + RDKit padding are intentional.  They create a real external
    # annotation corridor around the structure, instead of forcing labels into it.
    W, H = 2600, 1500
    drawer = rdMolDraw2D.MolDraw2DCairo(W, H)
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.continuousHighlight = True
    opts.highlightBondWidthMultiplier = 7
    opts.bondLineWidth = 2.35
    opts.minFontSize = 20
    opts.maxFontSize = 44
    try:
        opts.padding = 0.18
    except Exception:
        pass
    try:
        opts.atomHighlightsAreCircles = True
    except Exception:
        pass

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, m,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=radii,
    )

    atom_xy = {}
    for i in range(m.GetNumAtoms()):
        try:
            pt = drawer.GetDrawCoords(i)
            atom_xy[i] = (float(pt.x), float(pt.y))
        except Exception:
            pass

    if atom_xy:
        xs = [q[0] for q in atom_xy.values()]
        ys = [q[1] for q in atom_xy.values()]
        mol_bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        mol_bbox = (W * 0.25, H * 0.25, W * 0.75, H * 0.75)

    drawer.FinishDrawing()
    img = Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGBA')

    anchors = []
    for tag, ids, color in group_meta:
        pts = [atom_xy[a] for a in ids if a in atom_xy]
        if pts:
            anchors.append((
                tag,
                (float(np.mean([q[0] for q in pts])), float(np.mean([q[1] for q in pts]))),
                color,
            ))

    img = _draw_external_label_overlay(img, anchors, mol_bbox, (W, H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(out_path, dpi=(dpi, dpi))
    return ranked

def _branch_grid_atoms(entries: Sequence[Dict], out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, e in zip(axes.flat, entries):
        img = _open_trimmed_image(e["image"])
        ax.imshow(img)
        ax.set_title(
            f"{e['case']} | y={e['label']} pred={e['prediction']} | p(blocker)={e['p_blocker']:.3f}\n"
            f"IG completeness error={e['ig_relative_error']:.2%}",
            fontsize=11,
        )
        ax.axis("off")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _branch_grid_group(entries: Sequence[Dict], out_path: Path, dpi: int, kind: str) -> None:
    fig = plt.figure(figsize=(20, 14))
    outer = fig.add_gridspec(2, 2, wspace=0.20, hspace=0.28)
    for idx, e in enumerate(entries):
        r, c = divmod(idx, 2)
        sub = outer[r, c].subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.05)
        ax_m = fig.add_subplot(sub[0, 0])
        ax_b = fig.add_subplot(sub[0, 1])
        ax_m.imshow(_open_trimmed_image(e["image"]))
        ax_m.set_title(f"{e['case']} | p(blocker)={e['p_blocker']:.3f}", fontsize=11)
        ax_m.axis("off")
        ranked = e["ranked"]
        labels, vals = [], []
        for j, row in enumerate(ranked, start=1):
            if kind == "fragment":
                labels.append(f"F{j}: {str(row.get('fragment_smiles',''))[:24]}")
            elif kind == "motif_nodes":
                labels.append(f"M{j}: {row.get('motif_name','')} [{row.get('motif_family_short','')}]" )
            else:
                labels.append(f"P{j}: {row.get('motif_name','')} [{row.get('motif_family_short','')}]" )
            vals.append(float(row.get("logit_contribution", 0.0)))
        y = np.arange(len(labels))
        bar_colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
        ax_b.barh(y, vals, color=bar_colors, alpha=0.82)
        ax_b.set_yticks(y, labels=labels, fontsize=8)
        ax_b.invert_yaxis()
        ax_b.axvline(0.0, color="black", lw=0.8)
        ax_b.set_xlabel("Direct occlusion Δ blocker logit", fontsize=9)
        ax_b.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2))
        ax_b.tick_params(axis="x", labelsize=8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _relation_candidates_by_attention(batch: Dict, item: Dict, full_out: Dict) -> List[Dict]:
    """Return explicit motif-pair metadata ranked by auxiliary attention, without perturbation."""
    n_atoms = int(item["n_atoms"])
    n_motifs = int(item["n_motifs"])
    relation = batch["relation"][0].detach().cpu().numpy()
    pair_mask = batch["pair_mask"][0].detach().cpu().numpy().astype(bool)
    attn = full_out.get("hierarchical_attention")
    if torch.is_tensor(attn) and attn.numel() > 0:
        am = attn[0].mean(dim=0).detach().cpu().numpy()
    else:
        am = None
    rows = []
    for i in range(n_motifs):
        fi = int(item["motif_families"][i])
        if fi not in (1, 2, 3, 4):
            continue
        ni = n_atoms + i
        for j in range(i + 1, n_motifs):
            fj = int(item["motif_families"][j])
            if fj not in (1, 2, 3, 4):
                continue
            nj = n_atoms + j
            if not pair_mask[ni, nj] and not pair_mask[nj, ni]:
                continue
            aux = 0.0 if am is None else 0.5 * (float(am[ni, nj]) + float(am[nj, ni]))
            rows.append({
                "motif_i": i, "motif_j": j, "node_i": ni, "node_j": nj,
                "label_i": str(item["motif_names"][i]), "label_j": str(item["motif_names"][j]),
                "family_i": _family_short(fi), "family_j": _family_short(fj),
                "relation_id": int(relation[ni, nj]),
                "relation_name": REL_NAMES.get(int(relation[ni, nj]), f"rel_{int(relation[ni, nj])}"),
                "motif_topological_distance": int(batch["pair_dist"][0, ni, nj].item()),
                "aux_last_layer_attention": aux,
            })
    rows.sort(key=lambda r: (-float(r["aux_last_layer_attention"]), int(r["motif_i"]), int(r["motif_j"])))
    return rows


def _perturb_relation_candidates(model, batch: Dict, candidates: Sequence[Dict], full_prob: float, full_logit: float) -> List[Dict]:
    rows = []
    for base in candidates:
        r = dict(base)
        pdel, zdel, _ = _prob_and_logit(model, _mask_pair_edge(batch, int(r["node_i"]), int(r["node_j"])))
        _, zneu, _ = _prob_and_logit(model, _neutralize_pair_relation(batch, int(r["node_i"]), int(r["node_j"])))
        r.update({
            "p_blocker_full": full_prob,
            "p_blocker_masked": pdel,
            "probability_contribution": full_prob - pdel,
            "logit_contribution": full_logit - zdel,
            "relation_type_neutralization_logit": full_logit - zneu,
            "direction": "blocker" if (full_logit - zdel) > 0 else "non_blocker" if (full_logit - zdel) < 0 else "neutral",
        })
        rows.append(r)
    rows.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    return rows


def _draw_pig_4(entries: Sequence[Dict], out_path: Path, dpi: int, top_relations: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, e in zip(axes.flat, entries):
        motifs = e["motifs"]
        relations = sorted(e["relations"], key=lambda r: -abs(float(r["logit_contribution"])))[:top_relations]
        # Keep only endpoints of displayed relations. This prevents visually
        # distracting isolated nodes in the publication PIG panel.
        keep = {int(r["motif_i"]) for r in relations} | {int(r["motif_j"]) for r in relations}
        if not keep:
            keep = {int(r["motif_id"]) for r in sorted(motifs, key=lambda r: -abs(float(r["logit_contribution"])))[:4]}
        md = {int(r["motif_id"]): r for r in motifs if int(r["motif_id"]) in keep}
        G = nx.Graph()
        for mid, row in md.items():
            G.add_node(mid, row=row)
        for rr in relations:
            i, j = int(rr["motif_i"]), int(rr["motif_j"])
            if i in G and j in G:
                G.add_edge(i, j, row=rr)
        if len(G) == 0:
            ax.text(0.5, 0.5, "No eligible PIG relation", ha="center", va="center")
            ax.axis("off")
            continue
        pos = nx.spring_layout(G, seed=42, weight=None)
        node_vals = [float(G.nodes[n]["row"]["logit_contribution"]) for n in G.nodes]
        vmax = max(max(abs(v) for v in node_vals), 1e-8)
        node_colors = [FAMILY_COLORS.get(str(G.nodes[n]["row"].get("motif_family_short", "STR")), FAMILY_COLORS["STR"]) for n in G.nodes]
        node_sizes = [1300 + 2200 * min(1.0, abs(float(G.nodes[n]["row"]["logit_contribution"])) / vmax) for n in G.nodes]
        edge_vals = [float(G.edges[x]["row"]["logit_contribution"]) for x in G.edges]
        emax = max(max([abs(v) for v in edge_vals], default=0.0), 1e-8)
        edge_widths = [1.4 + 6.0 * abs(float(G.edges[x]["row"]["logit_contribution"])) / emax for x in G.edges]
        edge_colors = ["tab:red" if float(G.edges[x]["row"]["logit_contribution"]) > 0 else "tab:blue" for x in G.edges]
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors, alpha=0.80)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=node_colors, edgecolors="black", linewidths=1.2)
        display_order = {n: k + 1 for k, n in enumerate(sorted(G.nodes))}
        labels = {n: f"P{display_order[n]}\n{G.nodes[n]['row'].get('motif_family_short','')}\n{str(G.nodes[n]['row'].get('motif_name',''))[:12]}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8)
        edge_labels = {}
        for x in G.edges:
            rr = G.edges[x]["row"]
            edge_labels[x] = f"{rr.get('relation_name','')}\nΔz={float(rr.get('logit_contribution',0)):+.2g}"
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels, font_size=6, rotate=False)
        ax.set_title(f"{e['case']} | p(blocker)={e['p_blocker']:.3f}", fontsize=11)
        ax.axis("off")
    for ax in axes.flat[len(entries):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)



def _draw_fixed_motif_subset_structure(
    smiles: str,
    motif_rows: Sequence[Dict],
    labels_by_motif_id: Dict[int, str],
    out_path: Path,
    dpi: int,
) -> None:
    """Draw exactly the explicit PIG pharmacophores with external P labels."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        raise ValueError(smiles)
    m = Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(m)
    by_id = {int(r["motif_id"]): r for r in motif_rows}
    atoms, atom_colors, radii, group_meta = [], {}, {}, []
    for mid, tag in labels_by_motif_id.items():
        row = by_id.get(int(mid))
        if row is None:
            continue
        fam = str(row.get("motif_family_short", "STR"))
        color = FAMILY_COLORS.get(fam, FAMILY_COLORS["STR"])
        ids = [int(x) for x in str(row.get("atoms", "")).split() if str(x).strip()]
        if not ids:
            continue
        group_meta.append((tag, ids, color))
        for a in ids:
            if a not in atom_colors:
                atom_colors[a]=color
            radii[a]=0.38
            if a not in atoms:
                atoms.append(a)
    W,H=2200,1250
    drawer=rdMolDraw2D.MolDraw2DCairo(W,H)
    opts=drawer.drawOptions()
    opts.fillHighlights=True
    opts.continuousHighlight=True
    opts.highlightBondWidthMultiplier=8
    opts.bondLineWidth=2.3
    opts.minFontSize=20
    opts.maxFontSize=44
    rdMolDraw2D.PrepareAndDrawMolecule(drawer,m,
        highlightAtoms=atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=radii)
    atom_xy={}
    for i in range(m.GetNumAtoms()):
        try:
            pt=drawer.GetDrawCoords(i)
            atom_xy[i]=(float(pt.x),float(pt.y))
        except Exception:
            pass
    # Bounding box of the molecule in RDKit drawing coordinates.
    # The external-label renderer needs this box to keep P-label cartouches
    # outside the molecular structure instead of placing them on top of atoms.
    if atom_xy:
        xs = [q[0] for q in atom_xy.values()]
        ys = [q[1] for q in atom_xy.values()]
        mol_bbox = (min(xs), min(ys), max(xs), max(ys))
    else:
        mol_bbox = (W * 0.25, H * 0.25, W * 0.75, H * 0.75)

    drawer.FinishDrawing()
    img=Image.open(io.BytesIO(drawer.GetDrawingText())).convert('RGBA')
    anchors=[]
    for tag,ids,color in group_meta:
        pts=[atom_xy[a] for a in ids if a in atom_xy]
        if pts:
            anchors.append((tag,(float(np.mean([q[0] for q in pts])),float(np.mean([q[1] for q in pts]))),color))
    img=_draw_external_label_overlay(img, anchors, mol_bbox, (W, H))
    out_path.parent.mkdir(parents=True,exist_ok=True)
    img.convert('RGB').save(out_path,dpi=(dpi,dpi))

def _draw_case_header_axis(ax, entry: Dict, letter: str) -> None:
    """Dedicated header axis: never shares space with the molecule drawing."""
    ax.set_axis_off()
    case = str(entry.get("case", ""))
    color = "#8B1010" if case == "TP" else "#0E3F87" if case == "TN" else "#333333"
    ax.text(0.012, 0.50, letter, ha="left", va="center", fontsize=18, fontweight="bold", color="#111111")
    ax.text(0.50, 0.50, _case_banner_text(case, float(entry.get("p_blocker", 0.0))),
            ha="center", va="center", fontsize=12.8, fontweight="bold", color=color,
            bbox=dict(boxstyle="round,pad=0.34", facecolor=_case_bg(case), edgecolor="none", alpha=0.98))


def _draw_smiles_axis(ax, smiles: str, width: int = 52, fontsize: float = 8.3) -> None:
    """Dedicated SMILES axis placed below the molecule; cannot overlap atoms."""
    ax.set_axis_off()
    txt = _wrap_smiles_text(smiles, width)
    ax.text(0.50, 0.78, "Canonical SMILES", ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#4B5563")
    ax.text(0.50, 0.34, txt, ha="center", va="center", fontsize=fontsize,
            family="monospace", color="#222222", linespacing=1.18, clip_on=False)


def _publication_atom_figure(entries: Sequence[Dict], out_path: Path, dpi: int) -> None:
    """Q1 main figure: four large molecules with signed IG and a dedicated SMILES band."""
    _set_publication_style()
    entries = list(entries)[:4]
    if not entries:
        return
    fig = plt.figure(figsize=(18.8, 13.6))
    outer = fig.add_gridspec(2, 2, left=0.028, right=0.972, top=0.91, bottom=0.17,
                             wspace=0.055, hspace=0.12)
    letters = ["(a)", "(b)", "(c)", "(d)"]
    for k, (e, letter) in enumerate(zip(entries, letters)):
        holder = outer[k // 2, k % 2]
        sub = holder.subgridspec(3, 1, height_ratios=[0.52, 4.55, 0.92], hspace=0.015)
        ax_h = fig.add_subplot(sub[0])
        ax_m = fig.add_subplot(sub[1])
        ax_s = fig.add_subplot(sub[2])

        _draw_case_header_axis(ax_h, e, letter)
        ax_m.imshow(_open_trimmed_image(e["image"], pad=20))
        ax_m.set_xticks([]); ax_m.set_yticks([])
        _panel_border(ax_m, str(e["case"]), lw=1.30)
        _draw_smiles_axis(ax_s, e.get("smiles", ""), width=54, fontsize=8.25)

    cax = fig.add_axes([0.225, 0.078, 0.55, 0.027])
    grad = np.linspace(0, 1, 512)[None, :]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("ig_signed", [POS_COLOR, "#FFFFFF", NEG_COLOR])
    cax.imshow(grad, aspect="auto", cmap=cmap, origin="lower")
    cax.set_xticks([0, 256, 511], labels=["+", "0", "−"], fontsize=13, fontweight="bold")
    cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8); sp.set_edgecolor("#444444")
    fig.text(0.215, 0.091, "Supports blocker\n(positive IG)", ha="right", va="center",
             color=POS_COLOR, fontsize=11.8, fontweight="bold")
    fig.text(0.785, 0.091, "Supports non-blocker\n(negative IG)", ha="left", va="center",
             color=NEG_COLOR, fontsize=11.8, fontweight="bold")
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)


def _wrap_smiles_text(smiles: str, width: int = 34) -> str:
    s = str(smiles or '').strip()
    if not s:
        return ''
    parts = [s[i:i+width] for i in range(0, len(s), width)]
    return "\n".join(parts)


def _draw_smiles_under_axis(ax, smiles: str, fontsize: float = 8.8, y: float = 0.018) -> None:
    txt = _wrap_smiles_text(smiles, 34)
    if not txt:
        return
    ax.text(0.5, y, 'SMILES: ' + txt, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=fontsize, color='#222222', family='monospace',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.90),
            zorder=20)

def _open_trimmed_image(path: str | Path, pad: int = 28) -> Image.Image:
    """Crop the large white RDKit canvas so the molecule fills its panel."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    # Keep any pixel that is not almost-white. This preserves thin black bonds,
    # atom labels, annotations and translucent highlights.
    mask = np.any(arr < 248, axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    x0, x1 = max(0, int(xs.min()) - pad), min(im.width, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(im.height, int(ys.max()) + pad + 1)
    return im.crop((x0, y0, x1, y1))


def _format_delta(v: float) -> str:
    av = abs(float(v))
    if av == 0:
        return "0"
    if av < 1e-3:
        return f"{v:+.2e}"
    return f"{v:+.3f}"


def _decorate_table(table, rows: Sequence[Dict], delta_col: int, family_col: Optional[int] = None) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.65)
        if r == 0:
            cell.set_facecolor("#EEF2F7")
            cell.get_text().set_fontweight("bold")
    for i, row in enumerate(rows, start=1):
        if (i, delta_col) in table.get_celld():
            v = float(row.get("logit_contribution", 0.0))
            table[i, delta_col].get_text().set_color(POS_COLOR if v > 0 else NEG_COLOR if v < 0 else "#444444")
            table[i, delta_col].get_text().set_fontweight("bold")
        if family_col is not None and (i, family_col) in table.get_celld():
            fam = str(row.get("motif_family_short", "STR"))
            rgb = FAMILY_COLORS.get(fam, FAMILY_COLORS["STR"])
            pale = tuple(0.78 + 0.22 * x for x in rgb)
            table[i, family_col].set_facecolor(pale)


def _case_panel_container(fig, spec, case: str):
    """One visual container around header + molecule + SMILES + score band."""
    ax = fig.add_subplot(spec)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_facecolor("#FFFFFF")
    border = "#E88989" if str(case) == "TP" else "#78ACE8" if str(case) == "TN" else "#B0B0B0"
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(border); sp.set_linewidth(1.35)
    ax.set_zorder(0)
    return ax


def _draw_case_header_strip(ax, entry: Dict, letter: str) -> None:
    """Full-width case strip used by Figs. 2 and 3; avoids floating banner placement."""
    case = str(entry.get("case", ""))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(_case_bg(case))
    for sp in ax.spines.values():
        sp.set_visible(False)
    color = "#9C1C1C" if case == "TP" else "#154C93" if case == "TN" else "#333333"
    ax.text(0.018, 0.5, letter, ha="left", va="center", fontsize=17.5, fontweight="bold", color="#111111")
    ax.text(0.52, 0.5, _case_banner_text(case, float(entry.get("p_blocker", 0.0))),
            ha="center", va="center", fontsize=12.4, fontweight="bold", color=color)


def _draw_two_line_summary_axis(ax, lines: Sequence[str], case: str, title: str) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(_case_bg(case))
    for sp in ax.spines.values():
        sp.set_visible(False)
    color = "#9C1C1C" if str(case) == "TP" else "#154C93" if str(case) == "TN" else "#333333"
    ax.text(0.018, 0.76, title, ha="left", va="center", fontsize=9.4, fontweight="bold", color=color)
    if lines:
        ax.text(0.5, 0.47, "   |   ".join(lines[:2]), ha="center", va="center", fontsize=8.9, color="#222222")
    if len(lines) > 2:
        ax.text(0.5, 0.18, "   |   ".join(lines[2:4]), ha="center", va="center", fontsize=8.9, color="#222222")


def _publication_motif_figure(entries: Sequence[Dict], out_path: Path, dpi: int) -> None:
    """Four-panel structural motif view with a cohesive panel and non-gray node colors."""
    _set_publication_style()
    entries = list(entries)[:4]
    if not entries:
        return

    fig = plt.figure(figsize=(18.8, 13.2))
    outer = fig.add_gridspec(2, 2, left=0.025, right=0.975, top=0.895, bottom=0.105,
                             wspace=0.045, hspace=0.085)
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for k, (e, letter) in enumerate(zip(entries, letters)):
        holder = outer[k // 2, k % 2]
        _case_panel_container(fig, holder, str(e["case"]))
        sub = holder.subgridspec(4, 1, height_ratios=[0.56, 4.25, 0.88, 1.08], hspace=0.018)
        ax_h = fig.add_subplot(sub[0]); ax_m = fig.add_subplot(sub[1])
        ax_s = fig.add_subplot(sub[2]); ax_i = fig.add_subplot(sub[3])
        for ax in (ax_h, ax_m, ax_s, ax_i):
            ax.set_zorder(2)
            ax.patch.set_alpha(0.0)

        _draw_case_header_strip(ax_h, e, letter)
        ax_m.imshow(_open_trimmed_image(e["image"], pad=18))
        ax_m.set_axis_off()
        _draw_smiles_axis(ax_s, e.get("smiles", ""), width=58, fontsize=8.1)

        ranked = list(e.get("ranked", []))[:4]
        lines = []
        for j, row in enumerate(ranked, 1):
            desc = _structural_motif_descriptor(str(e.get("smiles", "")), row, 20)
            dz = _format_delta(float(row.get("logit_contribution", 0.0)))
            lines.append(f"M{j}: {desc}  Δlogit={dz}")
        _draw_two_line_summary_axis(ax_i, lines, str(e["case"]), "Top structural motif nodes (direct occlusion Δlogit)")

    handles = [mpatches.Patch(color=MOTIF_NODE_COLORS[i], label=f"M{i+1}: ranked structural node") for i in range(4)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.036), fontsize=9.9)
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)


def _publication_pharmacophore_figure(entries: Sequence[Dict], out_path: Path, dpi: int) -> None:
    """Four-panel explicit pharmacophore view with a cohesive, aligned case panel."""
    _set_publication_style()
    entries = list(entries)[:4]
    if not entries:
        return

    fig = plt.figure(figsize=(18.8, 13.2))
    outer = fig.add_gridspec(2, 2, left=0.025, right=0.975, top=0.895, bottom=0.105,
                             wspace=0.045, hspace=0.085)
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for k, (e, letter) in enumerate(zip(entries, letters)):
        holder = outer[k // 2, k % 2]
        _case_panel_container(fig, holder, str(e["case"]))
        sub = holder.subgridspec(4, 1, height_ratios=[0.56, 4.25, 0.88, 1.08], hspace=0.018)
        ax_h = fig.add_subplot(sub[0]); ax_m = fig.add_subplot(sub[1])
        ax_s = fig.add_subplot(sub[2]); ax_i = fig.add_subplot(sub[3])
        for ax in (ax_h, ax_m, ax_s, ax_i):
            ax.set_zorder(2)
            ax.patch.set_alpha(0.0)

        _draw_case_header_strip(ax_h, e, letter)
        ax_m.imshow(_open_trimmed_image(e["image"], pad=18))
        ax_m.set_axis_off()
        _draw_smiles_axis(ax_s, e.get("smiles", ""), width=58, fontsize=8.1)

        ranked = list(e.get("ranked", []))[:4]
        lines = []
        for j, row in enumerate(ranked, 1):
            fam = str(row.get("motif_family_short", ""))
            name = _short_text(row.get("motif_name", ""), 17)
            pid = str(row.get("pharmacophore_label", f"P{j}"))
            dz = _format_delta(float(row.get("logit_contribution", 0.0)))
            lines.append(f"{pid}: {name} [{fam}]  Δlogit={dz}")
        _draw_two_line_summary_axis(ax_i, lines, str(e["case"]), "Top explicit pharmacophores (direct occlusion Δlogit)")

    handles = [mpatches.Patch(color=FAMILY_COLORS[x], label=f"{x}: {FAMILY_LONG[x]}") for x in ["BN", "AR", "NHC", "PF"]]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.036), fontsize=10.0)
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)


def _pig_case_score(entry: Dict, k: int = 3) -> float:
    vals = sorted([abs(float(r.get("logit_contribution", 0.0))) for r in entry.get("relations", [])], reverse=True)
    return float(sum(vals[:k]))


def _select_two_pig_cases(entries: Sequence[Dict]) -> List[Dict]:
    selected: List[Dict] = []
    for case in ("TP", "TN"):
        g = [e for e in entries if str(e.get("case")) == case and len(e.get("relations", [])) > 0]
        if g:
            selected.append(max(g, key=lambda e: _pig_case_score(e, 3)))
    if len(selected) < 2:
        rest = [e for e in entries if e not in selected and len(e.get("relations", [])) > 0]
        rest.sort(key=lambda e: _pig_case_score(e, 3), reverse=True)
        selected.extend(rest[: 2 - len(selected)])
    return selected[:2]


def _pig_fixed_positions(nodes: Sequence[int]) -> Dict[int, Tuple[float, float]]:
    nodes=list(nodes)
    presets={
        1:[(0.0,0.0)],
        2:[(-1.0,0.0),(1.0,0.0)],
        3:[(-1.0,-0.40),(1.0,-0.40),(0.0,0.95)],
        4:[(-1.05,0.72),(1.05,0.72),(-1.05,-0.72),(1.05,-0.72)],
    }
    pts=presets.get(min(len(nodes),4),presets[4])
    return {n:pts[i] for i,n in enumerate(nodes[:4])}

def _publication_pig_figure(entries: Sequence[Dict], out_path: Path, dpi: int, top_relations: int) -> None:
    """One readable PIG case per figure: full-width molecule on top, graph + table below."""
    _set_publication_style()
    entries = list(entries)
    if not entries:
        return
    same_case = {str(e.get("case")) for e in entries}
    valid = [e for e in entries if e.get("relations")]
    if not valid:
        return
    chosen = [max(valid, key=lambda e: _pig_case_score(e, 3))] if len(same_case) == 1 else _select_two_pig_cases(entries)
    e = chosen[0]

    rels = sorted(e["relations"], key=lambda r: -abs(float(r.get("logit_contribution", 0.0))))[:max(1, top_relations)]
    endpoint_ids: List[int] = []
    for rr in rels:
        for mid in (int(rr["motif_i"]), int(rr["motif_j"])):
            if mid not in endpoint_ids:
                endpoint_ids.append(mid)
    endpoint_ids = endpoint_ids[:4]
    rels = [r for r in rels if int(r["motif_i"]) in endpoint_ids and int(r["motif_j"]) in endpoint_ids]
    all_explicit = _assign_persistent_pharmacophore_ids(e["motifs"])
    motif_by_id_all = {int(r["motif_id"]): r for r in all_explicit}
    labels = {mid: str(motif_by_id_all[mid].get("pharmacophore_label", f"P{mid+1}"))
              for mid in endpoint_ids if mid in motif_by_id_all}
    motif_rows = [motif_by_id_all[mid] for mid in endpoint_ids if mid in motif_by_id_all]

    tmp = out_path.parent / "_pig_case_structures" / f"row_{e['row_id']}_pig_structure.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    _draw_fixed_motif_subset_structure(e["smiles"], motif_rows, labels, tmp, dpi)

    fig = plt.figure(figsize=(16.6, 10.8))
    gs = fig.add_gridspec(5, 1, left=0.04, right=0.96, top=0.91, bottom=0.105, hspace=0.11,
                          height_ratios=[0.55, 3.35, 0.72, 0.16, 3.65])
    ax_h = fig.add_subplot(gs[0])
    ax_m = fig.add_subplot(gs[1])
    ax_sm = fig.add_subplot(gs[2])
    ax_gap = fig.add_subplot(gs[3])
    bottom = gs[4].subgridspec(1, 2, width_ratios=[1.38, 1.0], wspace=0.055)
    ax_g = fig.add_subplot(bottom[0])
    ax_t = fig.add_subplot(bottom[1])

    panel_letter = "(a)" if str(e.get("case")) == "TP" else "(b)" if str(e.get("case")) == "TN" else ""
    _draw_case_header_strip(ax_h, e, panel_letter)
    ax_m.imshow(_open_trimmed_image(tmp, pad=20))
    ax_m.set_xticks([]); ax_m.set_yticks([])
    _panel_border(ax_m, str(e["case"]), lw=1.25)
    ax_m.set_title("(i)", fontsize=13.5, fontweight="bold", pad=5, loc="left")
    _draw_smiles_axis(ax_sm, e.get("smiles", ""), width=72, fontsize=8.35)
    ax_gap.set_axis_off()

    # PIG graph: only node IDs/families and short E1/E2/E3 labels.
    ax_g.set_facecolor("#FFFFFF")
    _panel_border(ax_g, str(e["case"]), lw=1.2)
    G = nx.Graph()
    bymid = {int(r["motif_id"]): r for r in motif_rows}
    for mid in endpoint_ids:
        if mid in bymid:
            G.add_node(mid, row=bymid[mid])
    for ei, rr in enumerate(rels, 1):
        i, j = int(rr["motif_i"]), int(rr["motif_j"])
        if i in G and j in G:
            G.add_edge(i, j, row=rr, edge_label=f"E{ei}")
    pos = _pig_fixed_positions(list(G.nodes))
    if len(G):
        node_colors = [FAMILY_COLORS.get(str(G.nodes[n]["row"].get("motif_family_short", "STR")), FAMILY_COLORS["STR"]) for n in G.nodes]
        nx.draw_networkx_nodes(G, pos, ax=ax_g, node_color=node_colors, node_size=3000, edgecolors="#1F2937", linewidths=1.7)
        node_labels = {n: f"{labels[n]}\n{G.nodes[n]['row'].get('motif_family_short','')}" for n in G.nodes}
        nx.draw_networkx_labels(G, pos, ax=ax_g, labels=node_labels, font_size=10.2, font_weight="bold")
        max_edge = max([abs(float(G.edges[x]["row"].get("logit_contribution", 0.0))) for x in G.edges], default=1.0)
        max_edge = max(max_edge, 1e-12)
        edge_lbl = {}
        for edge in G.edges:
            rr = G.edges[edge]["row"]
            v = float(rr.get("logit_contribution", 0.0))
            width = 2.8 + 6.8 * abs(v) / max_edge
            nx.draw_networkx_edges(G, pos, ax=ax_g, edgelist=[edge], width=width,
                                   edge_color=POS_COLOR if v > 0 else NEG_COLOR, alpha=0.94)
            edge_lbl[edge] = G.edges[edge]["edge_label"]
        nx.draw_networkx_edge_labels(G, pos, ax=ax_g, edge_labels=edge_lbl, font_size=9.4, rotate=False,
                                     bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#AAB2BF", alpha=0.96))
    ax_g.set_xlim(-1.38, 1.38); ax_g.set_ylim(-1.08, 1.08)
    ax_g.set_xticks([]); ax_g.set_yticks([])
    ax_g.set_title("(ii)", fontsize=13.5, fontweight="bold", pad=5, loc="left")

    # Large relation table on the right; all dense quantitative text lives here.
    ax_t.set_facecolor("#FFFFFF")
    _panel_border(ax_t, str(e["case"]), lw=1.2)
    ax_t.set_xticks([]); ax_t.set_yticks([])
    rows = []
    for ei, rr in enumerate(rels, 1):
        pi, pj = labels[int(rr["motif_i"])], labels[int(rr["motif_j"])]
        rows.append([
            f"E{ei}", f"{pi}–{pj}", _clean_relation_name(rr.get("relation_name", "")),
            str(rr.get("motif_topological_distance", "—")),
            _format_delta(float(rr.get("logit_contribution", 0.0))),
        ])
    table = ax_t.table(cellText=rows, colLabels=["Edge", "Pair", "Relation", "d_topo", "Δlogit"],
                       cellLoc="center", colLoc="center", loc="center",
                       colWidths=[0.12, 0.17, 0.33, 0.15, 0.20])
    table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1.02, 1.62)
    for (ri, cc), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR); cell.set_linewidth(0.75)
        if ri == 0:
            cell.set_facecolor("#EAF0F7"); cell.get_text().set_fontweight("bold")
    for i, rr in enumerate(rels, start=1):
        v = float(rr.get("logit_contribution", 0.0))
        table[i, 4].get_text().set_color(POS_COLOR if v > 0 else NEG_COLOR)
        table[i, 4].get_text().set_fontweight("bold")
    ax_t.set_title("(iii)", fontsize=13.5, fontweight="bold", pad=5, loc="left")

    fam_handles = [mpatches.Patch(color=FAMILY_COLORS[x], label=f"{x}: {FAMILY_LONG[x]}") for x in ["BN", "AR", "NHC", "PF"]]
    edge_handles = [mpatches.Patch(color=POS_COLOR, label="positive Δlogit: supports blocker"),
                    mpatches.Patch(color=NEG_COLOR, label="negative Δlogit: supports non-blocker")]
    fig.legend(handles=fam_handles + edge_handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.038), fontsize=9.25)
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)


def _final_atom_entry(model, feat_cfg, scaler, device, record: Record, threshold: float, steps: int, out_dir: Path, dpi: int) -> Tuple[Dict, List[Dict]]:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    ig = integrated_gradients_raw_atoms(model, batch, item, steps)
    p = float(ig["normal_prob"])
    pred = int(p >= threshold)
    case = _case_name(record.label, pred)
    scores = np.asarray(ig["atom_scores"], dtype=np.float32)
    img = out_dir / "branch_images" / f"atom_row_{record.row_id}.png"
    _draw_signed_atom_percentile(can, scores, img, "", dpi)
    rows = _atom_rows(record.row_id, can, record.label, pred, case, p, scores)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        raw_dir / f"row_{record.row_id}_raw_input_ig.npz",
        feature_attributions=np.asarray(ig["feature_attributions"], dtype=np.float32),
        atom_scores=scores,
        smiles=np.asarray(can),
        row_id=np.asarray(record.row_id),
    )
    return {
        "row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case,
        "p_blocker": p, "blocker_logit": float(ig["normal_logit"]), "ig_relative_error": float(ig["relative_error"]),
        "ig_completeness_delta": float(ig["completeness_delta"]), "image": str(img),
    }, rows


def _final_fragment_entry(model, feat_cfg, scaler, device, record: Record, threshold: float, out_dir: Path, dpi: int, top_n: int) -> Tuple[Dict, List[Dict]]:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    p, z, _ = _prob_and_logit(model, batch)
    pred, case = int(p >= threshold), _case_name(record.label, int(p >= threshold))
    with torch.no_grad():
        atom_proj = model.graph_encoder.node_proj(batch["node_x"][:, : int(item["n_atoms"])]).squeeze(0)
    rows = _fragment_occlusion(model, batch, item, p, z, atom_proj)
    rows.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    _annotate_rows(rows, record.row_id, can, record.label, pred, case)
    img = out_dir / "branch_images" / f"fragment_row_{record.row_id}.png"
    ranked = _draw_group_structure(can, rows, img, "fragment", top_n, dpi)
    return {"row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case, "p_blocker": p, "image": str(img), "ranked": ranked}, rows


def _final_motif_node_entry(model, feat_cfg, scaler, device, record: Record, threshold: float, out_dir: Path, dpi: int, top_n: int) -> Tuple[Dict, List[Dict]]:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    p, z, full_out = _prob_and_logit(model, batch)
    pred, case = int(p >= threshold), _case_name(record.label, int(p >= threshold))
    rows, _ = _motif_and_family_occlusion(model, batch, item, p, z)
    _add_motif_attention(rows, full_out, item)
    all_rows = [dict(r) for r in rows if str(r.get("motif_name", "")) != "whole_molecule"]
    # Fig. 2 is deliberately the STRUCTURAL hierarchical view.  Explicit BN/AR/NHC/PF
    # nodes are reserved for Fig. 3, otherwise both figures collapse to the same regions.
    structural_rows = [r for r in all_rows if str(r.get("motif_family_short", "STR")) == "STR"]
    rows = structural_rows if structural_rows else all_rows
    rows.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    _annotate_rows(rows, record.row_id, can, record.label, pred, case)
    img = out_dir / "branch_images" / f"motif_node_row_{record.row_id}.png"
    ranked = _draw_group_structure(can, rows, img, "motif_nodes", top_n, dpi)
    return {"row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case, "p_blocker": p, "image": str(img), "ranked": ranked}, rows


def _final_motif_entry(model, feat_cfg, scaler, device, record: Record, threshold: float, out_dir: Path, dpi: int, top_n: int) -> Tuple[Dict, List[Dict], List[Dict]]:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    p, z, full_out = _prob_and_logit(model, batch)
    pred, case = int(p >= threshold), _case_name(record.label, int(p >= threshold))
    rows, fam_rows = _motif_and_family_occlusion(model, batch, item, p, z)
    _add_motif_attention(rows, full_out, item)
    rows = _explicit_motif_rows(rows)
    rows.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    _annotate_rows(rows, record.row_id, can, record.label, pred, case)
    _annotate_rows(fam_rows, record.row_id, can, record.label, pred, case)
    img = out_dir / "branch_images" / f"motif_row_{record.row_id}.png"
    ranked = _draw_group_structure(can, rows, img, "motif", top_n, dpi)
    return {"row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case, "p_blocker": p, "image": str(img), "ranked": ranked}, rows, fam_rows


def _final_relation_entry(model, feat_cfg, scaler, device, record: Record, threshold: float) -> Tuple[Dict, List[Dict], List[Dict]]:
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    p, z, full_out = _prob_and_logit(model, batch)
    pred, case = int(p >= threshold), _case_name(record.label, int(p >= threshold))
    motifs, _ = _motif_and_family_occlusion(model, batch, item, p, z)
    _add_motif_attention(motifs, full_out, item)
    motifs = _explicit_motif_rows(motifs)
    rel = _explicit_relation_rows(_relation_occlusion(model, batch, item, p, z, full_out))
    _add_relation_neutralization(model, batch, rel, z)
    label_by_mid = {int(r["motif_id"]): str(r.get("pharmacophore_label", f"P{int(r['motif_id'])+1}")) for r in motifs}
    for rr in rel:
        rr["pharmacophore_i"] = label_by_mid.get(int(rr["motif_i"]), f"P{int(rr['motif_i'])+1}")
        rr["pharmacophore_j"] = label_by_mid.get(int(rr["motif_j"]), f"P{int(rr['motif_j'])+1}")
    motifs.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    rel.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    _annotate_rows(motifs, record.row_id, can, record.label, pred, case)
    _annotate_rows(rel, record.row_id, can, record.label, pred, case)
    return {"row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case, "p_blocker": p, "motifs": motifs, "relations": rel}, motifs, rel


def _final_relation_entry_hierarchical(model, feat_cfg, scaler, device, record: Record, threshold: float) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Full hierarchical PIG explanation including structural and explicit motifs."""
    item, can = _build_single_item(record.smiles, record.label, record.row_id, feat_cfg, scaler)
    batch = move_batch(collate_hera_hgt([item]), device)
    p, z, full_out = _prob_and_logit(model, batch)
    pred, case = int(p >= threshold), _case_name(record.label, int(p >= threshold))
    motifs, _ = _motif_and_family_occlusion(model, batch, item, p, z)
    _add_motif_attention(motifs, full_out, item)
    motifs = [dict(r) for r in motifs if str(r.get("motif_name", "")) != "whole_molecule"]
    rel = _relation_occlusion(model, batch, item, p, z, full_out)
    _add_relation_neutralization(model, batch, rel, z)
    motifs.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    rel.sort(key=lambda r: -abs(float(r["logit_contribution"])))
    _annotate_rows(motifs, record.row_id, can, record.label, pred, case)
    _annotate_rows(rel, record.row_id, can, record.label, pred, case)
    return {"row_id": record.row_id, "smiles": can, "label": record.label, "prediction": pred, "case": case, "p_blocker": p, "motifs": motifs, "relations": rel}, motifs, rel


def _records_lookup(records: Sequence[Record]) -> Dict[int, Record]:
    return {int(r.row_id): r for r in records}


def _prescreen(pred_df: pd.DataFrame, n_per_class: int) -> pd.DataFrame:
    pieces = []
    for case in ("TP", "TN"):
        g = pred_df[(pred_df["case"] == case) & (pred_df["eligible_base"])].copy()
        if len(g) <= n_per_class:
            pieces.append(g)
        else:
            # Quantile spread across confidence, deterministic and independent of
            # visual appearance or attribution results.
            pieces.append(_interior_quantile_pick(g, n_per_class))
    return pd.concat(pieces, ignore_index=True) if pieces else pred_df.iloc[0:0].copy()


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def _bootstrap_mean_ci(values: Sequence[float], reps: int, seed: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=float)
    for i in range(reps):
        means[i] = float(rng.choice(arr, size=len(arr), replace=True).mean())
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mask_pair_edges(batch: Dict, pairs: Sequence[Tuple[int, int]]) -> Dict:
    out = _clone_batch(batch)
    max_pd = int(out["pair_dist"].max().item()) if out["pair_dist"].numel() else 0
    for i, j in pairs:
        for a, b in ((int(i), int(j)), (int(j), int(i))):
            out["pair_mask"][0, a, b] = False
            out["relation"][0, a, b] = 0
            out["pair_bond10"][0, a, b, :] = 0.0
            out["pair_dist"][0, a, b] = max_pd
    return out


def _faithfulness_analysis(model, feat_cfg, scaler, device, records: Sequence[Record], threshold: float, n_molecules: int, controls: int, bootstrap_reps: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    # deterministic balanced subset
    recs = list(records)
    by_label = {0: [r for r in recs if r.label == 0], 1: [r for r in recs if r.label == 1]}
    chosen: List[Record] = []
    each = max(1, n_molecules // 2)
    for label in (1, 0):
        pool = by_label[label]
        if len(pool) > each:
            idx = np.linspace(0, len(pool) - 1, each).round().astype(int)
            chosen.extend([pool[i] for i in idx])
        else:
            chosen.extend(pool)

    for rec in chosen:
        item, can = _build_single_item(rec.smiles, rec.label, rec.row_id, feat_cfg, scaler)
        batch = move_batch(collate_hera_hgt([item]), device)
        p, z, out = _prob_and_logit(model, batch)
        pred = int(p >= threshold)
        if pred != rec.label:
            continue
        motifs, _ = _motif_and_family_occlusion(model, batch, item, p, z)
        _add_motif_attention(motifs, out, item)
        explicit = _explicit_motif_rows(motifs)
        if len(explicit) >= 3:
            # Independent ranking signal for the faithfulness test: global->motif
            # attention. The outcome metric is direct node-occlusion logit change.
            ranked = sorted(explicit, key=lambda r: -float(r.get("aux_global_to_motif_attention", 0.0)))
            all_by_id = {int(r["motif_id"]): r for r in explicit}
            for k in (1, 2, 3):
                if len(ranked) < k:
                    continue
                top = ranked[:k]
                top_nodes = [int(r["motif_node_id"]) for r in top]
                _, ztop, _ = _prob_and_logit(model, _mask_motif_nodes(batch, top_nodes))
                etop = abs(z - ztop)
                top_ids = {int(r["motif_id"]) for r in top}
                pool = [r for r in explicit if int(r["motif_id"]) not in top_ids]
                rand_eff = []
                for _ in range(controls):
                    selected = []
                    used = set()
                    for tr in top:
                        fam = int(tr["motif_family_id"])
                        candidates = [r for r in pool if int(r["motif_family_id"]) == fam and int(r["motif_id"]) not in used]
                        if not candidates:
                            candidates = [r for r in pool if int(r["motif_id"]) not in used]
                        if not candidates:
                            break
                        rr = candidates[int(rng.integers(0, len(candidates)))]
                        used.add(int(rr["motif_id"]))
                        selected.append(rr)
                    if len(selected) == k:
                        nodes = [int(r["motif_node_id"]) for r in selected]
                        _, zr, _ = _prob_and_logit(model, _mask_motif_nodes(batch, nodes))
                        rand_eff.append(abs(z - zr))
                if rand_eff:
                    rows.append({"level": "motif_attention_to_node_occlusion", "row_id": rec.row_id, "k": k, "top_effect": etop, "random_effect": float(np.mean(rand_eff)), "gap": etop - float(np.mean(rand_eff))})

        rel_all = _relation_candidates_by_attention(batch, item, out)
        if len(rel_all) >= 3:
            ranked_r = rel_all
            for k in (1, 2, 3):
                if len(ranked_r) < k:
                    continue
                top = ranked_r[:k]
                top_pairs = [(int(r["node_i"]), int(r["node_j"])) for r in top]
                _, ztop, _ = _prob_and_logit(model, _mask_pair_edges(batch, top_pairs))
                etop = abs(z - ztop)
                top_keys = {(int(r["motif_i"]), int(r["motif_j"])) for r in top}
                pool = [r for r in rel_all if (int(r["motif_i"]), int(r["motif_j"])) not in top_keys]
                rand_eff = []
                for _ in range(controls):
                    selected = []
                    used = set()
                    for tr in top:
                        relname = str(tr.get("relation_name", ""))
                        candidates = [r for r in pool if str(r.get("relation_name", "")) == relname and (int(r["motif_i"]), int(r["motif_j"])) not in used]
                        if not candidates:
                            candidates = [r for r in pool if (int(r["motif_i"]), int(r["motif_j"])) not in used]
                        if not candidates:
                            break
                        rr = candidates[int(rng.integers(0, len(candidates)))]
                        key = (int(rr["motif_i"]), int(rr["motif_j"]))
                        used.add(key)
                        selected.append(rr)
                    if len(selected) == k:
                        pairs = [(int(r["node_i"]), int(r["node_j"])) for r in selected]
                        _, zr, _ = _prob_and_logit(model, _mask_pair_edges(batch, pairs))
                        rand_eff.append(abs(z - zr))
                if rand_eff:
                    rows.append({"level": "relation_attention_to_edge_deletion", "row_id": rec.row_id, "k": k, "top_effect": etop, "random_effect": float(np.mean(rand_eff)), "gap": etop - float(np.mean(rand_eff))})

    detail = pd.DataFrame(rows)
    summary_rows: List[Dict] = []
    if not detail.empty:
        pvals = []
        pending = []
        for (level, k), g in detail.groupby(["level", "k"]):
            gaps = g["gap"].to_numpy(float)
            mean_gap, lo, hi = _bootstrap_mean_ci(gaps, bootstrap_reps, seed + int(k))
            try:
                stat = wilcoxon(g["top_effect"], g["random_effect"], alternative="greater", zero_method="wilcox")
                pval = float(stat.pvalue)
            except Exception:
                pval = 1.0
            row = {
                "level": level, "k": int(k), "n_molecules": int(len(g)),
                "mean_top_effect": float(g["top_effect"].mean()),
                "mean_random_effect": float(g["random_effect"].mean()),
                "mean_gap": mean_gap, "gap_ci_low": lo, "gap_ci_high": hi,
                "wilcoxon_p": pval,
            }
            pending.append(row)
            pvals.append(pval)
        qvals = _bh_adjust(pvals)
        for row, q in zip(pending, qvals):
            row["bh_q"] = float(q)
            summary_rows.append(row)
    return detail, pd.DataFrame(summary_rows)


def _draw_faithfulness(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    levels = ["motif_attention_to_node_occlusion", "relation_attention_to_edge_deletion"]
    titles = ["Motif ranking faithfulness", "PIG relation ranking faithfulness"]
    for ax, level, title in zip(axes, levels, titles):
        g = summary[summary["level"] == level].sort_values("k") if not summary.empty else pd.DataFrame()
        if g.empty:
            ax.text(0.5, 0.5, "Insufficient eligible molecules", ha="center", va="center")
            ax.axis("off")
            continue
        x = np.arange(len(g))
        w = 0.35
        ax.bar(x - w/2, g["mean_top_effect"], width=w, label="top-ranked")
        ax.bar(x + w/2, g["mean_random_effect"], width=w, label="matched random")
        ax.set_xticks(x, [f"k={int(v)}" for v in g["k"]])
        ax.set_ylabel("|Δ blocker logit|")
        ax.set_title(title)
        ax.legend(fontsize=8)
        for xi, (_, row) in zip(x, g.iterrows()):
            ax.text(xi, max(row["mean_top_effect"], row["mean_random_effect"]) * 1.04 + 1e-12, f"q={row['bh_q']:.3g}", ha="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Primary explanation faithfulness and adaptive-fusion diagnostics
# =============================================================================

def _balanced_correct_records(
    records: Sequence[Record],
    pred_df: pd.DataFrame,
    n_molecules: int,
) -> List[Record]:
    """Deterministically select a balanced TP/TN set from eligible molecules."""
    if n_molecules < 2:
        raise ValueError("n_molecules must be >= 2")
    by_id = {int(r.row_id): r for r in records}
    chosen: List[Record] = []
    each = max(1, n_molecules // 2)
    for case in ("TP", "TN"):
        g = pred_df[(pred_df["eligible_base"]) & (pred_df["case"] == case)].copy()
        if g.empty:
            continue
        sel = _interior_quantile_pick(g, min(each, len(g)))
        for rid in sel["row_id"].astype(int).tolist():
            if rid in by_id:
                chosen.append(by_id[rid])
    return chosen


def _prediction_support_sign(prediction: int) -> float:
    """+1 for blocker predictions and -1 for non-blocker predictions."""
    return 1.0 if int(prediction) == 1 else -1.0


def _raw_atom_deletion_logit(model, batch: Dict, item: Dict, atom_ids: Sequence[int]) -> float:
    """Zero selected raw atom features and rebuild motif raw chemistry exactly."""
    n_atoms = int(item["n_atoms"])
    raw = batch["node_x"][0, :n_atoms].detach().clone()
    ids = sorted({int(i) for i in atom_ids if 0 <= int(i) < n_atoms})
    if ids:
        raw[torch.as_tensor(ids, dtype=torch.long, device=raw.device)] = 0.0
    with torch.no_grad():
        out = _raw_atom_forward(model, batch, raw, item["motif_atoms"])
    return float(out["logits"][0].item())


def _matched_random_atom_set(
    mol: Chem.Mol,
    top_ids: Sequence[int],
    excluded: Sequence[int],
    rng: np.random.Generator,
) -> List[int]:
    """Match random atoms by element/aromaticity when possible."""
    excluded_set = {int(x) for x in excluded}
    selected: List[int] = []
    used = set(excluded_set)
    all_ids = list(range(mol.GetNumAtoms()))
    for tid in top_ids:
        atom = mol.GetAtomWithIdx(int(tid))
        symbol = atom.GetSymbol()
        aromatic = bool(atom.GetIsAromatic())
        degree = int(atom.GetDegree())
        pools = [
            [i for i in all_ids if i not in used and mol.GetAtomWithIdx(i).GetSymbol() == symbol
             and bool(mol.GetAtomWithIdx(i).GetIsAromatic()) == aromatic
             and int(mol.GetAtomWithIdx(i).GetDegree()) == degree],
            [i for i in all_ids if i not in used and mol.GetAtomWithIdx(i).GetSymbol() == symbol
             and bool(mol.GetAtomWithIdx(i).GetIsAromatic()) == aromatic],
            [i for i in all_ids if i not in used and mol.GetAtomWithIdx(i).GetSymbol() == symbol],
            [i for i in all_ids if i not in used],
        ]
        candidates = next((p for p in pools if p), [])
        if not candidates:
            return []
        pick = int(candidates[int(rng.integers(0, len(candidates)))])
        selected.append(pick)
        used.add(pick)
    return selected


def _matched_random_motif_set(
    top_rows: Sequence[Dict],
    pool_rows: Sequence[Dict],
    rng: np.random.Generator,
) -> List[Dict]:
    """Match random motif controls by motif family when possible."""
    selected: List[Dict] = []
    used = set()
    for tr in top_rows:
        fam = int(tr.get("motif_family_id", -1))
        candidates = [
            r for r in pool_rows
            if int(r.get("motif_family_id", -2)) == fam and int(r["motif_id"]) not in used
        ]
        if not candidates:
            candidates = [r for r in pool_rows if int(r["motif_id"]) not in used]
        if not candidates:
            return []
        rr = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(rr)
        used.add(int(rr["motif_id"]))
    return selected


def _matched_random_relation_set(
    top_rows: Sequence[Dict],
    pool_rows: Sequence[Dict],
    rng: np.random.Generator,
) -> List[Dict]:
    """Match random PIG relation controls by relation type when possible."""
    selected: List[Dict] = []
    used = set()
    for tr in top_rows:
        relname = str(tr.get("relation_name", ""))
        candidates = [
            r for r in pool_rows
            if str(r.get("relation_name", "")) == relname
            and (int(r["motif_i"]), int(r["motif_j"])) not in used
        ]
        if not candidates:
            candidates = [
                r for r in pool_rows
                if (int(r["motif_i"]), int(r["motif_j"])) not in used
            ]
        if not candidates:
            return []
        rr = candidates[int(rng.integers(0, len(candidates)))]
        selected.append(rr)
        used.add((int(rr["motif_i"]), int(rr["motif_j"])))
    return selected


def _summarize_primary_faithfulness(
    detail: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Paired top-vs-random tests with bootstrap CIs and BH correction."""
    from scipy.stats import wilcoxon

    if detail.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    pvals: List[float] = []
    pending: List[Dict] = []
    level_offsets = {"atom_ig": 1000, "motif_occlusion": 2000, "pig_edge_deletion": 3000}
    for (level, k), g in detail.groupby(["level", "k"], sort=True):
        top = g["top_effect"].to_numpy(float)
        rnd = g["random_effect"].to_numpy(float)
        gap = top - rnd
        offset = level_offsets.get(str(level), 9000) + int(k) * 101
        mean_gap, gap_lo, gap_hi = _bootstrap_mean_ci(gap, bootstrap_reps, seed + offset)
        mean_top, top_lo, top_hi = _bootstrap_mean_ci(top, bootstrap_reps, seed + offset + 17)
        mean_rnd, rnd_lo, rnd_hi = _bootstrap_mean_ci(rnd, bootstrap_reps, seed + offset + 31)
        try:
            stat = wilcoxon(top, rnd, alternative="greater", zero_method="wilcox")
            pval = float(stat.pvalue)
        except Exception:
            pval = 1.0
        row = {
            "level": str(level),
            "k": int(k),
            "n_molecules": int(len(g)),
            "mean_top_effect": mean_top,
            "top_ci_low": top_lo,
            "top_ci_high": top_hi,
            "mean_random_effect": mean_rnd,
            "random_ci_low": rnd_lo,
            "random_ci_high": rnd_hi,
            "mean_gap": mean_gap,
            "gap_ci_low": gap_lo,
            "gap_ci_high": gap_hi,
            "wilcoxon_p": pval,
        }
        pending.append(row)
        pvals.append(pval)
    qvals = _bh_adjust(pvals)
    for row, q in zip(pending, qvals):
        row["bh_q"] = float(q)
        rows.append(row)
    return pd.DataFrame(rows)


def _primary_explanation_faithfulness(
    model,
    feat_cfg,
    scaler,
    device,
    records: Sequence[Record],
    pred_df: pd.DataFrame,
    threshold: float,
    n_molecules: int,
    controls: int,
    ig_steps: int,
    bootstrap_reps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate primary RHMGT explanations by cumulative top-k deletion.

    IMPORTANT PUBLICATION RULE
    --------------------------
    For each interpretation level, the same eligible molecule subset is used for
    k=1,2,3. A molecule contributes to a level only if it supports all three
    top-k perturbations and their matched-random controls. This prevents the
    sample size from changing across k inside a figure panel.

    Atom ranking: signed Integrated Gradients for the predicted class.
    Motif ranking: direct single-node occlusion contribution for the predicted class.
    Relation ranking: direct single-edge deletion contribution for the predicted class.

    For each level, deleting top-k prediction-supporting elements is compared with
    matched random controls. The measured quantity is the signed decrease in
    predicted-class logit support, so larger positive values indicate a stronger
    faithful effect in the predicted direction.
    """
    if controls < 1:
        raise ValueError("primary-faithfulness-controls must be >= 1")

    rng = np.random.default_rng(seed)
    chosen = _balanced_correct_records(records, pred_df, n_molecules)
    rows: List[Dict] = []
    ks = (1, 2, 3)

    for idx, rec in enumerate(chosen, start=1):
        item, can = _build_single_item(rec.smiles, rec.label, rec.row_id, feat_cfg, scaler)
        batch = move_batch(collate_hera_hgt([item]), device)
        p, z, full_out = _prob_and_logit(model, batch)
        pred = int(p >= threshold)
        if pred != int(rec.label):
            continue

        sign = _prediction_support_sign(pred)
        case = _case_name(rec.label, pred)
        mol = mol_from_smiles(can)
        if mol is None:
            continue

        # -------------------------------------------------------------
        # A) Atom IG ranking -> independent raw-feature deletion test.
        # Keep molecule only if k=1,2,3 can all be evaluated.
        # -------------------------------------------------------------
        ig = integrated_gradients_raw_atoms(model, batch, item, ig_steps)
        atom_scores = np.asarray(ig["atom_scores"], dtype=float)
        atom_support = sign * atom_scores
        ranked_atoms = [int(i) for i in np.argsort(-atom_support) if atom_support[int(i)] > 0]
        atom_rows_for_molecule: List[Dict] = []
        atom_eligible = len(ranked_atoms) >= 3 and mol.GetNumAtoms() >= 6
        if atom_eligible:
            for k in ks:
                top_ids = ranked_atoms[:k]
                ztop = _raw_atom_deletion_logit(model, batch, item, top_ids)
                top_effect = float(sign * (z - ztop))
                random_effects: List[float] = []
                for _ in range(controls):
                    rand_ids = _matched_random_atom_set(mol, top_ids, top_ids, rng)
                    if len(rand_ids) != k:
                        random_effects = []
                        break
                    zr = _raw_atom_deletion_logit(model, batch, item, rand_ids)
                    random_effects.append(float(sign * (z - zr)))
                if len(random_effects) != controls:
                    atom_eligible = False
                    break
                rnd = float(np.mean(random_effects))
                atom_rows_for_molecule.append({
                    "level": "atom_ig", "row_id": int(rec.row_id), "case": case,
                    "label": int(rec.label), "prediction": pred, "k": k,
                    "top_effect": top_effect, "random_effect": rnd,
                    "gap": top_effect - rnd,
                })
        if atom_eligible and len(atom_rows_for_molecule) == len(ks):
            rows.extend(atom_rows_for_molecule)

        # -------------------------------------------------------------
        # B) Hierarchical motif ranking -> cumulative node deletion.
        # Keep molecule only if all k values have enough ranked and
        # matched-random candidate motifs.
        # -------------------------------------------------------------
        motifs, _ = _motif_and_family_occlusion(model, batch, item, p, z)
        motif_candidates = [
            dict(r) for r in motifs
            if str(r.get("motif_name", "")) != "whole_molecule"
        ]
        for r in motif_candidates:
            r["prediction_support"] = float(sign * float(r.get("logit_contribution", 0.0)))
        ranked_motifs = sorted(
            [r for r in motif_candidates if float(r["prediction_support"]) > 0],
            key=lambda r: -float(r["prediction_support"]),
        )
        motif_rows_for_molecule: List[Dict] = []
        motif_eligible = len(ranked_motifs) >= 3 and len(motif_candidates) >= 6
        if motif_eligible:
            for k in ks:
                top = ranked_motifs[:k]
                top_nodes = [int(r["motif_node_id"]) for r in top]
                _, ztop, _ = _prob_and_logit(model, _mask_motif_nodes(batch, top_nodes))
                top_effect = float(sign * (z - ztop))
                top_ids = {int(r["motif_id"]) for r in top}
                pool = [r for r in motif_candidates if int(r["motif_id"]) not in top_ids]
                random_effects: List[float] = []
                for _ in range(controls):
                    rr = _matched_random_motif_set(top, pool, rng)
                    if len(rr) != k:
                        random_effects = []
                        break
                    nodes = [int(r["motif_node_id"]) for r in rr]
                    _, zr, _ = _prob_and_logit(model, _mask_motif_nodes(batch, nodes))
                    random_effects.append(float(sign * (z - zr)))
                if len(random_effects) != controls:
                    motif_eligible = False
                    break
                rnd = float(np.mean(random_effects))
                motif_rows_for_molecule.append({
                    "level": "motif_occlusion", "row_id": int(rec.row_id), "case": case,
                    "label": int(rec.label), "prediction": pred, "k": k,
                    "top_effect": top_effect, "random_effect": rnd,
                    "gap": top_effect - rnd,
                })
        if motif_eligible and len(motif_rows_for_molecule) == len(ks):
            rows.extend(motif_rows_for_molecule)

        # -------------------------------------------------------------
        # C) Explicit PIG edge ranking -> cumulative edge deletion.
        # Keep molecule only if all k values have enough explicit
        # relations and matched-random relation controls.
        # -------------------------------------------------------------
        rel_all = _explicit_relation_rows(
            _relation_occlusion(model, batch, item, p, z, full_out)
        )
        for r in rel_all:
            r["prediction_support"] = float(sign * float(r.get("logit_contribution", 0.0)))
        ranked_rel = sorted(
            [r for r in rel_all if float(r["prediction_support"]) > 0],
            key=lambda r: -float(r["prediction_support"]),
        )
        rel_rows_for_molecule: List[Dict] = []
        rel_eligible = len(ranked_rel) >= 3 and len(rel_all) >= 6
        if rel_eligible:
            for k in ks:
                top = ranked_rel[:k]
                top_pairs = [(int(r["node_i"]), int(r["node_j"])) for r in top]
                _, ztop, _ = _prob_and_logit(model, _mask_pair_edges(batch, top_pairs))
                top_effect = float(sign * (z - ztop))
                top_keys = {(int(r["motif_i"]), int(r["motif_j"])) for r in top}
                pool = [
                    r for r in rel_all
                    if (int(r["motif_i"]), int(r["motif_j"])) not in top_keys
                ]
                random_effects: List[float] = []
                for _ in range(controls):
                    rr = _matched_random_relation_set(top, pool, rng)
                    if len(rr) != k:
                        random_effects = []
                        break
                    pairs = [(int(r["node_i"]), int(r["node_j"])) for r in rr]
                    _, zr, _ = _prob_and_logit(model, _mask_pair_edges(batch, pairs))
                    random_effects.append(float(sign * (z - zr)))
                if len(random_effects) != controls:
                    rel_eligible = False
                    break
                rnd = float(np.mean(random_effects))
                rel_rows_for_molecule.append({
                    "level": "pig_edge_deletion", "row_id": int(rec.row_id), "case": case,
                    "label": int(rec.label), "prediction": pred, "k": k,
                    "top_effect": top_effect, "random_effect": rnd,
                    "gap": top_effect - rnd,
                })
        if rel_eligible and len(rel_rows_for_molecule) == len(ks):
            rows.extend(rel_rows_for_molecule)

        if idx == 1 or idx % 10 == 0 or idx == len(chosen):
            print(f"  primary faithfulness {idx}/{len(chosen)}", flush=True)

    detail = pd.DataFrame(rows)

    # Defensive invariant: within each explanation level, n must be identical
    # for k=1,2,3 after the all-k eligibility rule above.
    if not detail.empty:
        for level, g in detail.groupby("level"):
            counts = g.groupby("k")["row_id"].nunique().to_dict()
            if counts and len(set(counts.values())) != 1:
                raise RuntimeError(
                    f"Primary faithfulness invariant failed for {level}: "
                    f"molecule counts differ across k: {counts}"
                )

    summary = _summarize_primary_faithfulness(detail, bootstrap_reps, seed)
    return detail, summary

def _draw_primary_explanation_faithfulness(
    summary: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """Main-paper Fig. 6: quantitative validation of primary explanations."""
    _set_publication_style()
    levels = ["atom_ig", "motif_occlusion", "pig_edge_deletion"]
    titles = [
        "(a) Atom-level IG",
        "(b) Hierarchical motif nodes",
        "(c) PIG relations",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.9), sharey=False)
    for ax, level, title in zip(axes, levels, titles):
        g = summary[summary["level"] == level].sort_values("k") if not summary.empty else pd.DataFrame()
        if g.empty:
            ax.text(0.5, 0.5, "Insufficient eligible molecules", ha="center", va="center")
            ax.set_title(title, fontweight="bold")
            ax.axis("off")
            continue
        x = np.arange(len(g), dtype=float)
        w = 0.34
        top = g["mean_top_effect"].to_numpy(float)
        rnd = g["mean_random_effect"].to_numpy(float)
        top_err = np.vstack([
            np.maximum(0.0, top - g["top_ci_low"].to_numpy(float)),
            np.maximum(0.0, g["top_ci_high"].to_numpy(float) - top),
        ])
        rnd_err = np.vstack([
            np.maximum(0.0, rnd - g["random_ci_low"].to_numpy(float)),
            np.maximum(0.0, g["random_ci_high"].to_numpy(float) - rnd),
        ])
        ax.bar(x - w/2, top, width=w, yerr=top_err, capsize=4, label="Top-ranked")
        ax.bar(x + w/2, rnd, width=w, yerr=rnd_err, capsize=4, label="Matched random")
        ax.axhline(0.0, linewidth=0.9)
        ax.set_xticks(x, [f"k={int(v)}" for v in g["k"]])
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xlabel("Number of perturbed elements")
        ax.set_ylabel("Predicted-class support drop (Δlogit)")
        ymax = max(float(np.nanmax(top_err[1] + top)), float(np.nanmax(rnd_err[1] + rnd)), 1e-8)
        ymin = min(float(np.nanmin(top - top_err[0])), float(np.nanmin(rnd - rnd_err[0])), 0.0)
        span = max(ymax - ymin, 1e-8)
        ax.set_ylim(ymin - 0.08 * span, ymax + 0.24 * span)
        for xi, (_, row) in zip(x, g.iterrows()):
            ax.text(
                xi,
                max(float(row["top_ci_high"]), float(row["random_ci_high"])) + 0.06 * span,
                f"q={float(row['bh_q']):.3g}\nn={int(row['n_molecules'])}",
                ha="center", va="bottom", fontsize=8.4,
            )
        ax.legend(frameon=False, fontsize=8.8)
    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.98])
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)


def _fusion_gate_summary(
    detail: pd.DataFrame,
    bootstrap_reps: int,
    seed: int,
) -> pd.DataFrame:
    """Summarize adaptive-fusion gates and effective projected contributions."""
    if detail.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    order = [c for c in ("TP", "TN", "FP", "FN") if c in set(detail["case"].astype(str))]
    for ci, case in enumerate(order):
        g = detail[detail["case"] == case].copy()
        row: Dict = {"case": case, "n": int(len(g))}
        metrics = [
            "alpha_graph", "alpha_evidence",
            "graph_projected_norm", "evidence_projected_norm",
            "graph_weighted_norm", "evidence_weighted_norm",
            "rho_graph", "rho_evidence",
        ]
        for mi, col in enumerate(metrics):
            vals = g[col].to_numpy(float)
            mean, lo, hi = _bootstrap_mean_ci(
                vals, bootstrap_reps, seed + 5000 + ci * 503 + mi * 37
            )
            row[f"mean_{col}"] = mean
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _fusion_gate_analysis(
    model,
    feat_cfg,
    scaler,
    device,
    records: Sequence[Record],
    threshold: float,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str],
    bootstrap_reps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collect adaptive-fusion gates and effective graph/evidence contributions.

    The raw softmax gates alpha_G/alpha_E are not interpreted as contribution
    magnitudes by themselves. For the final FHGN-inspired adaptive fusion,

        h_F = LN(alpha_G W_G h_G + alpha_E W_E h_E + b),

    this diagnostic also computes

        c_G = ||alpha_G W_G h_G||_2,
        c_E = ||alpha_E W_E h_E||_2,
        rho_G = c_G / (c_G + c_E),
        rho_E = 1 - rho_G.

    rho therefore reports the relative magnitude of the two projected weighted
    branch contributions before the learned bias and final LayerNorm.
    """
    fusion = getattr(model, "fusion", None)
    required = ("summarize_evidence", "graph_proj", "evidence_proj")
    if fusion is None or not all(hasattr(fusion, name) for name in required):
        return pd.DataFrame(), pd.DataFrame()

    ds = HERAHGTDataset(records, feat_cfg, scaler, cache_dir)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_hera_hgt,
    )
    rows: List[Dict] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            gates = out.get("fusion_weights")
            global_h = out.get("global_repr")
            evidence_tokens = out.get("evidence_tokens")

            if (
                not torch.is_tensor(gates) or gates.ndim != 2 or gates.shape[1] != 2
                or not torch.is_tensor(global_h) or global_h.ndim != 2
                or not torch.is_tensor(evidence_tokens) or evidence_tokens.ndim != 3
            ):
                return pd.DataFrame(), pd.DataFrame()

            h_e = fusion.summarize_evidence(evidence_tokens)
            h_g_proj = fusion.graph_proj(global_h)
            h_e_proj = fusion.evidence_proj(h_e)

            weighted_g = gates[:, 0:1] * h_g_proj
            weighted_e = gates[:, 1:2] * h_e_proj

            proj_g_norm = torch.linalg.vector_norm(h_g_proj, ord=2, dim=-1)
            proj_e_norm = torch.linalg.vector_norm(h_e_proj, ord=2, dim=-1)
            c_g = torch.linalg.vector_norm(weighted_g, ord=2, dim=-1)
            c_e = torch.linalg.vector_norm(weighted_e, ord=2, dim=-1)
            denom = (c_g + c_e).clamp_min(1e-12)
            rho_g = c_g / denom
            rho_e = c_e / denom

            g_np = gates.detach().cpu().numpy()
            proj_g_np = proj_g_norm.detach().cpu().numpy()
            proj_e_np = proj_e_norm.detach().cpu().numpy()
            c_g_np = c_g.detach().cpu().numpy()
            c_e_np = c_e.detach().cpu().numpy()
            rho_g_np = rho_g.detach().cpu().numpy()
            rho_e_np = rho_e.detach().cpu().numpy()
            probs = out["prob"].detach().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy().astype(int)
            row_ids = batch["row_id"].detach().cpu().numpy().astype(int)

            for rid, y, p, gate, ng, ne, cg, ce, rg, re in zip(
                row_ids, labels, probs, g_np,
                proj_g_np, proj_e_np, c_g_np, c_e_np, rho_g_np, rho_e_np,
            ):
                pred = int(float(p) >= threshold)
                rows.append({
                    "row_id": int(rid),
                    "label": int(y),
                    "prediction": pred,
                    "case": _case_name(int(y), pred),
                    "p_blocker": float(p),
                    "alpha_graph": float(gate[0]),
                    "alpha_evidence": float(gate[1]),
                    "graph_projected_norm": float(ng),
                    "evidence_projected_norm": float(ne),
                    "graph_weighted_norm": float(cg),
                    "evidence_weighted_norm": float(ce),
                    "rho_graph": float(rg),
                    "rho_evidence": float(re),
                })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()

    # Numerical sanity checks.
    if not np.allclose(
        detail["alpha_graph"].to_numpy(float) + detail["alpha_evidence"].to_numpy(float),
        1.0, atol=1e-5,
    ):
        raise RuntimeError("Adaptive fusion gates do not sum to one")
    if not np.allclose(
        detail["rho_graph"].to_numpy(float) + detail["rho_evidence"].to_numpy(float),
        1.0, atol=1e-5,
    ):
        raise RuntimeError("Effective contribution ratios do not sum to one")

    summary = _fusion_gate_summary(detail, bootstrap_reps, seed)
    return detail, summary


def _draw_fusion_gate_diagnostic(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
    dpi: int,
) -> None:
    """Supplementary adaptive-fusion diagnostic with gate and effective contribution."""
    _set_publication_style()
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.7))

    if detail.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "Adaptive fusion diagnostics unavailable", ha="center", va="center")
            ax.axis("off")
    else:
        # (a) Raw graph gate alpha_G.
        ax = axes[0]
        vals = detail["alpha_graph"].to_numpy(float)
        ax.hist(vals, bins=np.linspace(0.0, 1.0, 21), edgecolor="white")
        ax.axvline(
            float(np.mean(vals)), linewidth=1.6, linestyle="--",
            label=fr"mean $\alpha_G$={np.mean(vals):.3f}",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Graph gate $\alpha_G$")
        ax.set_ylabel("Molecules")
        ax.set_title("(a)", fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5)

        # (b) Effective graph contribution rho_G.
        ax = axes[1]
        vals = detail["rho_graph"].to_numpy(float)
        ax.hist(vals, bins=np.linspace(0.0, 1.0, 21), edgecolor="white")
        ax.axvline(
            float(np.mean(vals)), linewidth=1.6, linestyle="--",
            label=fr"mean $\rho_G$={np.mean(vals):.3f}",
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Effective graph contribution $\rho_G$")
        ax.set_ylabel("Molecules")
        ax.set_title("(b)", fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5)

        # (c) Effective contribution ratios by prediction case, bootstrap 95% CI.
        ax = axes[2]
        if summary.empty:
            ax.text(0.5, 0.5, "Case-level summary unavailable", ha="center", va="center")
            ax.axis("off")
        else:
            order = [c for c in ("TP", "TN", "FP", "FN") if c in set(summary["case"].astype(str))]
            ss = summary.set_index("case").loc[order].reset_index()
            rg = ss["mean_rho_graph"].to_numpy(float)
            re = ss["mean_rho_evidence"].to_numpy(float)
            rg_err = np.vstack([
                np.maximum(0.0, rg - ss["rho_graph_ci_low"].to_numpy(float)),
                np.maximum(0.0, ss["rho_graph_ci_high"].to_numpy(float) - rg),
            ])
            re_err = np.vstack([
                np.maximum(0.0, re - ss["rho_evidence_ci_low"].to_numpy(float)),
                np.maximum(0.0, ss["rho_evidence_ci_high"].to_numpy(float) - re),
            ])
            x = np.arange(len(order), dtype=float)
            w = 0.36
            ax.bar(x - w/2, rg, width=w, yerr=rg_err, capsize=4, label=r"$\rho_G$ graph")
            ax.bar(x + w/2, re, width=w, yerr=re_err, capsize=4, label=r"$\rho_E$ evidence")
            ax.set_xticks(x, order)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Mean effective contribution ratio")
            ax.set_title("(c)", fontweight="bold")
            for xi, n in zip(x, ss["n"].astype(int).tolist()):
                ax.text(xi, 0.98, f"n={n}", ha="center", va="top", fontsize=8.5)
            ax.legend(frameon=False, fontsize=8.5)

    fig.tight_layout(rect=[0.01, 0.03, 0.99, 0.98])
    _save_publication_figure(fig, out_path, dpi)
    plt.close(fig)

def _global_pass(model, feat_cfg, scaler, device, records: Sequence[Record], threshold: float, relation_candidates: int, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frag_all: List[Dict] = []
    motif_all: List[Dict] = []
    fam_all: List[Dict] = []
    rel_all: List[Dict] = []
    total = len(records)
    for idx, rec in enumerate(records, start=1):
        item, can = _build_single_item(rec.smiles, rec.label, rec.row_id, feat_cfg, scaler)
        batch = move_batch(collate_hera_hgt([item]), device)
        p, z, full_out = _prob_and_logit(model, batch)
        pred, case = int(p >= threshold), _case_name(rec.label, int(p >= threshold))
        with torch.no_grad():
            atom_proj = model.graph_encoder.node_proj(batch["node_x"][:, : int(item["n_atoms"])]).squeeze(0)
        fr = _fragment_occlusion(model, batch, item, p, z, atom_proj)
        mo, fa = _motif_and_family_occlusion(model, batch, item, p, z)
        _add_motif_attention(mo, full_out, item)
        mo = _explicit_motif_rows(mo)
        candidates = _relation_candidates_by_attention(batch, item, full_out)[:relation_candidates]
        # Attention only screens a small candidate set for scalable global edge
        # perturbation. The reported effect is direct deletion/neutralization.
        rr = _perturb_relation_candidates(model, batch, candidates, p, z)
        for rows in (fr, mo, fa, rr):
            _annotate_rows(rows, rec.row_id, can, rec.label, pred, case)
        frag_all.extend(fr)
        motif_all.extend(mo)
        fam_all.extend(fa)
        rel_all.extend(rr)
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"  global {idx}/{total}", flush=True)
    return pd.DataFrame(frag_all), pd.DataFrame(motif_all), pd.DataFrame(fam_all), pd.DataFrame(rel_all)


def _bootstrap_ci_mean(values: Sequence[float], reps: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Deterministic molecule-level bootstrap CI for a mean."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1 or reps <= 1:
        return mean, mean, mean
    rng = np.random.default_rng(int(seed))
    # Chunked resampling avoids large temporary allocations for common families.
    boot = np.empty(int(reps), dtype=np.float64)
    chunk = 256
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = rng.integers(0, arr.size, size=(m, arr.size))
        boot[done:done + m] = arr[idx].mean(axis=1)
        done += m
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _bootstrap_ci_contrast(
    blocker_values: Sequence[float],
    nonblocker_values: Sequence[float],
    reps: int = 2000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% CI for mean(blocker) - mean(non-blocker)."""
    b = np.asarray(list(blocker_values), dtype=np.float64)
    n = np.asarray(list(nonblocker_values), dtype=np.float64)
    b = b[np.isfinite(b)]
    n = n[np.isfinite(n)]
    if b.size == 0 or n.size == 0:
        return float("nan"), float("nan"), float("nan")
    contrast = float(np.mean(b) - np.mean(n))
    if reps <= 1:
        return contrast, contrast, contrast
    rng = np.random.default_rng(int(seed))
    boot = np.empty(int(reps), dtype=np.float64)
    chunk = 256
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        ib = rng.integers(0, b.size, size=(m, b.size))
        inn = rng.integers(0, n.size, size=(m, n.size))
        boot[done:done + m] = b[ib].mean(axis=1) - n[inn].mean(axis=1)
        done += m
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return contrast, float(lo), float(hi)


def _global_summary_tables(
    motif_df: pd.DataFrame,
    family_df: pd.DataFrame,
    relation_df: pd.DataFrame,
    bootstrap_reps: int = 2000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Molecule-aware global aggregation with deterministic bootstrap 95% CIs.

    The resampling unit is the molecule, not the individual motif occurrence.
    This avoids pseudo-replication when one molecule contains multiple matching
    occurrences of the same motif/relation.  For each retained group we report
    class-specific mean effects and a direct bootstrap CI for the blocker-minus-
    non-blocker class contrast.
    """

    def summarize(df: pd.DataFrame, group_cols: List[str], seed_offset: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        x = df.copy()
        x["logit_contribution"] = pd.to_numeric(x["logit_contribution"], errors="coerce")
        x = x[np.isfinite(x["logit_contribution"])].copy()
        if x.empty:
            return pd.DataFrame()

        # First collapse repeated occurrences within one molecule/group.
        per_mol = x.groupby(["row_id", "label"] + group_cols, as_index=False, dropna=False)["logit_contribution"].mean()
        per_mol["abs_logit"] = per_mol["logit_contribution"].abs()

        base = per_mol.groupby(["label"] + group_cols, as_index=False, dropna=False).agg(
            n=("row_id", "nunique"),
            mean_logit=("logit_contribution", "mean"),
            median_logit=("logit_contribution", "median"),
            std_logit=("logit_contribution", "std"),
            mean_abs_logit=("abs_logit", "mean"),
        )
        base["std_logit"] = base["std_logit"].fillna(0.0)

        class_ci_rows: List[Dict] = []
        contrast_rows: List[Dict] = []
        grouped = per_mol.groupby(group_cols, dropna=False, sort=True)
        for gi, (gkey, g) in enumerate(grouped):
            if not isinstance(gkey, tuple):
                gkey = (gkey,)
            key_payload = dict(zip(group_cols, gkey))
            stable = zlib.crc32(("|".join(map(str, gkey))).encode("utf-8")) & 0xFFFFFFFF
            vals_by_label: Dict[int, np.ndarray] = {}
            for lab in (0, 1):
                vals = g.loc[g["label"].astype(int) == lab, "logit_contribution"].to_numpy(dtype=float)
                vals_by_label[lab] = vals
                if vals.size:
                    mean, lo, hi = _bootstrap_ci_mean(
                        vals,
                        reps=int(bootstrap_reps),
                        seed=int(seed + seed_offset + stable + 1009 * lab),
                    )
                    class_ci_rows.append({
                        **key_payload,
                        "label": int(lab),
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci95": max(mean - lo, hi - mean),
                        "bootstrap_reps": int(bootstrap_reps),
                    })
            if vals_by_label[0].size and vals_by_label[1].size:
                contrast, lo, hi = _bootstrap_ci_contrast(
                    vals_by_label[1], vals_by_label[0],
                    reps=int(bootstrap_reps),
                    seed=int(seed + seed_offset + stable + 7919),
                )
                contrast_rows.append({
                    **key_payload,
                    "contrast": contrast,
                    "contrast_ci_low": lo,
                    "contrast_ci_high": hi,
                    "bootstrap_reps": int(bootstrap_reps),
                })

        if class_ci_rows:
            base = base.merge(pd.DataFrame(class_ci_rows), on=["label"] + group_cols, how="left")
        else:
            base["ci_low"] = base["mean_logit"]
            base["ci_high"] = base["mean_logit"]
            base["ci95"] = 0.0
            base["bootstrap_reps"] = int(bootstrap_reps)
        if contrast_rows:
            base = base.merge(pd.DataFrame(contrast_rows), on=group_cols, how="left", suffixes=("", "_contrast"))
            if "bootstrap_reps_contrast" in base.columns:
                base.drop(columns=["bootstrap_reps_contrast"], inplace=True)
        else:
            base["contrast"] = np.nan
            base["contrast_ci_low"] = np.nan
            base["contrast_ci_high"] = np.nan
        return base

    fam_summary = summarize(family_df, ["motif_family_short", "motif_family"], 11000) if not family_df.empty else pd.DataFrame()
    motif_summary = summarize(motif_df, ["motif_name", "motif_family_short"], 22000) if not motif_df.empty else pd.DataFrame()

    rel = relation_df.copy()
    if rel.empty:
        rel_summary = pd.DataFrame()
    else:
        rel["family_pair"] = rel.apply(
            lambda r: "–".join(sorted([str(r["family_i"]), str(r["family_j"])])), axis=1
        )
        rel_summary = summarize(rel, ["relation_name", "family_pair"], 33000)
        if "aux_last_layer_attention" in rel.columns and not rel_summary.empty:
            att = rel.groupby(["label", "relation_name", "family_pair"], as_index=False, dropna=False)["aux_last_layer_attention"].mean().rename(
                columns={"aux_last_layer_attention": "mean_attention"}
            )
            rel_summary = rel_summary.merge(att, on=["label", "relation_name", "family_pair"], how="left")
    return fam_summary, motif_summary, rel_summary

def _draw_global_q1(fam_summary: pd.DataFrame, motif_summary: pd.DataFrame, rel_summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    """Render the global interpretation as THREE independent journal figures.

    The former 3-panel layout was too dense after manuscript down-scaling,
    especially for long motif/relation labels.  This renderer deliberately
    writes Fig. 5A, 5B and 5C separately so every label, support count and
    uncertainty interval remains readable in a Q1 journal layout.

    ``out_path`` is kept in the signature for backward compatibility with the
    rest of the pipeline; only its parent directory is used.
    """
    _set_publication_style()
    article_dir = Path(out_path).parent
    article_dir.mkdir(parents=True, exist_ok=True)

    # Remove the obsolete combined Fig. 5 so the manuscript folder contains
    # one unambiguous final set only.
    legacy = article_dir / "Fig_Interp5_Global_Interpretability.png"
    for ext in (".png", ".pdf", ".svg"):
        q = legacy.with_suffix(ext)
        if q.exists():
            q.unlink()
    legacy_family = article_dir / "Fig_Interp5A_Global_Pharmacophore_Families.png"
    for ext in (".png", ".pdf", ".svg"):
        q = legacy_family.with_suffix(ext)
        if q.exists():
            q.unlink()

    # ------------------------------------------------------------------
    # Shared helper: class contrast with molecule-level bootstrap 95% CI.
    # ------------------------------------------------------------------
    def contrast_table(df: pd.DataFrame, keys: List[str], min_n: int, top_n: int) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        v = df.copy()
        val = v.pivot_table(index=keys, columns="label", values="mean_logit")
        cnt = v.pivot_table(index=keys, columns="label", values="n", aggfunc="max")
        if 0 not in val.columns or 1 not in val.columns:
            return pd.DataFrame()
        out = val.copy()
        out["contrast"] = out[1] - out[0]
        if 0 in cnt.columns and 1 in cnt.columns:
            out["nN"] = cnt[0]
            out["nB"] = cnt[1]
            out = out[(out["nN"] >= min_n) & (out["nB"] >= min_n)]
        else:
            out["nN"] = np.nan
            out["nB"] = np.nan

        # New summaries contain a direct bootstrap CI for the contrast.
        if {"contrast_ci_low", "contrast_ci_high"}.issubset(v.columns):
            extra_cols = keys + ["contrast_ci_low", "contrast_ci_high"]
            if "bootstrap_reps" in v.columns:
                extra_cols.append("bootstrap_reps")
            extra = v[extra_cols].drop_duplicates(keys)
            out = out.reset_index().merge(extra, on=keys, how="left").set_index(keys)
        else:
            # Backward-compatible fallback for legacy caches.
            ci = v.pivot_table(index=keys, columns="label", values="ci95") if "ci95" in v.columns else None
            if ci is not None and 0 in ci.columns and 1 in ci.columns:
                half = np.sqrt(ci[1].pow(2) + ci[0].pow(2))
                out["contrast_ci_low"] = out["contrast"] - half
                out["contrast_ci_high"] = out["contrast"] + half
            else:
                out["contrast_ci_low"] = out["contrast"]
                out["contrast_ci_high"] = out["contrast"]
        out["contrast_err_low"] = (out["contrast"] - out["contrast_ci_low"]).clip(lower=0.0)
        out["contrast_err_high"] = (out["contrast_ci_high"] - out["contrast"]).clip(lower=0.0)
        out["abs_contrast"] = out["contrast"].abs()
        return out.sort_values("abs_contrast", ascending=False).head(top_n).reset_index()

    # ================================================================
    # Fig. 5A — hierarchical motif-family effects
    # ================================================================
    fig = plt.figure(figsize=(11.8, 8.6))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.18)
    if fam_summary.empty:
        ax.text(0.5, 0.5, "No family data", ha="center", va="center", fontsize=15)
        ax.axis("off")
    else:
        mean = fam_summary.pivot_table(index="motif_family_short", columns="label", values="mean_logit")
        ci = fam_summary.pivot_table(index="motif_family_short", columns="label", values="ci95")
        ci_low = fam_summary.pivot_table(index="motif_family_short", columns="label", values="ci_low") if "ci_low" in fam_summary.columns else None
        ci_high = fam_summary.pivot_table(index="motif_family_short", columns="label", values="ci_high") if "ci_high" in fam_summary.columns else None
        n = fam_summary.pivot_table(index="motif_family_short", columns="label", values="n", aggfunc="max")
        order = [x for x in ["BN", "AR", "NHC", "PF", "STR"] if x in mean.index]
        mean = mean.reindex(order)
        ci = ci.reindex(order)
        if ci_low is not None: ci_low = ci_low.reindex(order)
        if ci_high is not None: ci_high = ci_high.reindex(order)
        n = n.reindex(order)
        for col in [0, 1]:
            if col not in mean.columns:
                mean[col] = 0.0
            if col not in ci.columns:
                ci[col] = 0.0
            if col not in n.columns:
                n[col] = 0
        x = np.arange(len(order))
        w = 0.34
        if ci_low is not None and ci_high is not None and 0 in ci_low.columns and 1 in ci_low.columns:
            yerr_b = np.vstack([(mean[1] - ci_low[1]).clip(lower=0).to_numpy(float),
                                (ci_high[1] - mean[1]).clip(lower=0).to_numpy(float)])
            yerr_n = np.vstack([(mean[0] - ci_low[0]).clip(lower=0).to_numpy(float),
                                (ci_high[0] - mean[0]).clip(lower=0).to_numpy(float)])
        else:
            yerr_b = ci[1].to_numpy(float)
            yerr_n = ci[0].to_numpy(float)
        ax.bar(x - w/2, mean[1].to_numpy(float), width=w, yerr=yerr_b,
               color="#EF635F", alpha=0.92, capsize=5, label="True blocker")
        ax.bar(x + w/2, mean[0].to_numpy(float), width=w, yerr=yerr_n,
               color="#4A90E2", alpha=0.92, capsize=5, label="True non-blocker")
        ax.axhline(0, color="#374151", lw=1.15)
        ax.set_xticks(x, order, fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11.5)
        ax.set_ylabel("Mean family-occlusion Δlogit", fontsize=13)
        ax.grid(axis="y", color=GRID_COLOR, lw=0.7, alpha=0.72)
        ax.legend(frameon=False, fontsize=12, loc="upper right")

        # Add class support below each family without crowding the bars.
        for i, fam in enumerate(order):
            nb = int(n.loc[fam, 1]) if fam in n.index and pd.notna(n.loc[fam, 1]) else 0
            nn = int(n.loc[fam, 0]) if fam in n.index and pd.notna(n.loc[fam, 0]) else 0
            ax.text(i, -0.115, f"nB={nb}\nnN={nn}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9.5, color="#4B5563")

    _save_publication_figure(fig, article_dir / "Fig_Interp5A_Global_Motif_Family_Effects.png", dpi)
    plt.close(fig)

    # ================================================================
    # Fig. 5B — explicit motifs by class contrast
    # Clean publication layout: no title/subtitle/legend/value labels inside
    # the image. Explanatory details belong in the manuscript caption.
    # ================================================================
    gm = contrast_table(motif_summary, ["motif_name", "motif_family_short"], min_n=20, top_n=6)
    fig = plt.figure(figsize=(11.8, 6.8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.34, right=0.96, top=0.96, bottom=0.16)

    if gm.empty:
        ax.text(0.5, 0.5, "Insufficient explicit-motif data",
                ha="center", va="center", fontsize=15)
        ax.axis("off")
    else:
        gm = gm.iloc[::-1].reset_index(drop=True)
        vals = gm["contrast"].to_numpy(float)
        err_low = gm["contrast_err_low"].fillna(0.0).to_numpy(float)
        err_high = gm["contrast_err_high"].fillna(0.0).to_numpy(float)
        errs = np.vstack([err_low, err_high])
        colors = [POS_COLOR if x > 0 else NEG_COLOR for x in vals]

        # Keep only the chemically meaningful label in the plot.
        # Support counts (nB/nN) are reported in the manuscript caption/table.
        labels = [
            f"{_short_text(r.motif_name, 26)} [{r.motif_family_short}]"
            for r in gm.itertuples(index=False)
        ]

        y = np.arange(len(gm))
        ax.barh(
            y,
            vals,
            xerr=errs,
            color=colors,
            alpha=0.88,
            error_kw=dict(
                ecolor="#374151",
                elinewidth=1.1,
                capsize=4,
                capthick=1.0,
            ),
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=10.8)
        ax.axvline(0.0, color="#374151", lw=1.15)
        ax.set_xlabel("Blocker–non-blocker contrast in Δlogit", fontsize=11.5, labelpad=7)
        ax.grid(axis="x", color=GRID_COLOR, lw=0.7, alpha=0.72)

        # Preserve the full bootstrap CI range with modest visual padding.
        lo_vals = vals - err_low
        hi_vals = vals + err_high
        lim = max(
            float(np.max(np.maximum(np.abs(lo_vals), np.abs(hi_vals)))),
            1e-12,
        )
        ax.set_xlim(
            min(-1.18 * lim, float(np.min(lo_vals)) * 1.12),
            max( 1.18 * lim, float(np.max(hi_vals)) * 1.12),
        )

    _save_publication_figure(
        fig,
        article_dir / "Fig_Interp5B_Global_Explicit_Motifs.png",
        dpi,
    )
    plt.close(fig)

    # ================================================================
    # Fig. 5C — PIG relation effects by class contrast
    # Clean publication layout: no title/subtitle/legend/value labels inside
    # the image. Explanatory details belong in the manuscript caption.
    # ================================================================
    gr = contrast_table(rel_summary, ["relation_name", "family_pair"], min_n=20, top_n=6)
    fig = plt.figure(figsize=(11.8, 6.8))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.31, right=0.96, top=0.96, bottom=0.16)

    if gr.empty:
        ax.text(0.5, 0.5, "Insufficient PIG-relation data",
                ha="center", va="center", fontsize=15)
        ax.axis("off")
    else:
        gr = gr.iloc[::-1].reset_index(drop=True)
        vals = gr["contrast"].to_numpy(float)
        err_low = gr["contrast_err_low"].fillna(0.0).to_numpy(float)
        err_high = gr["contrast_err_high"].fillna(0.0).to_numpy(float)
        errs = np.vstack([err_low, err_high])
        colors = [POS_COLOR if x > 0 else NEG_COLOR for x in vals]

        # Keep only the compact relation name in the plot.
        # Support counts (nB/nN) are reported in the manuscript caption/table.
        labels = [
            _short_text(
                _compact_relation_display(r.relation_name, r.family_pair),
                24,
            )
            for r in gr.itertuples(index=False)
        ]

        y = np.arange(len(gr))
        ax.barh(
            y,
            vals,
            xerr=errs,
            color=colors,
            alpha=0.88,
            error_kw=dict(
                ecolor="#374151",
                elinewidth=1.1,
                capsize=4,
                capthick=1.0,
            ),
        )

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=11.5)
        ax.tick_params(axis="x", labelsize=10.8)
        ax.axvline(0.0, color="#374151", lw=1.15)
        ax.set_xlabel("Blocker–non-blocker relation contrast in Δlogit", fontsize=11.5, labelpad=7)
        ax.grid(axis="x", color=GRID_COLOR, lw=0.7, alpha=0.72)

        # Preserve the full bootstrap CI range with modest visual padding.
        lo_vals = vals - err_low
        hi_vals = vals + err_high
        lim = max(
            float(np.max(np.maximum(np.abs(lo_vals), np.abs(hi_vals)))),
            1e-12,
        )
        ax.set_xlim(
            min(-1.18 * lim, float(np.min(lo_vals)) * 1.12),
            max( 1.18 * lim, float(np.max(hi_vals)) * 1.12),
        )

    _save_publication_figure(
        fig,
        article_dir / "Fig_Interp5C_Global_PIG_Relations.png",
        dpi,
    )
    plt.close(fig)

def _global_atom_sample(model, feat_cfg, scaler, device, pred_df: pd.DataFrame, record_map: Dict[int, Record], threshold: float, n_per_class: int, steps: int) -> pd.DataFrame:
    rows: List[Dict] = []
    for case in ("TP", "TN"):
        g = pred_df[(pred_df["case"] == case) & (pred_df["eligible_base"])].copy()
        chosen = _interior_quantile_pick(g, min(n_per_class, len(g)))
        for rr in chosen.itertuples(index=False):
            rec = record_map[int(rr.row_id)]
            item, can = _build_single_item(rec.smiles, rec.label, rec.row_id, feat_cfg, scaler)
            batch = move_batch(collate_hera_hgt([item]), device)
            ig = integrated_gradients_raw_atoms(model, batch, item, steps)
            pred = int(float(ig["normal_prob"]) >= threshold)
            rows.extend(_atom_rows(rec.row_id, can, rec.label, pred, _case_name(rec.label, pred), float(ig["normal_prob"]), np.asarray(ig["atom_scores"])))
    return pd.DataFrame(rows)


def _write_q1_report(out_dir: Path, args: argparse.Namespace, selections: Dict[str, pd.DataFrame], threshold: float, n_screened: int, train_overlap_excluded: bool) -> None:
    lines = [
        "# RHMGT Q1 Interpretability Report",
        "",
        "## Protocol",
        f"- Evaluation source: `{args.eval_csv}`." if args.eval_csv else f"- Reconstructed split: `{args.split}` from `{args.split_json}`.",
        f"- Frozen checkpoint: `{args.checkpoint}`.",
        f"- Decision threshold: {threshold:.6f}.",
        f"- Exact train-overlap exclusion for showcase figures: {train_overlap_excluded}.",
        f"- Numerically screened molecules: {n_screened}.",
        f"- Screening IG steps: {args.screen_ig_steps}; final publication IG steps: {args.final_ig_steps}.",
        f"- Structural-diversity constraint: Morgan/Tanimoto <= {args.max_tanimoto:.2f}, relaxed only if needed to fill a class quota.",
        "- Primary motif importance: direct motif-node occlusion (Δ blocker logit).",
        "- Primary PIG relation importance: direct pair-edge deletion (Δ blocker logit).",
        f"- Global uncertainty: molecule-level bootstrap 95% CI with {args.bootstrap_reps} resamples; global Top-6 motif/relation panels require nB,nN >= 20.",
        "- Explicit pharmacophore P identifiers are persistent within each molecule (tied to motif_id) and are reused between Fig. 3 and Fig. 4.",
        "- Explicit feature definitions are non-exclusive; distinct named occurrences may legitimately share atoms.",
        "- Main global contrast figures use a minimum support of nB >= 20 and nN >= 20 for displayed motifs/relations.",
        "- Typed-relation neutralization is exported separately; learned attention is auxiliary and is explicitly tested by perturbation-based faithfulness.",
        "",
        "## Branch-specific showcase selections",
    ]
    for name, df in selections.items():
        lines.append(f"### {name}")
        if df.empty:
            lines.append("No eligible molecules.")
            continue
        for r in df.itertuples(index=False):
            score_col = {"Atoms": "atom_quality", "Fragments": "fragment_quality", "Motif nodes": "motif_quality", "Pharmacophores": "motif_quality", "PIG relations": "relation_quality"}[name]
            lines.append(f"- row {int(r.row_id)} | {r.case} | score={float(getattr(r, score_col)):.4f} | p(blocker)={float(r.p_blocker):.4f}")
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "These outputs describe the trained model's internal predictive sensitivity. They are not experimental evidence of physical ligand-channel contacts or a causal biochemical binding mechanism.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")




def _discover_showcase_row_ids(out_dir: Path) -> List[int]:
    """Recover showcase row IDs from legacy/cached outputs without assuming filenames."""
    ids: List[int] = []
    patterns = [
        out_dir / "raw",
        out_dir / "branch_images",
        out_dir / "branch_images_publication",
    ]
    import re
    for folder in patterns:
        if not folder.exists():
            continue
        for fp in folder.glob("*"):
            m = re.search(r"row[_-](\d+)", fp.name)
            if m:
                ids.append(int(m.group(1)))
    # Also inspect any existing final table, regardless of exact final_* subtype.
    for fp in out_dir.glob("*.csv"):
        if not any(k in fp.name.lower() for k in ("atom", "motif", "pharmac", "pig", "selected", "showcase")):
            continue
        try:
            df = pd.read_csv(fp, usecols=lambda c: c == "row_id")
        except Exception:
            continue
        if "row_id" in df.columns:
            ids.extend(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
    # Stable unique order.
    return list(dict.fromkeys(ids))


def _resolve_eval_csv_for_repair(args: argparse.Namespace) -> Path:
    if args.eval_csv:
        p = Path(args.eval_csv)
        if p.exists():
            return p
    default = Path("data/paper_split/hERGAT_test_df.csv")
    if default.exists():
        return default
    raise FileNotFoundError(
        "Repair/render mode needs the frozen test CSV to reconstruct missing showcase metadata. "
        "Pass --test-csv data/paper_split/hERGAT_test_df.csv."
    )


def _hydrate_selection_from_test(
    row_ids: Sequence[int],
    args: argparse.Namespace,
    model,
    feat_cfg,
    scaler,
    device: torch.device,
    threshold: float,
) -> Tuple[pd.DataFrame, Dict[int, Record]]:
    eval_csv = _resolve_eval_csv_for_repair(args)
    records = load_records_csv(str(eval_csv), args.smiles_col, args.label_col)
    record_map = _records_lookup(records)
    rows: List[Dict] = []
    for rid in row_ids:
        rid = int(rid)
        if rid not in record_map:
            continue
        rec = record_map[rid]
        item, can = _build_single_item(rec.smiles, rec.label, rec.row_id, feat_cfg, scaler)
        batch = move_batch(collate_hera_hgt([item]), device)
        p_blocker, z, _ = _prob_and_logit(model, batch)
        pred = int(p_blocker >= threshold)
        rows.append({
            "row_id": rid,
            "smiles": can,
            "label": int(rec.label),
            "prediction": pred,
            "case": _case_name(rec.label, pred),
            "p_blocker": float(p_blocker),
            "blocker_logit": float(z),
        })
    return pd.DataFrame(rows), record_map


def _balance_showcase_selection(df: pd.DataFrame, n_per_class: int = 2) -> pd.DataFrame:
    if df.empty:
        return df
    # Prefer the already intended main-paper composition: 2 TP + 2 TN.
    picks = []
    if "case" in df.columns:
        for case in ("TP", "TN"):
            g = df[df["case"].astype(str) == case].drop_duplicates("row_id")
            if not g.empty:
                picks.append(g.head(n_per_class))
    if picks:
        out = pd.concat(picks, ignore_index=True)
        if len(out) >= 2 * n_per_class:
            return out.head(2 * n_per_class)
    return df.drop_duplicates("row_id").head(2 * n_per_class).reset_index(drop=True)


def _repair_selected_explanations(
    out_dir: Path,
    selected: pd.DataFrame,
    record_map: Dict[int, Record],
    args: argparse.Namespace,
    model,
    feat_cfg,
    scaler,
    device: torch.device,
    threshold: float,
) -> None:
    """Recompute only the 4 showcase molecules; never rescan the full test set."""
    need_atom = not (out_dir / "final_atom_scores.csv").exists() or not all(
        (out_dir / "raw" / f"row_{int(r)}_raw_input_ig.npz").exists() for r in selected["row_id"]
    )
    need_motif = not (out_dir / "final_motif_node_scores.csv").exists()
    need_pharma = not (out_dir / "final_pharmacophore_scores.csv").exists()
    need_pig = not ((out_dir / "final_pig_nodes.csv").exists() and (out_dir / "final_pig_relations.csv").exists())
    need_frag = not (out_dir / "final_fragment_scores.csv").exists()
    if not any((need_atom, need_motif, need_pharma, need_pig, need_frag)):
        return

    print("Repair mode: recomputing ONLY the selected showcase molecules (not the full test set).", flush=True)
    atom_rows_all: List[Dict] = []
    motif_rows_all: List[Dict] = []
    pharma_rows_all: List[Dict] = []
    family_rows_all: List[Dict] = []
    pig_nodes_all: List[Dict] = []
    pig_rel_all: List[Dict] = []
    frag_rows_all: List[Dict] = []

    for r in selected.itertuples(index=False):
        rid = int(r.row_id)
        rec = record_map.get(rid)
        if rec is None:
            continue
        print(f"  repairing row {rid} ({getattr(r, 'case', '')})", flush=True)
        if need_atom:
            _, rows = _final_atom_entry(model, feat_cfg, scaler, device, rec, threshold, args.final_ig_steps, out_dir, args.dpi)
            atom_rows_all.extend(rows)
        if need_motif:
            _, rows = _final_motif_node_entry(model, feat_cfg, scaler, device, rec, threshold, out_dir, args.dpi, args.top_pharmacophores)
            motif_rows_all.extend(rows)
        if need_pharma:
            _, rows, fam = _final_motif_entry(model, feat_cfg, scaler, device, rec, threshold, out_dir, args.dpi, args.top_pharmacophores)
            pharma_rows_all.extend(rows); family_rows_all.extend(fam)
        if need_pig:
            _, mrows, rrows = _final_relation_entry(model, feat_cfg, scaler, device, rec, threshold)
            pig_nodes_all.extend(mrows); pig_rel_all.extend(rrows)
        if need_frag:
            _, rows = _final_fragment_entry(model, feat_cfg, scaler, device, rec, threshold, out_dir, args.dpi, args.top_fragments)
            frag_rows_all.extend(rows)

    if atom_rows_all:
        pd.DataFrame(atom_rows_all).to_csv(out_dir / "final_atom_scores.csv", index=False)
    if motif_rows_all:
        pd.DataFrame(motif_rows_all).to_csv(out_dir / "final_motif_node_scores.csv", index=False)
    if pharma_rows_all:
        pd.DataFrame(pharma_rows_all).to_csv(out_dir / "final_pharmacophore_scores.csv", index=False)
    if family_rows_all:
        pd.DataFrame(family_rows_all).to_csv(out_dir / "final_family_scores.csv", index=False)
    if pig_nodes_all:
        pd.DataFrame(pig_nodes_all).to_csv(out_dir / "final_pig_nodes.csv", index=False)
    if pig_rel_all:
        pd.DataFrame(pig_rel_all).to_csv(out_dir / "final_pig_relations.csv", index=False)
    if frag_rows_all:
        pd.DataFrame(frag_rows_all).to_csv(out_dir / "final_fragment_scores.csv", index=False)


def _remove_legacy_combined_publication_figures(article_dir: Path) -> None:
    """Remove obsolete layouts so the final article figure folder is unambiguous."""
    legacy_stems = [
        "Fig_Interp2A_Hierarchical_Motifs_Blockers",
        "Fig_Interp2B_Hierarchical_Motifs_NonBlockers",
        "Fig_Interp3A_Explicit_Pharmacophores_Blockers",
        "Fig_Interp3B_Explicit_Pharmacophores_NonBlockers",
        "Fig_Interp4_PIG_Relations_2_Molecules",
        "Fig_Interp5_Global_Interpretability",
    ]
    for stem in legacy_stems:
        for ext in (".png", ".pdf", ".svg"):
            p = article_dir / f"{stem}{ext}"
            if p.exists():
                p.unlink()


def _split_case_entries(entries: Sequence[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return TP entries and TN entries in stable row-id order."""
    tp = [e for e in entries if str(e.get("case")) == "TP"]
    tn = [e for e in entries if str(e.get("case")) == "TN"]
    tp.sort(key=lambda x: int(x.get("row_id", 0)))
    tn.sort(key=lambda x: int(x.get("row_id", 0)))
    return tp, tn


def _render_existing_publication_figures(out_dir: Path, args: argparse.Namespace) -> None:
    """Rebuild publication figures, repairing only the 4 showcase cases if needed."""
    article_dir = out_dir / "article_figures"
    supplementary_dir = out_dir / "supplementary_figures"
    branch_dir = out_dir / "branch_images_publication"
    article_dir.mkdir(parents=True, exist_ok=True)
    supplementary_dir.mkdir(parents=True, exist_ok=True)
    branch_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_combined_publication_figures(article_dir)

    selected_path = out_dir / "shared_hierarchical_showcase_selected.csv"
    selected = pd.read_csv(selected_path) if selected_path.exists() else pd.DataFrame()

    # 1) Recover from screening table when available.
    screening_path = out_dir / "screening_branch_scores.csv"
    if selected.empty and screening_path.exists():
        screening_df = pd.read_csv(screening_path)
        selected = _select_shared_hierarchy_panel(screening_df, args.branch_per_class, args.max_tanimoto)

    # 2) Recover metadata from any per-molecule final table.
    if selected.empty:
        frames = []
        for rp in [
            out_dir / "final_atom_scores.csv",
            out_dir / "final_motif_node_scores.csv",
            out_dir / "final_pharmacophore_scores.csv",
            out_dir / "final_pig_nodes.csv",
        ]:
            if not rp.exists():
                continue
            try:
                rdf = pd.read_csv(rp)
            except Exception:
                continue
            cols = [c for c in ("row_id", "smiles", "label", "prediction", "case", "p_blocker") if c in rdf.columns]
            if "row_id" in cols:
                frames.append(rdf[cols].drop_duplicates("row_id"))
        if frames:
            selected = _balance_showcase_selection(pd.concat(frames, ignore_index=True, sort=False), args.branch_per_class)

    # 3) Recover row IDs from the command or from legacy raw/branch image filenames.
    recovered_ids: List[int] = []
    if selected.empty:
        if args.selected_row_ids:
            recovered_ids = [int(x) for x in args.selected_row_ids]
        else:
            recovered_ids = _discover_showcase_row_ids(out_dir)

    # If selection metadata is incomplete OR only row IDs were recovered, hydrate
    # it using the frozen test CSV and checkpoint. This is cheap: only a few rows.
    required_cols = {"row_id", "smiles", "label", "prediction", "case", "p_blocker"}
    needs_hydration = selected.empty or not required_cols.issubset(set(selected.columns))
    if needs_hydration:
        ids = recovered_ids if recovered_ids else (
            pd.to_numeric(selected.get("row_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()
        )
        if not ids:
            raise FileNotFoundError(
                "Could not infer showcase row IDs from cached outputs. Pass them explicitly with "
                "--selected-row-ids, e.g. --selected-row-ids 556 1345 1157 511."
            )
        device = torch.device(args.device)
        model, feat_cfg, scaler, extra = load_checkpoint(args.checkpoint, map_location=device)
        model = model.to(device).eval()
        threshold = float(args.threshold if args.threshold is not None else extra.get("selected_threshold", 0.5))
        selected, record_map = _hydrate_selection_from_test(ids, args, model, feat_cfg, scaler, device, threshold)
        selected = _balance_showcase_selection(selected, args.branch_per_class)
        if selected.empty:
            raise RuntimeError("Recovered row IDs were not found in the frozen test CSV.")
        selected.to_csv(selected_path, index=False)
        print(f"Recovered/hydrated showcase selection: {selected['row_id'].astype(int).tolist()}", flush=True)
        # Always repair missing per-molecule outputs after hydration. This is only
        # 4 molecules and does not redo screening/global interpretation.
        _repair_selected_explanations(out_dir, selected, record_map, args, model, feat_cfg, scaler, device, threshold)
    else:
        selected = _balance_showcase_selection(selected, args.branch_per_class)
        selected.to_csv(selected_path, index=False)
        # Optional explicit repair when the selection exists but caches do not.
        if args.repair_selected:
            device = torch.device(args.device)
            model, feat_cfg, scaler, extra = load_checkpoint(args.checkpoint, map_location=device)
            model = model.to(device).eval()
            threshold = float(args.threshold if args.threshold is not None else extra.get("selected_threshold", 0.5))
            ids = selected["row_id"].astype(int).tolist()
            hydrated, record_map = _hydrate_selection_from_test(ids, args, model, feat_cfg, scaler, device, threshold)
            if not hydrated.empty:
                selected = _balance_showcase_selection(hydrated, args.branch_per_class)
                selected.to_csv(selected_path, index=False)
                _repair_selected_explanations(out_dir, selected, record_map, args, model, feat_cfg, scaler, device, threshold)

    # Atom figure from saved raw IG arrays.
    atom_entries: List[Dict] = []
    screening = pd.read_csv(out_dir / "screening_branch_scores.csv") if (out_dir / "screening_branch_scores.csv").exists() else pd.DataFrame()
    screen_by_id = {int(r.row_id): r for r in screening.itertuples(index=False)} if not screening.empty else {}
    for r in selected.itertuples(index=False):
        row_id = int(r.row_id)
        raw_path = out_dir / "raw" / f"row_{row_id}_raw_input_ig.npz"
        if not raw_path.exists():
            continue
        arr = np.load(raw_path, allow_pickle=True)
        smiles = str(arr["smiles"].item()) if np.asarray(arr["smiles"]).ndim == 0 else str(arr["smiles"])
        scores = np.asarray(arr["atom_scores"], dtype=float)
        img = branch_dir / f"atom_row_{row_id}.png"
        _draw_signed_atom_percentile(smiles, scores, img, "", args.dpi)
        sr = screen_by_id.get(row_id)
        atom_entries.append({
            "row_id": row_id,
            "smiles": smiles,
            "label": int(getattr(r, "label", 1 if str(r.case) == "TP" else 0)),
            "prediction": int(getattr(r, "prediction", 1 if str(r.case) == "TP" else 0)),
            "case": str(r.case),
            "p_blocker": float(r.p_blocker),
            "ig_relative_error": float(getattr(sr, "atom_ig_relative_error")) if sr is not None and hasattr(sr, "atom_ig_relative_error") else None,
            "image": str(img),
        })
    if atom_entries:
        _publication_atom_figure(atom_entries, article_dir / "Fig_Interp1_Atom_Level_4_Molecules.png", args.dpi)

    # Hierarchical motif nodes.
    motif_path = out_dir / "final_motif_node_scores.csv"
    motif_entries: List[Dict] = []
    if motif_path.exists():
        mdf = pd.read_csv(motif_path)
        for r in selected.itertuples(index=False):
            rows = mdf[mdf["row_id"].astype(int) == int(r.row_id)].to_dict("records")
            if not rows: continue
            rows.sort(key=lambda x: -abs(float(x.get("logit_contribution", 0.0))))
            img = branch_dir / f"motif_nodes_row_{int(r.row_id)}.png"
            ranked = _draw_group_structure(str(r.smiles), rows, img, "motif_nodes", args.top_pharmacophores, args.dpi)
            motif_entries.append({"row_id": int(r.row_id), "smiles": str(r.smiles), "label": int(r.label), "prediction": int(r.prediction), "case": str(r.case), "p_blocker": float(r.p_blocker), "image": str(img), "ranked": ranked})
    if motif_entries:
        _publication_motif_figure(motif_entries, article_dir / "Fig_Interp2_Hierarchical_Motif_Nodes_4_Molecules.png", args.dpi)

    # Explicit pharmacophores.
    pharma_path = out_dir / "final_pharmacophore_scores.csv"
    pharma_entries: List[Dict] = []
    if pharma_path.exists():
        pdf = pd.read_csv(pharma_path)
        for r in selected.itertuples(index=False):
            rows = pdf[pdf["row_id"].astype(int) == int(r.row_id)].to_dict("records")
            if not rows: continue
            rows = _assign_persistent_pharmacophore_ids(rows)
            rows.sort(key=lambda x: -abs(float(x.get("logit_contribution", 0.0))))
            img = branch_dir / f"pharmacophore_row_{int(r.row_id)}.png"
            ranked = _draw_group_structure(str(r.smiles), rows, img, "motif", args.top_pharmacophores, args.dpi)
            pharma_entries.append({"row_id": int(r.row_id), "smiles": str(r.smiles), "label": int(r.label), "prediction": int(r.prediction), "case": str(r.case), "p_blocker": float(r.p_blocker), "image": str(img), "ranked": ranked})
    if pharma_entries:
        _publication_pharmacophore_figure(pharma_entries, article_dir / "Fig_Interp3_Explicit_Pharmacophores_4_Molecules.png", args.dpi)

    # PIG relations.
    nodes_path = out_dir / "final_pig_nodes.csv"
    rel_path = out_dir / "final_pig_relations.csv"
    rel_entries: List[Dict] = []
    if nodes_path.exists() and rel_path.exists():
        ndf, rdf = pd.read_csv(nodes_path), pd.read_csv(rel_path)
        for r in selected.itertuples(index=False):
            motifs = ndf[ndf["row_id"].astype(int) == int(r.row_id)].to_dict("records")
            rels = rdf[rdf["row_id"].astype(int) == int(r.row_id)].to_dict("records")
            if not motifs or not rels: continue
            motifs = _assign_persistent_pharmacophore_ids(motifs)
            rels.sort(key=lambda x: -abs(float(x.get("logit_contribution", 0.0))))
            rel_entries.append({"row_id": int(r.row_id), "smiles": str(r.smiles), "label": int(r.label), "prediction": int(r.prediction), "case": str(r.case), "p_blocker": float(r.p_blocker), "motifs": motifs, "relations": rels})
    if rel_entries:
        tp_entries, tn_entries = _split_case_entries(rel_entries)
        if tp_entries:
            _publication_pig_figure(tp_entries, article_dir / "Fig_Interp4A_PIG_Relation_Blocker.png", args.dpi, args.top_relations)
        if tn_entries:
            _publication_pig_figure(tn_entries, article_dir / "Fig_Interp4B_PIG_Relation_NonBlocker.png", args.dpi, args.top_relations)

    # Global figure reconstructed from molecule-level perturbation CSVs, not the old summary tables.
    gmot = out_dir / "global_pharmacophore_occlusion.csv"
    gfam = out_dir / "global_family_occlusion.csv"
    grel = out_dir / "global_pig_relation_effects.csv"
    if gmot.exists() and gfam.exists() and grel.exists():
        fam_summary, motif_summary, rel_summary = _global_summary_tables(pd.read_csv(gmot), pd.read_csv(gfam), pd.read_csv(grel), args.bootstrap_reps, args.seed)
        fam_summary.to_csv(out_dir / "global_family_summary_publication.csv", index=False)
        motif_summary.to_csv(out_dir / "global_motif_summary_publication.csv", index=False)
        rel_summary.to_csv(out_dir / "global_relation_summary_publication.csv", index=False)
        _draw_global_q1(fam_summary, motif_summary, rel_summary, article_dir / "Fig_Interp5_Global_Interpretability.png", args.dpi)
    else:
        # Legacy run fallback: use already aggregated global tables if the raw
        # molecule-level perturbation tables were not retained.
        sf = out_dir / "global_family_summary.csv"
        sm = out_dir / "global_motif_summary.csv"
        sr = out_dir / "global_relation_summary.csv"
        if sf.exists() and sm.exists() and sr.exists():
            _draw_global_q1(pd.read_csv(sf), pd.read_csv(sm), pd.read_csv(sr), article_dir / "Fig_Interp5_Global_Interpretability.png", args.dpi)
        else:
            print("WARNING: global cached tables were not found; Fig_Interp5A/5B/5C were not regenerated. The showcase figures are unaffected.", flush=True)

    # BRICS and faithfulness remain supplementary.
    frag_path = out_dir / "final_fragment_scores.csv"
    if frag_path.exists():
        fdf = pd.read_csv(frag_path)
        frag_entries = []
        for r in selected.itertuples(index=False):
            rows = fdf[fdf["row_id"].astype(int) == int(r.row_id)].to_dict("records")
            if not rows: continue
            rows.sort(key=lambda x: -abs(float(x.get("logit_contribution", 0.0))))
            img = branch_dir / f"fragment_row_{int(r.row_id)}.png"
            ranked = _draw_group_structure(str(r.smiles), rows, img, "fragment", args.top_fragments, args.dpi)
            frag_entries.append({"row_id": int(r.row_id), "smiles": str(r.smiles), "label": int(r.label), "prediction": int(r.prediction), "case": str(r.case), "p_blocker": float(r.p_blocker), "image": str(img), "ranked": ranked})
        if frag_entries:
            _branch_grid_group(frag_entries, supplementary_dir / "Fig_S1_BRICS_Fragments_4_Molecules.png", args.dpi, "fragment")
    faith = out_dir / "faithfulness_summary.csv"
    if faith.exists():
        _draw_faithfulness(pd.read_csv(faith), supplementary_dir / "Fig_S2_Attention_Ranking_Faithfulness.png", args.dpi)

    primary_faith = out_dir / "primary_faithfulness_summary.csv"
    if primary_faith.exists():
        _draw_primary_explanation_faithfulness(
            pd.read_csv(primary_faith),
            article_dir / "Fig_Interp6_Explanation_Faithfulness.png",
            args.dpi,
        )

    fusion_detail = out_dir / "fusion_gate_molecule_level.csv"
    fusion_summary = out_dir / "fusion_gate_summary.csv"
    if fusion_detail.exists():
        fd = pd.read_csv(fusion_detail)
        fs = pd.read_csv(fusion_summary) if fusion_summary.exists() else pd.DataFrame()
        # Old cached CSVs do not contain effective contribution columns.
        # In that case, a full rerun is required; do not fabricate rho values.
        if {"rho_graph", "rho_evidence"}.issubset(fd.columns):
            _draw_fusion_gate_diagnostic(
                fd, fs,
                supplementary_dir / "Fig_S3_Adaptive_Fusion_Gates.png",
                args.dpi,
            )
        else:
            print(
                "WARNING: cached fusion_gate_molecule_level.csv predates effective "
                "contribution diagnostics; rerun without --render-existing to regenerate Fig. S3.",
                flush=True,
            )

    print("Publication figures regenerated from existing interpretation outputs.")
    for fig_path in sorted(article_dir.glob("Fig_Interp*.png")):
        print(f"  {fig_path}")


def _render_global_only(out_dir: Path, args: argparse.Namespace) -> None:
    """Regenerate only the three split global interpretability figures.

    Priority order:
      1) raw molecule-level global perturbation CSVs (preferred; summaries are rebuilt),
      2) publication summaries,
      3) legacy/global summary CSVs.

    This path is independent of showcase selection and therefore never needs
    --selected-row-ids or --repair-selected.
    """
    article_dir = out_dir / "article_figures"
    article_dir.mkdir(parents=True, exist_ok=True)

    gmot = out_dir / "global_pharmacophore_occlusion.csv"
    gfam = out_dir / "global_family_occlusion.csv"
    grel = out_dir / "global_pig_relation_effects.csv"

    fam_summary = motif_summary = rel_summary = None
    source = None

    if gmot.exists() and gfam.exists() and grel.exists():
        motif_raw = pd.read_csv(gmot)
        family_raw = pd.read_csv(gfam)
        relation_raw = pd.read_csv(grel)
        fam_summary, motif_summary, rel_summary = _global_summary_tables(
            motif_raw, family_raw, relation_raw, args.bootstrap_reps, args.seed
        )
        fam_summary.to_csv(out_dir / "global_family_summary_publication.csv", index=False)
        motif_summary.to_csv(out_dir / "global_motif_summary_publication.csv", index=False)
        rel_summary.to_csv(out_dir / "global_relation_summary_publication.csv", index=False)
        source = "raw molecule-level global perturbation CSVs"
    else:
        candidates = [
            (
                out_dir / "global_family_summary_publication.csv",
                out_dir / "global_motif_summary_publication.csv",
                out_dir / "global_relation_summary_publication.csv",
                "publication summary CSVs",
            ),
            (
                out_dir / "global_family_summary.csv",
                out_dir / "global_motif_summary.csv",
                out_dir / "global_relation_summary.csv",
                "legacy/global summary CSVs",
            ),
        ]
        for sf, sm, sr, desc in candidates:
            if sf.exists() and sm.exists() and sr.exists():
                fam_summary = pd.read_csv(sf)
                motif_summary = pd.read_csv(sm)
                rel_summary = pd.read_csv(sr)
                source = desc
                break

    if fam_summary is None or motif_summary is None or rel_summary is None:
        expected = [
            str(gfam), str(gmot), str(grel),
            str(out_dir / "global_family_summary.csv"),
            str(out_dir / "global_motif_summary.csv"),
            str(out_dir / "global_relation_summary.csv"),
        ]
        raise FileNotFoundError(
            "Cannot regenerate Fig. 5A/5B/5C because the cached global tables were not found. "
            "Expected either the three raw global CSVs or the three global summary CSVs in --out-dir.\n"
            + "\n".join(f"  - {x}" for x in expected)
        )

    # The renderer writes three independent figures. The passed legacy path is
    # used only to determine the output directory and to remove the old combined Fig. 5.
    _draw_global_q1(
        fam_summary,
        motif_summary,
        rel_summary,
        article_dir / "Fig_Interp5_Global_Interpretability.png",
        args.dpi,
    )

    outputs = [
        article_dir / "Fig_Interp5A_Global_Motif_Family_Effects.png",
        article_dir / "Fig_Interp5B_Global_Explicit_Motifs.png",
        article_dir / "Fig_Interp5C_Global_PIG_Relations.png",
    ]
    missing = [str(x) for x in outputs if not x.exists()]
    if missing:
        raise RuntimeError(
            "The global renderer completed but one or more expected figures were not created:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    print(f"Global figures regenerated from {source}.", flush=True)
    for fp in outputs:
        print(f"  {fp}", flush=True)
        print(f"  {fp.with_suffix('.pdf')}", flush=True)
        print(f"  {fp.with_suffix('.svg')}", flush=True)


def main() -> None:
    args = parse_args_q1()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    article_dir = out_dir / "article_figures"
    article_dir.mkdir(parents=True, exist_ok=True)

    if args.render_global_only:
        _render_global_only(out_dir, args)
        return

    if args.render_existing:
        _render_existing_publication_figures(out_dir, args)
        return

    # Full recomputation uses the latest journal layout; remove obsolete split
    # layouts from earlier design iterations before writing the new figures.
    _remove_legacy_combined_publication_figures(article_dir)

    device = torch.device(args.device)
    model, feat_cfg, scaler, extra = load_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device).eval()
    threshold = float(args.threshold if args.threshold is not None else extra.get("selected_threshold", 0.5))

    if args.finalize_existing:
        scored_path = out_dir / "screening_branch_scores.csv"
        selected_path = out_dir / "shared_hierarchical_showcase_selected.csv"
        if not scored_path.exists() or not selected_path.exists():
            raise FileNotFoundError(
                "--finalize-existing requires existing screening_branch_scores.csv and "
                "shared_hierarchical_showcase_selected.csv in --out-dir"
            )
        scored = pd.read_csv(scored_path)
        shared_selection = pd.read_csv(selected_path)
        selections = {
            "Atoms": shared_selection.copy(),
            "Fragments": shared_selection.copy(),
            "Motif nodes": shared_selection.copy(),
            "Pharmacophores": shared_selection.copy(),
            "PIG relations": shared_selection.copy(),
        }
        _write_q1_report(
            out_dir, args, selections, threshold, len(scored),
            train_overlap_excluded=bool(args.train_csv and not args.allow_train_overlap),
        )
        metadata = {
            "pipeline_version": "RHMGT-publication-interpretability-v5.1",
            "checkpoint": str(args.checkpoint),
            "official_split": args.split if args.split_json else None,
            "eval_csv": args.eval_csv,
            "train_csv": args.train_csv,
            "valid_csv": args.valid_csv,
            "threshold": threshold,
            "screen_per_class": args.screen_per_class,
            "screen_ig_steps": args.screen_ig_steps,
            "final_ig_steps": args.final_ig_steps,
            "branch_per_class": args.branch_per_class,
            "max_tanimoto": args.max_tanimoto,
            "exclude_exact_train_overlap_for_showcase": bool(args.train_csv and not args.allow_train_overlap),
            "atom_method": "signed Integrated Gradients on raw atom features; hierarchy-consistent motif raw chemistry recomputed along the path",
            "main_hierarchy": "the same fixed 2-TP/2-TN panel is reused for atom, hierarchical motif-node, and explicit pharmacophore figures; PIG displays the strongest TP and TN relation cases from that panel",
            "fragment_method": "BRICS fragment feature occlusion; supplementary decomposition; primary score is blocker-logit change",
            "motif_node_method": "Fig. 2 reports structural/BRICS hierarchical motif nodes only, ranked by direct node occlusion; explicit BN/AR/NHC/PF occurrences are reserved for Fig. 3 to prevent redundant local views",
            "pharmacophore_method": "explicit BN/AR/NHC/PF occurrence localization and direct motif-node occlusion; persistent P identifiers are tied to model motif_id and reused in the PIG figure",
            "pig_method": "main-paper PIG uses explicit BN/AR/NHC/PF motif pairs; direct edge deletion is primary, typed-relation neutralization is secondary, and attention is auxiliary",
            "global_ci_method": f"molecule-level bootstrap 95% CI; {args.bootstrap_reps} resamples; deterministic seed={args.seed}",
            "faithfulness": "supplementary learned-attention ranking vs matched-random perturbation with paired one-sided Wilcoxon + BH correction + bootstrap gap CI",
            "primary_explanation_faithfulness": "main-paper top-k deletion validation of signed atom IG, direct motif-node occlusion, and direct explicit-PIG edge deletion versus matched random controls; predicted-class logit support; paired one-sided Wilcoxon + BH correction + bootstrap 95% CI",
            "adaptive_fusion_diagnostic": "molecule-specific alpha_G/alpha_E gates plus effective projected contribution ratios rho_G/rho_E = ||alpha W h||_2 normalized across graph/evidence branches; case-level molecule-bootstrap 95% CI",
            "finalized_from_existing_outputs": True,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("Existing Q1 interpretability outputs finalized successfully.")
        print(f"Report: {out_dir / 'report.md'}")
        print(f"Metadata: {out_dir / 'metadata.json'}")
        print(f"Article figures: {article_dir}")
        for fp in sorted(article_dir.glob("*.png")):
            print(f"  {fp}")
        return

    if args.eval_csv:
        records = load_records_csv(args.eval_csv, args.smiles_col, args.label_col)
        inferred_train = args.train_csv
        if inferred_train is None:
            default_train = Path("data/paper_split/hERGAT_train_df.csv")
            if default_train.exists():
                inferred_train = str(default_train)
        train_records = load_records_csv(inferred_train, args.smiles_col, args.label_col) if inferred_train else []
        valid_records = load_records_csv(args.valid_csv, args.smiles_col, args.label_col) if args.valid_csv else []
        source_desc = f"fixed test CSV {args.eval_csv}"
        if valid_records:
            print(f"Validation split loaded for provenance only: {args.valid_csv}; n={len(valid_records)}", flush=True)
    elif args.csv and args.split_json:
        all_records = load_records_csv(args.csv, args.smiles_col, args.label_col)
        train_records, val_records, test_records = hergat_exact_split(all_records, args.split_json, validate_smiles=True)
        split_map = {"train": list(train_records), "val": list(val_records), "test": list(test_records)}
        records = split_map[args.split]
        source_desc = f"reconstructed {args.split} split from {args.split_json}"
    else:
        raise ValueError("Use --eval-csv (recommended for the frozen checkpoint) or provide both --csv and --split-json")
    record_map = _records_lookup(records)
    train_smiles = {r.smiles for r in train_records}

    print(f"Evaluation source: {source_desc}; n={len(records)}", flush=True)
    pred_df = _predict_records(model, records, feat_cfg, scaler, device, threshold, args.batch_size, args.num_workers, args.cache_dir)
    pred_df["train_overlap"] = pred_df["smiles"].isin(train_smiles)
    pred_df["correct"] = pred_df["label"] == pred_df["prediction"]
    pred_df["eligible_base"] = pred_df["correct"] & ((~pred_df["train_overlap"]) | bool(args.allow_train_overlap))
    pred_df.to_csv(out_dir / "all_predictions_with_selection_flags.csv", index=False)

    # Automatic resume: the user keeps ONE canonical command.  If a previous
    # run already completed the expensive 240-molecule screening, reuse it
    # instead of screening the same frozen test molecules again.
    scored_path = out_dir / "screening_branch_scores.csv"
    selected_path = out_dir / "shared_hierarchical_showcase_selected.csv"
    if scored_path.exists() and selected_path.exists():
        scored = pd.read_csv(scored_path)
        shared_selection = pd.read_csv(selected_path)
        print(
            "Resume mode: reusing cached screening and showcase selection: "
            f"{shared_selection['row_id'].astype(int).tolist() if not shared_selection.empty else []}",
            flush=True,
        )
    else:
        prescreen = _prescreen(pred_df, args.screen_per_class)
        prescreen.to_csv(out_dir / "prescreen_candidates.csv", index=False)
        print(f"Numerical prescreen: {len(prescreen)} correctly classified candidates", flush=True)

        screen_rows: List[Dict] = []
        for i, rr in enumerate(prescreen.itertuples(index=False), start=1):
            rec = record_map[int(rr.row_id)]
            print(f"  screening {i}/{len(prescreen)} row={rec.row_id} {rr.case}", flush=True)
            try:
                screen_rows.append(_screen_one(model, feat_cfg, scaler, device, rec, threshold, args.screen_ig_steps))
            except Exception as exc:
                screen_rows.append({"row_id": rec.row_id, "smiles": rec.smiles, "label": rec.label, "prediction": int(rr.prediction), "case": rr.case, "screen_error": str(exc)})
        screen_df = pd.DataFrame(screen_rows)
        good = screen_df[screen_df.get("screen_error", pd.Series(index=screen_df.index, dtype=object)).isna()].copy() if "screen_error" in screen_df.columns else screen_df.copy()
        scored = _score_screening(good, args.min_atoms, args.max_atoms)
        scored.to_csv(scored_path, index=False)

        shared_selection = _select_shared_hierarchy_panel(scored, args.branch_per_class, args.max_tanimoto)
        shared_selection.to_csv(selected_path, index=False)
        print(
            "Shared main-paper hierarchy panel: "
            f"{shared_selection['row_id'].tolist() if not shared_selection.empty else []}",
            flush=True,
        )

    # Main figures intentionally reuse the SAME four molecules for atom, motif, and
    # explicit pharmacophore views. The PIG figure then shows the strongest TP and
    # TN relation cases from that fixed panel; BRICS-only and faithfulness remain supplementary.
    selections = {
        "Atoms": shared_selection.copy(),
        "Fragments": shared_selection.copy(),
        "Motif nodes": shared_selection.copy(),
        "Pharmacophores": shared_selection.copy(),
        "PIG relations": shared_selection.copy(),
    }

    # ------------------------------------------------------------------
    # Final branch-specific explanations and publication figures
    # ------------------------------------------------------------------
    atom_entries, atom_rows_all = [], []
    for rr in shared_selection.itertuples(index=False):
        entry, rows = _final_atom_entry(model, feat_cfg, scaler, device, record_map[int(rr.row_id)], threshold, args.final_ig_steps, out_dir, args.dpi)
        atom_entries.append(entry); atom_rows_all.extend(rows)
    pd.DataFrame(atom_rows_all).to_csv(out_dir / "final_atom_scores.csv", index=False)
    if atom_entries:
        _publication_atom_figure(atom_entries, article_dir / "Fig_Interp1_Atom_Level_4_Molecules.png", args.dpi)

    motif_node_entries, motif_node_rows_all = [], []
    for rr in shared_selection.itertuples(index=False):
        entry, rows = _final_motif_node_entry(model, feat_cfg, scaler, device, record_map[int(rr.row_id)], threshold, out_dir, args.dpi, args.top_pharmacophores)
        motif_node_entries.append(entry); motif_node_rows_all.extend(rows)
    pd.DataFrame(motif_node_rows_all).to_csv(out_dir / "final_motif_node_scores.csv", index=False)
    if motif_node_entries:
        _publication_motif_figure(motif_node_entries, article_dir / "Fig_Interp2_Hierarchical_Motif_Nodes_4_Molecules.png", args.dpi)

    rel_entries, rel_motif_rows, relation_rows_all = [], [], []
    for rr in shared_selection.itertuples(index=False):
        entry, mrows, rrows = _final_relation_entry(model, feat_cfg, scaler, device, record_map[int(rr.row_id)], threshold)
        rel_entries.append(entry); rel_motif_rows.extend(mrows); relation_rows_all.extend(rrows)
    pd.DataFrame(rel_motif_rows).to_csv(out_dir / "final_pig_nodes.csv", index=False)
    pd.DataFrame(relation_rows_all).to_csv(out_dir / "final_pig_relations.csv", index=False)
    if rel_entries:
        tp_entries, tn_entries = _split_case_entries(rel_entries)
        if tp_entries:
            _publication_pig_figure(tp_entries, article_dir / "Fig_Interp4A_PIG_Relation_Blocker.png", args.dpi, args.top_relations)
        if tn_entries:
            _publication_pig_figure(tn_entries, article_dir / "Fig_Interp4B_PIG_Relation_NonBlocker.png", args.dpi, args.top_relations)

    # Supplementary chemical decompositions on the same molecules.
    supplementary_dir = out_dir / "supplementary_figures"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    fragment_entries, fragment_rows_all = [], []
    for rr in shared_selection.itertuples(index=False):
        entry, rows = _final_fragment_entry(model, feat_cfg, scaler, device, record_map[int(rr.row_id)], threshold, out_dir, args.dpi, args.top_fragments)
        fragment_entries.append(entry); fragment_rows_all.extend(rows)
    pd.DataFrame(fragment_rows_all).to_csv(out_dir / "final_fragment_scores.csv", index=False)
    if fragment_entries:
        _branch_grid_group(fragment_entries, supplementary_dir / "Fig_S1_BRICS_Fragments_4_Molecules.png", args.dpi, "fragment")

    pharma_entries, pharma_rows_all, family_rows_all = [], [], []
    for rr in shared_selection.itertuples(index=False):
        entry, rows, fam = _final_motif_entry(model, feat_cfg, scaler, device, record_map[int(rr.row_id)], threshold, out_dir, args.dpi, args.top_pharmacophores)
        pharma_entries.append(entry); pharma_rows_all.extend(rows); family_rows_all.extend(fam)
    pd.DataFrame(pharma_rows_all).to_csv(out_dir / "final_pharmacophore_scores.csv", index=False)
    pd.DataFrame(family_rows_all).to_csv(out_dir / "final_family_scores.csv", index=False)
    if pharma_entries:
        _publication_pharmacophore_figure(pharma_entries, article_dir / "Fig_Interp3_Explicit_Pharmacophores_4_Molecules.png", args.dpi)

    # ------------------------------------------------------------------
    # Global pass on the official split
    # ------------------------------------------------------------------
    global_records = records
    if args.global_max_molecules > 0:
        global_records = global_records[: args.global_max_molecules]
    if not args.skip_global:
        gfrag_path = out_dir / "global_fragment_occlusion.csv"
        gmot_path = out_dir / "global_pharmacophore_occlusion.csv"
        gfam_path = out_dir / "global_family_occlusion.csv"
        grel_path = out_dir / "global_pig_relation_effects.csv"
        if all(p.exists() for p in (gfrag_path, gmot_path, gfam_path, grel_path)):
            print("Resume mode: reusing cached global perturbation tables.", flush=True)
            gfrag = pd.read_csv(gfrag_path)
            gmot = pd.read_csv(gmot_path)
            gfam = pd.read_csv(gfam_path)
            grel = pd.read_csv(grel_path)
        else:
            print(f"Global quantitative pass over {len(global_records)} molecules", flush=True)
            gfrag, gmot, gfam, grel = _global_pass(
                model, feat_cfg, scaler, device, global_records, threshold,
                args.global_relation_candidates, out_dir
            )
            gfrag.to_csv(gfrag_path, index=False)
            gmot.to_csv(gmot_path, index=False)
            gfam.to_csv(gfam_path, index=False)
            grel.to_csv(grel_path, index=False)

        fam_summary, motif_summary, rel_summary = _global_summary_tables(gmot, gfam, grel, args.bootstrap_reps, args.seed)
        fam_summary.to_csv(out_dir / "global_family_summary.csv", index=False)
        motif_summary.to_csv(out_dir / "global_motif_summary.csv", index=False)
        rel_summary.to_csv(out_dir / "global_relation_summary.csv", index=False)
        _draw_global_q1(fam_summary, motif_summary, rel_summary, article_dir / "Fig_Interp5_Global_Interpretability.png", args.dpi)

        # Reuse the costly global atom-IG aggregate as well when it already exists.
        gatom_path = out_dir / "global_atom_ig_sample.csv"
        if gatom_path.exists():
            print("Resume mode: reusing cached global atom-IG sample.", flush=True)
            gatom = pd.read_csv(gatom_path)
        else:
            gatom = _global_atom_sample(
                model, feat_cfg, scaler, device, pred_df, record_map, threshold,
                args.global_atom_per_class, args.global_atom_ig_steps
            )
            gatom.to_csv(gatom_path, index=False)
        if not gatom.empty:
            gatom["abs_ig"] = gatom["ig_score_blocker_logit"].abs()
            gatom.groupby(["case", "symbol", "is_aromatic"], as_index=False).agg(
                n=("atom_idx", "size"), mean_ig=("ig_score_blocker_logit", "mean"), mean_abs_ig=("abs_ig", "mean")
            ).to_csv(out_dir / "global_atom_type_summary.csv", index=False)

    # ------------------------------------------------------------------
    # Supplementary diagnostic: faithfulness of learned-attention rankings
    # ------------------------------------------------------------------
    if not args.skip_faithfulness:
        faith_pool_df = pred_df[(pred_df["eligible_base"]) & (pred_df["case"].isin(["TP", "TN"]))]
        faith_records = [record_map[int(x)] for x in faith_pool_df["row_id"].tolist()]
        detail, summary = _faithfulness_analysis(
            model, feat_cfg, scaler, device, faith_records, threshold,
            args.faithfulness_molecules, args.faithfulness_controls, args.bootstrap_reps, args.seed,
        )
        detail.to_csv(out_dir / "faithfulness_molecule_level.csv", index=False)
        summary.to_csv(out_dir / "faithfulness_summary.csv", index=False)
        _draw_faithfulness(summary, supplementary_dir / "Fig_S2_Attention_Ranking_Faithfulness.png", args.dpi)

    # ------------------------------------------------------------------
    # Main-paper quantitative validation of the PRIMARY explanations.
    # ------------------------------------------------------------------
    if not args.skip_primary_faithfulness:
        print("Primary explanation faithfulness analysis", flush=True)
        primary_detail, primary_summary = _primary_explanation_faithfulness(
            model=model,
            feat_cfg=feat_cfg,
            scaler=scaler,
            device=device,
            records=records,
            pred_df=pred_df,
            threshold=threshold,
            n_molecules=args.primary_faithfulness_molecules,
            controls=args.primary_faithfulness_controls,
            ig_steps=args.primary_faithfulness_ig_steps,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
        )
        primary_detail.to_csv(out_dir / "primary_faithfulness_molecule_level.csv", index=False)
        primary_summary.to_csv(out_dir / "primary_faithfulness_summary.csv", index=False)
        _draw_primary_explanation_faithfulness(
            primary_summary,
            article_dir / "Fig_Interp6_Explanation_Faithfulness.png",
            args.dpi,
        )

    # ------------------------------------------------------------------
    # Supplementary adaptive graph/evidence fusion diagnostic.
    # ------------------------------------------------------------------
    if not args.skip_fusion_gates:
        print("Adaptive fusion-gate diagnostic", flush=True)
        fusion_detail, fusion_summary = _fusion_gate_analysis(
            model=model,
            feat_cfg=feat_cfg,
            scaler=scaler,
            device=device,
            records=records,
            threshold=threshold,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            cache_dir=args.cache_dir,
            bootstrap_reps=args.bootstrap_reps,
            seed=args.seed,
        )
        fusion_detail.to_csv(out_dir / "fusion_gate_molecule_level.csv", index=False)
        fusion_summary.to_csv(out_dir / "fusion_gate_summary.csv", index=False)
        if not fusion_detail.empty:
            _draw_fusion_gate_diagnostic(
                fusion_detail, fusion_summary,
                supplementary_dir / "Fig_S3_Adaptive_Fusion_Gates.png",
                args.dpi,
            )
        else:
            print(
                "WARNING: fusion_weights were unavailable or not two-dimensional; "
                "Fig_S3 was not generated. This is expected for non-adaptive fusion checkpoints.",
                flush=True,
            )

    _write_q1_report(
        out_dir, args, selections, threshold, len(scored),
        train_overlap_excluded=bool(train_records and not args.allow_train_overlap),
    )

    metadata = {
        "pipeline_version": "RHMGT-publication-interpretability-v5.1",
        "checkpoint": str(args.checkpoint),
        "official_split": args.split if args.split_json else None,
        "eval_csv": args.eval_csv,
        "train_csv": args.train_csv,
        "valid_csv": args.valid_csv,
        "threshold": threshold,
        "screen_per_class": args.screen_per_class,
        "screen_ig_steps": args.screen_ig_steps,
        "final_ig_steps": args.final_ig_steps,
        "branch_per_class": args.branch_per_class,
        "max_tanimoto": args.max_tanimoto,
        "exclude_exact_train_overlap_for_showcase": bool(train_records and not args.allow_train_overlap),
        "atom_method": "signed Integrated Gradients on raw atom features; hierarchy-consistent motif raw chemistry recomputed along the path",
        "main_hierarchy": "the same fixed 2-TP/2-TN panel is reused for atom, hierarchical motif-node, and explicit pharmacophore figures; PIG displays the strongest TP and TN relation cases from that panel",
        "fragment_method": "BRICS fragment feature occlusion; supplementary decomposition; primary score is blocker-logit change",
        "motif_node_method": "Fig. 2 reports structural/BRICS hierarchical motif nodes only, ranked by direct node occlusion; explicit BN/AR/NHC/PF occurrences are reserved for Fig. 3 to prevent redundant local views",
        "pharmacophore_method": "explicit BN/AR/NHC/PF occurrence localization and direct motif-node occlusion; persistent P identifiers are tied to model motif_id and reused in the PIG figure",
        "pig_method": "main-paper PIG uses explicit BN/AR/NHC/PF motif pairs; direct edge deletion is primary, typed-relation neutralization is secondary, and attention is auxiliary",
        "global_ci_method": f"molecule-level bootstrap 95% CI; {args.bootstrap_reps} resamples; deterministic seed={args.seed}",
            "faithfulness": "supplementary learned-attention ranking vs matched-random perturbation with paired one-sided Wilcoxon + BH correction + bootstrap gap CI",
            "primary_explanation_faithfulness": "main-paper top-k deletion validation of signed atom IG, direct motif-node occlusion, and direct explicit-PIG edge deletion versus matched random controls; predicted-class logit support; paired one-sided Wilcoxon + BH correction + bootstrap 95% CI",
            "adaptive_fusion_diagnostic": "molecule-specific alpha_G/alpha_E gates plus effective projected contribution ratios rho_G/rho_E = ||alpha W h||_2 normalized across graph/evidence branches; case-level molecule-bootstrap 95% CI",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nQ1 interpretability pipeline completed.")
    print(f"Article figures: {article_dir}")
    for p in sorted(article_dir.glob("*.png")):
        print(f"  {p}")


if __name__ == "__main__":
    main()
