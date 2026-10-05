from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

from .config import FeaturizerConfig, ModelConfig


@dataclass(frozen=True)
class AblationSpec:
    """One controlled HERA-HGT ablation.

    The registry is intentionally explicit: every ablation changes one conceptual
    component while leaving the optimization recipe untouched.
    """

    name: str
    description: str


ABLATION_SPECS: Dict[str, AblationSpec] = {
    "full": AblationSpec(
        "full",
        "Complete HERA-HGT with hierarchical graph, all evidence modalities, all pairwise biases, and the selected fusion.",
    ),
    "graph_only": AblationSpec(
        "graph_only",
        "Use only the hierarchical graph representation h_G; remove the complete evidence branch and graph-evidence fusion from prediction.",
    ),
    "evidence_only": AblationSpec(
        "evidence_only",
        "Use only Morgan, MACCS, AtomPair and descriptor evidence; remove the hierarchical graph representation from prediction.",
    ),
    "no_brics": AblationSpec(
        "no_brics",
        "Remove BRICS structural-fragment motifs while preserving hERG pharmacophore motifs and hierarchical connectivity.",
    ),
    "no_herg_motifs": AblationSpec(
        "no_herg_motifs",
        "Remove the explicit hERG pharmacophore motifs (BN, AR, NHC, PF) while preserving BRICS structural motifs.",
    ),
    "no_relation_bias": AblationSpec(
        "no_relation_bias",
        "Disable learned relation-type attention bias while preserving exactly the same graph edges/pair mask.",
    ),
    "no_distance_bias": AblationSpec(
        "no_distance_bias",
        "Disable learned topological-distance attention bias while preserving exactly the same graph edges/pair mask.",
    ),
    "no_bond_bias": AblationSpec(
        "no_bond_bias",
        "Disable the 10D bond-chemistry attention bias while preserving exactly the same graph edges/pair mask.",
    ),
    "no_morgan": AblationSpec(
        "no_morgan",
        "Remove molecular information carried by the Morgan evidence token.",
    ),
    "no_maccs": AblationSpec(
        "no_maccs",
        "Remove molecular information carried by the MACCS evidence token.",
    ),
    "no_atompair": AblationSpec(
        "no_atompair",
        "Remove molecular information carried by the AtomPair evidence token.",
    ),
    "no_descriptors": AblationSpec(
        "no_descriptors",
        "Remove molecular information carried by the physicochemical descriptor evidence token.",
    ),
}

ABLATION_CHOICES = tuple(ABLATION_SPECS.keys())


def apply_ablation(
    name: str,
    feat_cfg: FeaturizerConfig,
    model_cfg: ModelConfig,
) -> Tuple[FeaturizerConfig, ModelConfig]:
    """Return copied configs with exactly the requested ablation applied."""

    if name not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation={name!r}; allowed={list(ABLATION_CHOICES)}")

    # Always store the experiment identity in the checkpoint configuration.
    model_cfg = replace(model_cfg, ablation_name=name)

    if name == "full":
        return feat_cfg, model_cfg

    if name == "graph_only":
        return feat_cfg, replace(model_cfg, prediction_mode="graph_only")

    if name == "evidence_only":
        return feat_cfg, replace(model_cfg, prediction_mode="evidence_only")

    if name == "no_brics":
        return replace(feat_cfg, include_brics=False), model_cfg

    if name == "no_herg_motifs":
        return replace(feat_cfg, include_pharmacophores=False), model_cfg

    if name == "no_relation_bias":
        return feat_cfg, replace(model_cfg, use_relation_bias=False)

    if name == "no_distance_bias":
        return feat_cfg, replace(model_cfg, use_distance_bias=False)

    if name == "no_bond_bias":
        return feat_cfg, replace(model_cfg, use_bond_bias=False)

    if name == "no_morgan":
        return feat_cfg, replace(model_cfg, use_morgan=False)

    if name == "no_maccs":
        return feat_cfg, replace(model_cfg, use_maccs=False)

    if name == "no_atompair":
        return feat_cfg, replace(model_cfg, use_atompair=False)

    if name == "no_descriptors":
        return feat_cfg, replace(model_cfg, use_descriptors=False)

    raise AssertionError(f"Unhandled ablation={name!r}")


def ablation_description(name: str) -> str:
    if name not in ABLATION_SPECS:
        raise ValueError(f"Unknown ablation={name!r}")
    return ABLATION_SPECS[name].description
