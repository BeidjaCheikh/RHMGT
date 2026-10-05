from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class FeaturizerConfig:
    """Chemistry and hierarchical graph construction configuration."""

    max_atom_dist: int = 10
    max_motif_dist: int = 12
    max_motifs: int = 48
    min_brics_atoms: int = 2
    include_brics: bool = True
    include_pharmacophores: bool = True

    # Direct-comparison choice: hERGAT uses Morgan radius=3, 1024 bits.
    morgan_dim: int = 1024
    morgan_radius: int = 3
    maccs_dim: int = 167
    atom_pair_fp_dim: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeaturizerConfig":
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class ModelConfig:
    """Canonical HERA-HGT architecture configuration."""

    # hERGAT code-compatible atom and bond schemas.
    atom_dim: int = 39
    bond_dim: int = 10

    # Complementary molecular evidence.
    desc_dim: int = 10
    morgan_dim: int = 1024
    maccs_dim: int = 167
    atom_pair_fp_dim: int = 2048

    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.10
    ffn_multiplier: int = 4

    motif_family_vocab: int = 5  # structural, BN, AR, NHC, PF
    relation_vocab: int = 15
    max_pair_dist: int = 12

    fp_encoder_hidden: int = 512
    desc_encoder_hidden: int = 128
    classifier_hidden: int = 256

    # Final multimodal fusion. Keep cross_attention as the backward-compatible
    # dataclass fallback for historical checkpoints; train.py explicitly selects
    # fhgnn_adaptive for new final-model experiments.
    fusion_type: str = "cross_attention"

    # Controlled ablation switches. Defaults reproduce the complete model.
    ablation_name: str = "full"
    prediction_mode: str = "multimodal"  # multimodal | graph_only | evidence_only
    use_relation_bias: bool = True
    use_distance_bias: bool = True
    use_bond_bias: bool = True
    use_morgan: bool = True
    use_maccs: bool = True
    use_atompair: bool = True
    use_descriptors: bool = True

    def validate(self) -> None:
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.atom_dim != 39:
            raise ValueError("HERA-HGT uses the 39-dimensional hERGAT atom schema")
        if self.bond_dim != 10:
            raise ValueError("HERA-HGT uses the 10-dimensional hERGAT bond schema")
        if self.desc_dim != 10:
            raise ValueError("HERA-HGT uses 10 physicochemical descriptors")
        allowed_fusions = {"cross_attention", "concat", "fhgnn_adaptive"}
        if self.fusion_type not in allowed_fusions:
            raise ValueError(
                f"fusion_type must be one of {sorted(allowed_fusions)}; got {self.fusion_type!r}"
            )
        allowed_prediction_modes = {"multimodal", "graph_only", "evidence_only"}
        if self.prediction_mode not in allowed_prediction_modes:
            raise ValueError(
                "prediction_mode must be one of "
                f"{sorted(allowed_prediction_modes)}; got {self.prediction_mode!r}"
            )
        if self.prediction_mode == "evidence_only" and self.fusion_type != "fhgnn_adaptive":
            raise ValueError(
                "evidence_only uses the same evidence-summary block as the final "
                "fhgnn_adaptive model; set fusion_type='fhgnn_adaptive'"
            )
        if self.prediction_mode != "graph_only" and not any(
            [self.use_morgan, self.use_maccs, self.use_atompair, self.use_descriptors]
        ):
            raise ValueError("At least one evidence modality must remain enabled")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        fields = cls.__dataclass_fields__
        cfg = cls(**{k: v for k, v in data.items() if k in fields})
        cfg.validate()
        return cfg
