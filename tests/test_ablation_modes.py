import numpy as np
import torch

from hera_hgt.ablations import apply_ablation
from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import DescriptorScaler, collate_hera_hgt
from hera_hgt.hierarchy import RELATION_VOCAB_SIZE, featurize_hierarchical_smiles
from hera_hgt.model import HERAHGT


def _batch(feat_cfg=None):
    feat_cfg = feat_cfg or FeaturizerConfig(max_motifs=24)
    items = []
    raws = []
    for i, smi in enumerate(["CN(C)CCc1ccccc1", "CCN1CCCCC1"]):
        x = featurize_hierarchical_smiles(smi, feat_cfg)
        raws.append(x["desc_raw"])
        x["label"] = float(i)
        x["row_id"] = i
        items.append(x)
    a = np.stack(raws).astype(np.float32)
    scale = a.std(0)
    scale[scale < 1e-8] = 1.0
    sc = DescriptorScaler(a.mean(0), scale)
    for x in items:
        x["desc"] = sc.transform(x["desc_raw"])
    return collate_hera_hgt(items), items


def _cfg(**kwargs):
    base = dict(
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        fp_encoder_hidden=64,
        desc_encoder_hidden=32,
        classifier_hidden=64,
        relation_vocab=RELATION_VOCAB_SIZE,
        fusion_type="fhgnn_adaptive",
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_graph_only_forward():
    batch, _ = _batch()
    cfg = _cfg(prediction_mode="graph_only", ablation_name="graph_only")
    out = HERAHGT(cfg).eval()(batch)
    assert out["logits"].shape == (2,)
    assert out["global_repr"].shape == (2, 64)
    assert out["evidence_tokens"].shape == (2, 0, 64)


def test_evidence_only_forward_uses_adaptive_evidence_summary():
    batch, _ = _batch()
    cfg = _cfg(prediction_mode="evidence_only", ablation_name="evidence_only")
    out = HERAHGT(cfg).eval()(batch)
    assert out["logits"].shape == (2,)
    assert out["global_repr"].shape == (2, 0)
    assert out["evidence_tokens"].shape == (2, 4, 64)
    assert out["fused_repr"].shape == (2, 64)


def test_removed_evidence_token_is_exactly_zero():
    batch, _ = _batch()
    cfg = _cfg(use_morgan=False, ablation_name="no_morgan")
    model = HERAHGT(cfg).eval()
    with torch.no_grad():
        evidence = model.evidence_encoder(batch)
    assert torch.count_nonzero(evidence[:, 0]).item() == 0
    assert torch.count_nonzero(evidence[:, 1:]).item() > 0


def test_no_brics_featurizer_does_not_create_brics_named_nodes():
    feat_cfg, model_cfg = apply_ablation("no_brics", FeaturizerConfig(), _cfg())
    assert feat_cfg.include_brics is False
    assert model_cfg.ablation_name == "no_brics"
    _, items = _batch(feat_cfg)
    for item in items:
        assert all(not str(name).startswith("brics_") for name in item["motif_names"])


def test_no_herg_motifs_keeps_only_structural_family():
    feat_cfg, _ = apply_ablation("no_herg_motifs", FeaturizerConfig(), _cfg())
    _, items = _batch(feat_cfg)
    for item in items:
        assert all(int(family) == 0 for family in item["motif_families"])


def test_bias_ablation_changes_only_requested_switch():
    feat_cfg = FeaturizerConfig()
    base = _cfg()
    _, cfg = apply_ablation("no_relation_bias", feat_cfg, base)
    assert cfg.use_relation_bias is False
    assert cfg.use_distance_bias is True
    assert cfg.use_bond_bias is True
