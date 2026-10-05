import torch

from hera_hgt.config import ModelConfig
from hera_hgt.model import ConcatFusion, CrossModalFusion, FHGNAdaptiveFusion


def test_fusion_modes_shapes():
    B, H = 3, 256
    global_h = torch.randn(B, H)
    evidence = torch.randn(B, 4, H)

    cfg = ModelConfig(hidden_dim=H, n_heads=8, fusion_type="cross_attention")
    z, attn = CrossModalFusion(cfg)(global_h, evidence)
    assert z.shape == (B, H)
    assert attn.shape == (B, 8, 4)

    cfg = ModelConfig(hidden_dim=H, n_heads=8, fusion_type="concat")
    z, no_attn, weights = ConcatFusion(cfg)(global_h, evidence)
    assert z.shape == (B, H)
    assert no_attn.shape == (B, 1, 0)
    assert weights.shape == (B, 0)

    cfg = ModelConfig(hidden_dim=H, n_heads=8, fusion_type="fhgnn_adaptive")
    z, no_attn, gates = FHGNAdaptiveFusion(cfg)(global_h, evidence)
    assert z.shape == (B, H)
    assert no_attn.shape == (B, 1, 0)
    assert gates.shape == (B, 2)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(B), atol=1e-6)
