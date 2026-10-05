import numpy as np
import torch

from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import DescriptorScaler, collate_hera_hgt
from hera_hgt.hierarchy import RELATION_VOCAB_SIZE, featurize_hierarchical_smiles
from hera_hgt.model import HERAHGT


def _batch():
    items = []
    raws = []
    for i, smi in enumerate(["CN(C)CCc1ccccc1", "CCN1CCCCC1"]):
        x = featurize_hierarchical_smiles(smi, FeaturizerConfig(max_motifs=24))
        raws.append(x["desc_raw"])
        x["label"] = float(i)
        x["row_id"] = i
        items.append(x)
    a = np.stack(raws).astype(np.float32)
    scale = a.std(0); scale[scale < 1e-8] = 1.0
    sc = DescriptorScaler(a.mean(0), scale)
    for x in items:
        x["desc"] = sc.transform(x["desc_raw"])
    return collate_hera_hgt(items)


def test_forward_shapes_and_cross_attention():
    batch = _batch()
    cfg = ModelConfig(hidden_dim=64, n_heads=4, n_layers=2, fp_encoder_hidden=64,
                      desc_encoder_hidden=32, classifier_hidden=64,
                      relation_vocab=RELATION_VOCAB_SIZE)
    model = HERAHGT(cfg).eval()
    out = model(batch)
    assert out["logits"].shape == (2,)
    assert out["cross_attention"].shape == (2, 4, 4)
    means = out["cross_attention"].mean(dim=1)
    assert torch.allclose(means.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert out["global_repr"].shape == (2, 64)
    assert out["evidence_tokens"].shape == (2, 4, 64)
