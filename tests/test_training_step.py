import numpy as np
import torch

from hera_hgt.config import FeaturizerConfig, ModelConfig
from hera_hgt.data import DescriptorScaler, collate_hera_hgt
from hera_hgt.hierarchy import RELATION_VOCAB_SIZE, featurize_hierarchical_smiles
from hera_hgt.model import HERAHGT


def test_one_training_step_is_finite():
    items, raws = [], []
    for i, smi in enumerate(["CCN(CC)CC", "O=C(O)c1ccccc1"]):
        x = featurize_hierarchical_smiles(smi, FeaturizerConfig(max_motifs=16))
        raws.append(x["desc_raw"])
        x["label"] = float(i)
        x["row_id"] = i
        items.append(x)
    a = np.stack(raws).astype(np.float32)
    scale = a.std(0); scale[scale < 1e-8] = 1.0
    sc = DescriptorScaler(a.mean(0), scale)
    for x in items:
        x["desc"] = sc.transform(x["desc_raw"])
    batch = collate_hera_hgt(items)
    cfg = ModelConfig(hidden_dim=64, n_heads=4, n_layers=1, fp_encoder_hidden=64,
                      desc_encoder_hidden=32, classifier_hidden=64,
                      relation_vocab=RELATION_VOCAB_SIZE)
    model = HERAHGT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out = model(batch)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], batch["label"])
    assert torch.isfinite(loss)
    loss.backward(); opt.step()
