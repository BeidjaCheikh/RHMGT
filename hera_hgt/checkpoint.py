from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .config import FeaturizerConfig, ModelConfig
from .data import DescriptorScaler
from .model import HERAHGT


def save_checkpoint(
    path: str | Path,
    model: HERAHGT,
    feat_cfg: FeaturizerConfig,
    descriptor_scaler: DescriptorScaler,
    extra: Dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "featurizer_config": feat_cfg.to_dict(),
        "descriptor_scaler": descriptor_scaler.to_dict(),
        "extra": extra,
    }, path)


def load_checkpoint(
    path: str | Path,
    map_location="cpu",
) -> Tuple[HERAHGT, FeaturizerConfig, DescriptorScaler, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["model_config"])
    feat_cfg = FeaturizerConfig.from_dict(ckpt["featurizer_config"])
    scaler = DescriptorScaler.from_dict(ckpt["descriptor_scaler"])
    model = HERAHGT(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    return model, feat_cfg, scaler, ckpt.get("extra", {})
