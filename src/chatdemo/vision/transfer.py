from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from .model import DefectVisionModel


def load_explicit_state_mapping(
    model: DefectVisionModel,
    source_checkpoint: str | Path,
    mapping_file: str | Path,
) -> DefectVisionModel:
    """Load a fully specified external-weight mapping with shape validation.

    No parameter-name or tensor-shape guessing is allowed.  The mapping must
    cover every target parameter and every source tensor used by it, making
    license review and conversion ownership an explicit boundary.
    """
    source_path = Path(source_checkpoint)
    mapping_path = Path(mapping_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"transfer checkpoint not found: {source_path}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"transfer mapping not found: {mapping_path}")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        source_state = payload["state_dict"]
    elif isinstance(payload, dict):
        source_state = payload
    else:
        raise ValueError("transfer checkpoint must contain a state-dict mapping")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("transfer mapping must be a JSON object of target_name -> source_name")
    target_state = model.state_dict()
    if set(mapping) != set(target_state):
        missing = sorted(set(target_state).difference(mapping))
        extra = sorted(set(mapping).difference(target_state))
        raise ValueError(f"transfer mapping must cover all target tensors; missing={missing[:3]}, extra={extra[:3]}")
    converted = {}
    for target_name, source_name in mapping.items():
        if not isinstance(source_name, str) or source_name not in source_state:
            raise ValueError(f"transfer mapping references missing source tensor: {source_name!r}")
        source_tensor = source_state[source_name]
        target_tensor = target_state[target_name]
        if not isinstance(source_tensor, torch.Tensor) or tuple(source_tensor.shape) != tuple(target_tensor.shape):
            raise ValueError(f"shape mismatch for {target_name} <- {source_name}")
        converted[target_name] = source_tensor.detach().to(dtype=target_tensor.dtype).clone()
    model.load_state_dict(converted, strict=True)
    return model
