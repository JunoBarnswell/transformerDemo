from __future__ import annotations

import json
from pathlib import Path

import torch

from .model import DefectVisionModel


def load_explicit_state_mapping(model: DefectVisionModel, source_checkpoint: str | Path, mapping_file: str | Path) -> DefectVisionModel:
    """Apply an explicit, shape-checked one-time weight conversion.

    The mapping file owns every exception. Only final classification tensors
    may be listed under ``reinitialize_targets``; all other target tensors and
    all referenced source tensors must be covered exactly.
    """
    source_path, mapping_path = Path(source_checkpoint), Path(mapping_file)
    if not source_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError("transfer checkpoint and mapping JSON are both required")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    source = payload.get("model_state", payload.get("state_dict")) if isinstance(payload, dict) else None
    if source is None and isinstance(payload, dict) and hasattr(payload.get("model"), "state_dict"):
        source = payload["model"].state_dict()
    if source is None and isinstance(payload, dict):
        source = payload
    if not isinstance(source, dict):
        raise ValueError("transfer checkpoint must contain model_state/state_dict")
    spec = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or not isinstance(spec.get("mappings"), dict):
        raise ValueError("mapping JSON must contain mappings and optional reinitialize_targets")
    mappings = spec["mappings"]
    reinit = set(spec.get("reinitialize_targets", []))
    target = model.state_dict()
    if set(mappings) | reinit != set(target):
        raise ValueError("mapping must cover every target tensor exactly")
    # A COCO yolov8n checkpoint has cv3's hidden width derived from nc=80,
    # while the six-class target derives it from nc=6.  Therefore the entire
    # class tower is an explicit reinitialization boundary when shapes differ.
    allowed_reinit = {name for name in target if ".cv3." in name}
    if not reinit <= allowed_reinit:
        raise ValueError("only final classification predictor tensors may be reinitialized")
    converted = dict(target)
    for target_name, source_name in mappings.items():
        if not isinstance(source_name, str) or source_name not in source:
            raise ValueError(f"missing source tensor for {target_name}: {source_name!r}")
        value = source[source_name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(target[target_name].shape):
            raise ValueError(f"shape mismatch for {target_name} <- {source_name}")
        converted[target_name] = value.to(dtype=target[target_name].dtype).clone()
    model.load_state_dict(converted, strict=True)
    return model
