from __future__ import annotations

import torch

from chatdemo.vision.coordinates import canonical_to_original, model_to_canonical, original_to_canonical
from chatdemo.vision.preprocess import build_image_transform, restore_mask


def test_letterbox_and_canonical_roundtrip_for_wide_image():
    transform = build_image_transform(1280, 720)
    assert transform.resized_w == 640
    assert transform.resized_h == 360
    assert transform.pad_y == 140

    original = torch.tensor([[100.0, 80.0, 640.0, 500.0]])
    model_space = original_to_canonical(original, 1280, 720)
    restored = canonical_to_original(model_space, 1280, 720)
    assert torch.allclose(restored, original, atol=1e-4)

    model_box = torch.tensor([[100.0, 180.0, 420.0, 390.0]])
    canonical = model_to_canonical(model_box, transform)
    assert torch.allclose(canonical, torch.tensor([[100.0, 71.1111, 420.0, 444.4444]]), atol=1e-3)


def test_restore_mask_removes_padding_and_restores_original_size():
    transform = build_image_transform(1280, 720)
    mask = torch.zeros((640, 640), dtype=torch.float32)
    mask[140:500, :] = 1.0
    restored = restore_mask(mask, transform)
    assert restored.shape == (720, 1280)
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0
