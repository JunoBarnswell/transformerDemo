from __future__ import annotations

import torch

from chatdemo.vision.adapters import VisionTokenAdapter
from chatdemo.vision.tile import TileWindow, merge_tile_detections, should_use_tiles, tile_windows
from chatdemo.vision.types import Detection


def test_tile_windows_cover_large_image_and_auto_threshold_is_explicit():
    windows = tile_windows(1000, 800, tile_size=640, overlap=0.2)
    assert windows
    assert min(window.x0 for window in windows) == 0
    assert max(window.x1 for window in windows) == 1000
    assert max(window.y1 for window in windows) == 800
    assert should_use_tiles(4096, 3072, estimated_defect_short_side=3.0)
    assert not should_use_tiles(4096, 3072, estimated_defect_short_side=8.0)


def test_tile_detection_merge_preserves_global_canonical_contract():
    window = TileWindow(640, 0, 1000, 640)
    local = Detection(0, 0.9, (160.0, 160.0, 320.0, 320.0), "defect")
    merged = merge_tile_detections([(window, [local])], image_width=1000, image_height=640)
    assert len(merged) == 1
    assert 0.0 <= merged[0].box_xyxy[0] < merged[0].box_xyxy[2] <= 640.0


def test_visual_token_adapter_is_bounded_and_shape_stable():
    adapter = VisionTokenAdapter(p3_channels=8, p4_channels=16, d_model=12, num_tokens=6)
    tokens = adapter({
        "P3": torch.randn(2, 8, 20, 20),
        "P4": torch.randn(2, 16, 10, 10),
    })
    assert tokens.shape == (2, 6, 12)
