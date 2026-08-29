from __future__ import annotations

from typing import Any, Mapping


def vision_result_to_context(result: Mapping[str, Any]) -> str:
    """Serialize detector output for the legacy text model.

    This is not a trainable visual-token adapter; neural multimodal fusion is
    blocked until joint data and a validated checkpoint exist.
    """
    detections = result.get("detections", [])
    if not detections:
        detection_text = "无检测框"
    else:
        entries = []
        for item in detections:
            name = item.get("class_name") or f"class_{item.get('class_id', 0)}"
            entries.append(f"{name} score={float(item.get('score', 0.0)):.3f} box={item.get('box_xyxy')}")
        detection_text = "; ".join(entries)
    return f"[视觉分析上下文] 图像尺寸={result.get('image_size')}; 检测={detection_text}。"
