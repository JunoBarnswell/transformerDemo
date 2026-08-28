# Issue #1 视觉架构与产品契约

## 产品目标

在保留现有纯文本 Tiny Transformer、ConversationMemory 和 Thinking Mode 的前提下，增加一个真实可训练、可评测、可推理的工业缺陷视觉链路。视觉侧必须同时提供两条一级能力：

1. Detection：无需 prompt，独立 Detection Head 直接输出类别、置信度和矩形框。
2. Semantic Segmentation：独立 Decoder 输出像素级类别 mask，不能用检测框替代。

Issue #1 的架构图落地为：

```text
image -> letterbox 640 -> shared Backbone/Neck
                              |              |
                              v              v
                       Detection Head   Segmentation Decoder
                              |              |
                              +-------> explicit visual context
```

昂贵的 Backbone/Neck 只共享一次，两个任务头没有运行时硬依赖。检测框永远来自 Detection Head；mask 派生的连通域或 bbox 只允许作为离线诊断数据，不能进入检测主输出。

`VisionTokenAdapter` 默认对共享视觉特征执行 `detach`，避免尚未有视觉-文本联合 checkpoint 时由语言损失反向破坏视觉主线。当前已有文本模型没有视觉 embedding 输入契约，`chat_cli` 因此采用可检查的结构化视觉文本上下文；神经视觉 token 接入必须在单独的联合训练 checkpoint 上显式启用。

## 输入与标注契约

视觉数据使用 JSONL，每行至少包含 `image`。检测和分割标注可以独立存在：

```json
{"image":"ok.jpg","boxes":[],"labels":[]}
{"image":"box.jpg","boxes":[[12,20,80,90]],"labels":[0]}
{"image":"mask.jpg","mask":"mask.png"}
{"image":"both.jpg","boxes":[{"box":[12,20,80,90],"label":0}],"mask":"both_mask.png"}
```

`boxes` 字段存在但为空表示检测任务的明确负样本；字段不存在表示该样本没有检测监督。`mask` 缺失表示没有分割监督，不能被解释成全背景。路径相对 manifest 文件解析，并在加载时检查存在性。框必须是原图像素坐标的正面积 `xyxy`；mask 尺寸必须和原图严格一致。

## 输出契约

`chatdemo.analyze_image` 返回：

```json
{
  "image_size": [1920, 1080],
  "coordinate_space": [640, 640],
  "detections": [{"class_id": 0, "class_name": "defect", "score": 0.9, "box_xyxy": [1, 2, 3, 4]}],
  "segmentation": {"mask_path": "...", "mask_size": [1920, 1080], "classes": ["defect"]},
  "annotated_image": "..."
}
```

模型内部可以使用 Letterbox 坐标，但离开推理边界前必须先还原到原图，再映射到 canonical 640。反向绘制只使用 `orig_w / 640` 和 `orig_h / 640`，因此不受 padding 影响。

## 初始化与 fail-close

`vision_init.mode=project_base` 要求存在带有完全匹配 `vision_config` 和 `model_state` 的真实 checkpoint。`random` 只能显式用于开发/单元冒烟，不能成为生产模式缺权重时的隐式 fallback。`yolov8_transfer` 是需要单独完成许可证审查和权重转换的外部边界；它必须同时提供覆盖全部目标张量、逐项形状匹配的 mapping JSON，未提供时直接拒绝，不猜测参数映射。

## 评测边界

仓库当前没有工业视觉 manifest、`vision_base.pt` 或 YOLOv8 baseline 权重。代码提供 same-shot benchmark harness，但只有在用户提供相同 manifest、划分、shot 抽样和 YOLOv8 权重后，才能报告 mAP50、mAP50-95、AP75、small-defect recall、延迟和资源指标。本仓库不生成或伪造这些结果。

`mine_hard_negative_indices` 只返回模型在明确正常图上的误报样本索引，后续区域标注必须由数据负责人审核；它不伪造负样本框，也不静默修改 manifest。
