# Native YOLOv8 v8.4.0 Detection Contract

当前生产主线是检测优先的原生 YOLOv8 v8.4.0 行为实现。运行时只接受 640x640 RGB 输入，检测输出始终来自 Detection Head，不由 mask 或文本模型派生。

```text
image -> letterbox(640, fill=114)
      -> YOLOv8n Backbone (P1/2..P5/32) + SPPF
      -> concat PAN/FPN (P3/P4/P5)
      -> Detect (4*16 DFL + C class logits)
      -> TAL + CIoU + DFL + BCE
      -> canonical 640 xyxy + class-local NMS
```

`yolov8n-p2` 是后续显式架构版本：在通过 P3/P4/P5 质量门禁后增加 P2/4 分支，不引入注意力或其他同时变化的模块。两种架构的 checkpoint、state dict 和转换边界严格区分。

## 数学契约

- `reg_max=16`，raw box 为 `[B, 64, A]`，统一转换为 `[B,A,4,16]`。
- anchor center 使用 `0.5` offset；DFL projection 为 `[0..15]`。
- TAL 使用 `topk=10, alpha=0.5, beta=6.0`、v8.4.0 的 CIoU overlap、小框候选扩展和重叠 GT 最高 IoU 解冲突。
- 分类、CIoU、DFL 都除以 `max(target_scores.sum(), 1)`。
- AP 固定 `conf=0.001,max_det=300`；产品阈值单独从 validation PR 曲线校准。

## 权重与数据边界

`ultralytics==8.4.0` 只用于官方参考、权重转换和 parity 测试，不成为生产推理依赖。官方 COCO 权重转换必须使用仓库生成的完整 mapping JSON；COCO 的 80 类分类 tower 与六类目标 hidden width 不同，因此整个目标分类 tower 会被显式记录为 reinitialize 边界，不能隐式部分加载。

NEU-DET 图片/XML、官方权重和 `vision_base.pt` 都在仓库外提供。缺文件、hash、类别、架构或数据指纹不匹配时直接失败。

## Segmentation 状态

当前仓库没有真实像素 mask manifest，Semantic Segmentation、joint fine-tune 和神经视觉 token 训练入口保持 Blocked；不得将检测框、全背景 mask 或结构化文本结果冒充分割模型。
