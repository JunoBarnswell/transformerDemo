# YOLOv8 v8.4.0 算法对齐报告

## 已实现并验证

本仓库的 native graph 与官方 `ultralytics==8.4.0` 通过同一 YAML 图和 state-dict 键进行对照：

- `yolov8n`：355 个 state tensor 键全部一致，参数形状一致，640 输入的解码框/类别分数通过 `atol=1e-4, rtol=1e-4`。
- `yolov8n-p2`：437 个 state tensor 键全部一致，输出 anchor 数从 8400 增加到 34000。
- DFL、CIoU、TAL 的官方参考测试已纳入 `tests/test_detection_parity.py`。
- 官方 v8.4.0 Loss 实际调用 `topk=10, alpha=0.5, beta=6.0`；实现按该调用参数对齐，而不是照搬 TAL 类默认值。
- 官方评测 AP 使用 COCO 101-point interpolation；本地 evaluator 已采用同一插值口径。

参考源码：

- <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.0/ultralytics/utils/tal.py>
- <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.0/ultralytics/utils/loss.py>
- <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.0/ultralytics/nn/modules/head.py>
- <https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.0/ultralytics/cfg/models/v8/yolov8-p2.yaml>

## 明确的实现边界

官方 v8.4.0 在传入 `topk_mask` 时会保留 alignment 为零的 top-k 候选；为满足算法 parity，native 实现保留这一行为，不能另加“零 alignment 拒绝”而继续声称数值等价。COCO 80 类权重与六类目标的分类 tower hidden width 不同，转换工具会显式重初始化整个目标 `cv3` 分类 tower，Backbone/Neck/box branch 仍逐 tensor 严格映射。

## 尚未完成的外部实验

官方与 native 的完整 NEU-DET 三种子质量实验尚未完成。数据准备在发现来源仓库 `patches_101.jpg` 与 `patches_105.jpg` SHA-256 完全相同但 XML 框不同后按 fail-close 停止；需要数据负责人决定保留并标记重复样本，或剔除并重新平衡 split。当前不输出 mAP parity 或 Few-shot 质量结论。
