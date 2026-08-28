# Detection 质量整改 PRD

## 目标与完成标准

本次整改解决“训练 loss 下降但检测置信度、定位和 mAP 均为 0”的系统性问题。完成标准分为两层：

1. 训练链健康性：8 张训练图应能被模型明显拟合，`mAP@0.5` 和 Recall 不再为 0；若无法拟合，说明训练链仍有结构错误。
2. 留出能力：仅在未参与训练的 `inclusion_2`、`pitted_surface_2` 上报告指标，不把训练集拟合冒充泛化能力。

最终交付同时保留 8/2 留出实验模型和使用全部 10 张图训练的本地最终模型。10 张数据、无预训练权重的实验不用于声明 YOLOv8 等级精度。

## 本地数据证据

NEU-DET 10 图子集包含 24 个框；8 张训练图包含 19 个框。训练框面积占原图比例范围为 `0.0027–0.9168`，宽度比例范围为 `0.045–0.96`，高度比例范围为 `0.06–0.97`。因此检测必须覆盖 P2、P3、P4、P5，单独 P2–P4 无法稳定表达接近整图的大框。

留出集只包含 `inclusion` 和 `pitted_surface`。mAP 必须只平均存在 GT 的类别；其余类别的错误预测仍计入 false positive，但不存在类别不能作为 AP=0 拉低均值。

## 权威实现与数学契约

- TOOD 的 TAL 使用分类分数和 IoU 的 task-alignment metric 选择正样本，并要求分类与定位在训练和推理时对齐：<https://arxiv.org/abs/2108.07755>。
- Ultralytics TaskAlignedAssigner 对 alignment metric 乘以每个 GT 的最大 overlap，并按最大 alignment metric 归一化后生成 `target_scores`：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py>。
- Ultralytics v8DetectionLoss 使用 `BCE.sum() / target_scores_sum`，并让 CIoU 与 DFL 同样按正样本质量权重及同一分母归一化：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py>。
- GFL 定义 DFL 为四边连续距离的离散分布学习；`reg_max=16` 表示 16 个 bins，而不是 17 个：<https://arxiv.org/abs/2006.04388>。
- Ultralytics Detect Head 对 box bias 和按 stride 计算的稀疏分类 bias 做显式初始化：<https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py>。

## Clean-break 设计

### Head 与 DFL

- `reg_max` 统一表示 bin 数量；输出通道改为 `4 * reg_max`，投影 bins 为 `[0, reg_max-1]`。
- 训练和解码使用同一布局：`[4, reg_max, H, W] -> [H*W, 4, reg_max]`。
- 分类 bias 按 `log(5 / num_classes / (input_size / stride)^2)` 初始化，回归 bias 初始化为 2。

### TAL 与 Loss

- 正样本选择继续使用 `score^alpha * IoU^beta`。
- 每个 GT 的 target quality 使用 `alignment * max_iou / max_alignment` 归一化。
- BCE、CIoU、DFL 的分母统一为 `max(target_scores.sum(), 1)`；CIoU 和 DFL 使用相同 quality weight。
- 删除单纯 one-hot 1.0 target 与全锚点 mean BCE 两条错误语义。

### 训练与选模

- seed 必须在模型、增强和 DataLoader 创建前设置。
- 训练改为 epoch 所有样本完整遍历；支持显式保守增强、warmup + cosine schedule。
- `best.pt` 在有 validation manifest 时只按 `mAP50`、`mAP50-95` 选择；没有 validation 时按 epoch 平均 loss 选择，禁止按单个 batch 选模。
- 输出目录已有日志/checkpoint 时 fail-close，防止不同实验混写。
- `random` 初始化只能使用全量解冻阶段，禁止冻结随机 Backbone/box regression。

### 评测

- mAP 解码阈值固定使用低阈值 `0.001`，与产品推理阈值分离。
- AP 只平均有 GT 的类别；无 GT 类预测仍进入 false-positive 统计。
- 训练、留出和最终全量模型分别记录，不混写指标。

### Checkpoint

DFL 通道语义发生破坏性变更，checkpoint 格式升级为 `chatdemo-vision-v2`；旧 `v1` 权重明确拒绝加载，不做 shape shim 或兼容 fallback。
