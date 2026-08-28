# CPU Transformer Chat Demo（全自研，含 RoPE、双层记忆管理与思考模式）

这是一个**纯自研（从零参数初始化）**的 Transformer 聊天架构系统：

- **0 预训练权重**：不使用任何第三方模型权重，纯原生 PyTorch 搭建；
- **RoPE（旋转位置编码）**：自研 `MultiHeadSelfAttention`，无固定位置参数限制，支持理论上超长（1M+）上下文外推；
- **双层记忆管理器（ConversationMemory）**：永久摘要层（`permanent_summary`）+ 近端滑动窗口（`recent_turns`），配合模型自生成压缩（`compact`），跨轮对话永不遗忘第 1 轮；
- **思考模式（Thinking Mode）**：支持 `<think>推理链</think>` 特殊 token 训练与解析，分离思考过程与最终答复；
- **全套工程闭环**：训练、早停机制（patience ≥ 10）、推理、多轮 CLI 对话、BLEU 自动化评估、人工评分模板与 Loss 曲线可视化；
- **CPU 友好**：278k 级别参数量，单核 CPU 秒级响应。

---

## 目录结构

```text
transformerDemo/
├─ requirements.txt
├─ configs/
│  └─ base.yaml                 # 训练与推理配置（含 memory、thinking、generation 配置）
├─ data/
│  ├─ sample_conversations.jsonl # 训练样本（含多轮历史、think 思考链标注）
│  └─ vocab.json                # 字符级词表（含 <pad>, <unk>, <bos>, <eos>, <sep>, <think>, </think>）
├─ src/
│  └─ chatdemo/
│     ├─ __init__.py
│     ├─ tokenizer.py           # CharTokenizer，支持 think 特殊 token
│     ├─ data.py                # 样本解析与 prompt+think+reply 格式编码
│     ├─ model.py               # RoPE + Pre-LN TransformerBlock + 权重共享 LM
│     ├─ memory.py              # 双层记忆管理器（ConversationMemory）
│     ├─ train.py               # 训练脚本（早停、val 评估、多采样推理日志）
│     ├─ infer.py               # 推理生成与 thinking 分离
│     ├─ evaluate.py            # BLEU 评估与人工评分表导出
│     ├─ plot_loss.py           # Loss 曲线绘制与 CSV 导出
│     └─ chat_cli.py            # 交互式 CLI 聊天（支持 /memory、/reset、--show-thinking）
├─ tests/                       # 32 项自动化测试（100% 通过）
│  ├─ conftest.py
│  ├─ test_tokenizer.py
│  ├─ test_data.py
│  ├─ test_model.py
│  ├─ test_memory.py
│  ├─ test_thinking.py
│  ├─ test_train_smoke.py
│  ├─ test_infer_smoke.py
│  └─ test_chat_cli.py
└─ outputs/
   ├─ run_rope/
   │  ├─ best.pt                # 验证最优 checkpoint (val_loss ≈ 0.08)
   │  ├─ checkpoint.pt          # 最终 checkpoint
   │  ├─ loss_curve.json / .png # 训练曲线
   │  └─ train_log.jsonl
   └─ eval/
      ├─ metrics.json           # BLEU 评估结果 (BLEU ≈ 0.58)
      └─ human_scoring_template.csv
```

---

## 核心架构设计

### 1. 旋转位置编码 (RoPE)
摒弃传统的 `nn.Embedding(max_seq_len, d_model)` 绝对位置嵌入，采用无参数的旋转矩阵变换：
$$\mathbf{q}_m = \mathbf{R}_m \mathbf{W}_q \mathbf{x}_m, \quad \mathbf{k}_n = \mathbf{R}_n \mathbf{W}_k \mathbf{x}_n$$
内积仅依赖相对位移 $m-n$，序列长度不再受限，推理时可按需支持更长上下文。

### 2. 双层持久记忆机制 (ConversationMemory)
- **永久记忆层 (`permanent_summary`)**：当近期对话轮次达到阈值（如 8 轮）时，调用模型自生成历史摘要并沉淀为永久摘要，不随时间滑动弹出；
- **滑动窗口层 (`recent_turns`)**：保留最近 $N$ 轮原始对话详情；
- **Prompt 构造**：`[历史摘要] ... \n 最近对话...`，保证即使经历 10+ 轮甚至数百轮，第 1 轮的核心关键信息始终被模型感知。

### 3. 思考模式 (Thinking Mode)
- 词表中加入 `<think>` 和 `</think>` 特殊 token；
- 训练格式：`[BOS] 历史上下文 [SEP] 当前提问 [SEP] <think> 思考与推理过程 </think> 回复内容 [EOS]`；
- 推理引擎自动切分 `thinking` 链与 `response` 文本，支持通过 `--show-thinking` 观察模型的内部思考。

---

## 快速使用

### 1. 训练模型

```bash
PYTHONPATH=src python -m chatdemo.train --config configs/base.yaml
```

### 2. 交互式多轮对话（含思考模式与记忆管理）

```bash
# 启动聊天，开启思考过程显示
PYTHONPATH=src python -m chatdemo.chat_cli \
  --checkpoint outputs/run_rope/best.pt \
  --show-thinking

# 交互指令：
#   /memory  - 查看当前记忆状态（永久摘要 + 近期窗口）
#   /reset   - 清空当前会话记忆
#   quit     - 退出
```

### 3. 单次推理测试

```bash
PYTHONPATH=src python -m chatdemo.infer \
  --checkpoint outputs/run_rope/best.pt \
  --prompt "你好，请介绍一下自己"
```

### 4. 自动化评测 (BLEU + 人工评分模板)

```bash
PYTHONPATH=src python -m chatdemo.evaluate \
  --checkpoint outputs/run_rope/best.pt \
  --eval-data data/sample_conversations.jsonl \
  --output-dir outputs/eval
```

### 5. 可视化 Loss 训练曲线

```bash
PYTHONPATH=src python -m chatdemo.plot_loss \
  --loss-curve outputs/run_rope/loss_curve.json \
  --plot \
  --output-plot outputs/run_rope/loss_curve.png \
  --output-csv outputs/run_rope/loss_curve.csv
```

### 6. 运行全量测试套件

```bash
python -m pytest tests/ -q
# 输出: 32 passed in 16.46s (100% 通过)
```

---

## 工业缺陷视觉双主线（Issue #1）

视觉模块位于 `src/chatdemo/vision/`，只共享一套 Backbone + 多尺度 Neck，之后分叉为两个独立任务头：

- Detection Head：P2/P3/P4（可选 P5），Anchor-Free、解耦分类/回归、DFL `reg_max=16`、Task-Aligned Assigner、CIoU/BCE/NMS；框不会由 mask 派生；
- Segmentation Decoder：使用 P1/P2/P3/P4 输出独立语义 mask，支持 CE/Focal + Dice；检测和分割可以使用不同标注集；
- 输入内部采用 640 Letterbox，API 中的 `box_xyxy` 统一为原图映射后的 canonical `640×640` 坐标；mask 与 annotated image 恢复为原图尺寸；
- 可选重叠 Tile 用于大图极小缺陷，Tile 结果最终仍合并到同一个 canonical 坐标系。

### 视觉数据 manifest

每行一个 JSON 对象，路径相对 manifest 文件解析。`boxes` 字段存在但为空代表检测负样本；字段缺失代表没有该任务监督，不能被当作负样本。

```json
{"image":"ok.jpg","boxes":[],"labels":[]}
{"image":"box.jpg","boxes":[[12,20,80,90]],"labels":[0]}
{"image":"mask.jpg","mask":"mask.png"}
{"image":"both.jpg","boxes":[{"box":[12,20,80,90],"label":0}],"mask":"both_mask.png"}
```

### 训练与推理

`configs/vision.yaml` 的默认初始化模式是 `project_base`，要求提供真实的 `weights/vision_base.pt`。只有显式改为 `random` 才允许开发/单元冒烟从随机参数开始；缺少生产权重时不会静默降级。

```bash
# 真实标注 manifest + project_base 权重
PYTHONPATH=src python -m chatdemo.vision.train --config configs/vision.yaml

# 仅在明确的开发配置中使用 random 初始化
PYTHONPATH=src python -m chatdemo.analyze_image \
  --image path/to/image.jpg \
  --checkpoint outputs/vision/checkpoint.pt \
  --output-dir outputs/vision \
  --return-json
```

也可分别运行 `chatdemo.detect`、`chatdemo.segment`，或给现有 `chatdemo.chat_cli` 同时传入 `--image` 与 `--vision-checkpoint`，将一次视觉分析作为可检查的显式文本上下文复用于多轮聊天。

### YOLOv8 对标边界

```bash
PYTHONPATH=src python -m chatdemo.vision.benchmark \
  --checkpoint outputs/vision/checkpoint.pt \
  --manifest path/to/manifest.jsonl \
  --yolov8-weights path/to/yolov8n.pt
```

benchmark 要求同一 manifest、同一 640 输入和真实 YOLOv8 权重；没有这些外部材料时不会输出伪造的 parity 结论。当前仓库未包含工业视觉数据、`vision_base.pt` 或 YOLOv8 权重，因此这部分需要使用者提供数据与经过许可审查的权重后才能完成实测。
