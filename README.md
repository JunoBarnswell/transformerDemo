# CPU Transformer Chat Demo（全自研）

这是一个**纯自研（从零参数初始化）**的 Transformer 聊天 Demo：

- 不使用任何第三方预训练模型权重；
- 只依赖通用库（PyTorch / PyYAML / pytest / numpy）；
- 默认跑在 CPU 上；
- 覆盖训练、推理、命令行聊天、评估（含 BLEU/人工评分模板）。

## 目录结构

```text
transformerDemo/
├─ requirements.txt
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ sample_conversations.jsonl
│  └─ vocab.json
├─ src/
│  └─ chatdemo/
│     ├─ __init__.py
│     ├─ tokenizer.py
│     ├─ data.py
│     ├─ model.py
│     ├─ train.py
│     ├─ infer.py
│     ├─ evaluate.py
│     └─ chat_cli.py
├─ tests/
│  ├─ conftest.py
│  ├─ test_tokenizer.py
│  ├─ test_data.py
│  ├─ test_model.py
│  ├─ test_train_smoke.py
│  ├─ test_infer_smoke.py
│  └─ test_chat_cli.py
└─ .github/workflows/ci.yml
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
```

## 快速开始（CPU）

### 1) 训练（含上下文窗口/日志）

```bash
PYTHONPATH=src python -m chatdemo.train --config configs/base.yaml
```

默认会产出：

- `outputs/run/checkpoint.pt`：最终 checkpoint
- `outputs/run/best.pt`：验证集最优 checkpoint（本 demo 当前使用同一 loader 验证）
- `outputs/run/train_log.jsonl`：训练每步日志（含 train/val 记录）
- `outputs/run/loss_curve.json`：训练与验证 loss 曲线聚合文件，方便可视化
- `outputs/run/inference_log.jsonl`：训练过程按验证周期抽样的推理样例

### 2) 推理

```bash
PYTHONPATH=src python -m chatdemo.infer \
  --checkpoint outputs/run/checkpoint.pt \
  --prompt "你好，今天心情怎么样？"
```

返回会包含 prompt、response、response_ids、耗时（seconds）等字段。

### 3) 命令行聊天

```bash
PYTHONPATH=src python -m chatdemo.chat_cli --checkpoint outputs/run/checkpoint.pt
```

输入 `quit` / `exit` 结束。

### 4) 评估（BLEU + 人工评分模板）

```bash
PYTHONPATH=src python -m chatdemo.evaluate \
  --checkpoint outputs/run/checkpoint.pt \
  --eval-data data/sample_conversations.jsonl \
  --output-dir outputs/eval
```

评估会输出：

- `outputs/eval/predictions.jsonl`：每条样本 `prompt/reference/prediction`
- `outputs/eval/metrics.json`：BLEU 汇总
- `outputs/eval/human_scoring_template.csv`：人工评分填表模板（字段：`prompt, reference, prediction, human_score, comment`）

### 5) 可视化 loss 曲线（实验记录）

```bash
PYTHONPATH=src python -m chatdemo.plot_loss \
  --loss-curve outputs/run/loss_curve.json \
  --output-csv outputs/run/loss_curve.csv
```

如环境已安装 `matplotlib`，可附加 `--plot --output-plot outputs/run/loss_curve.png` 额外导出曲线图。

## 数据格式

本 demo 主要读取 JSONL，每行支持以下字段之一：

- `context` + `reply`
- `user` + `assistant`
- `question` + `answer`
- `prompt` + `response`

可选字段支持多轮上下文：

- `history`：历史轮次列表（字符串或 `{"user":"...","assistant":"..."}`）
- `conversation`：与 history 同语义

训练时会将历史轮次按最近 `max_context_turns` 进行拼接作为上下文。

## 配置扩展

`configs/base.yaml` 关键可调参数：

- `data.max_context_turns`：训练样本中保留的上下文轮数
- `train.train_log_file` / `train.loss_curve_file` / `train.inference_log_file`
- `train.num_infer_prompts`：训练中每次验证后要抽样推理的条数
- `generation.max_new_tokens`、`temperature`、`top_k`、`top_p`

## 约束说明

- 全程不使用第三方预训练模型；
- 词表与模型参数都由本地配置语料/随机初始化构建；
- 为演示与 CI 设计的小模型，参数较小，便于 CPU 跑通。
