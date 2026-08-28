# CPU Transformer Chat Demo (全自研)

这是一个**纯自研 (从零参数初始化)**的 Transformer 聊天 demo：
- 不使用任何第三方预训练模型权重；
- 支持 CPU 训练与推理；
- 覆盖训练、推理和聊天 CLI 的最小验证。

## 目录结构

```text
transformerDemo/
├─ requirements.txt
├─ configs/
│  └─ base.yaml
├─ data/
│  ├─ sample_conversations.jsonl
│  └─ vocab.json               # 首次训练后生成
├─ src/
│  └─ chatdemo/
│     ├─ __init__.py
│     ├─ tokenizer.py
│     ├─ data.py
│     ├─ model.py
│     ├─ train.py
│     ├─ infer.py
│     └─ chat_cli.py
├─ tests/
│  ├─ conftest.py
│  ├─ test_tokenizer.py
│  ├─ test_data.py
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

### 1) 训练

```bash
PYTHONPATH=src python -m chatdemo.train --config configs/base.yaml
```

训练日志会保存到 `outputs/run/run.log`，模型保存到 `outputs/run/checkpoint.pt`。

### 2) 推理

```bash
PYTHONPATH=src python -m chatdemo.infer --checkpoint outputs/run/checkpoint.pt --prompt "你好，今天心情怎么样？"
```

### 3) 命令行聊天

```bash
PYTHONPATH=src python -m chatdemo.chat_cli --checkpoint outputs/run/checkpoint.pt
```

输入 `quit` / `exit` 退出。

## 约束说明

- 只用 `torch` 等通用库，不加载预训练大模型；
- 词表通过本地语料训练（`build_vocab_from_file`）；
- 为演示与 CI，默认参数非常小。
