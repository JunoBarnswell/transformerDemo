已理解：你要一个**本地空目录下、CPU 可跑、聊天对话方向、全流程覆盖训练+推理测试**的 Transformer demo，且模型必须**全自研，不使用任何第三方预训练模型权重**。下面是可直接执行的实现计划（基于已有探索：目录为空、无现成文件）。

## 目标与约束
- 模型从零初始化参数（随机初始化），不加载任何现成 checkpoint；
- 仅依赖通用库（如 PyTorch、PyYAML、pytest、numpy），但不使用现成聊天大模型 API 或预训练权重；
- 默认支持 CPU 训练与推理；
- 给出可验证的训练/推理/对话回归测试。

## 实施计划

### 阶段 1：项目脚手架（本地空目录初始化）
1. 创建最小工程结构：
`requirements.txt`, `README.md`, `configs/base.yaml`, `src/chatdemo/{__init__.py,tokenizer.py,data.py,model.py,train.py,infer.py,chat_cli.py}`, `tests/{test_tokenizer.py,test_data.py,test_train_smoke.py,test_infer_smoke.py,test_chat_cli.py}`, `.github/workflows/ci.yml`。
2. `requirements.txt` 固定可复现版本（如 `torch`, `pyyaml`, `numpy`, `pytest`, `torchtext` 可选）。
3. `README.md` 写清楚：CPU 运行方式、训练数据准备、训练和推理命令。

### 阶段 2：全自研数据与分词（拒绝第三方模型）
1. 实现 `data.py`：读取纯文本对话文件（建议支持 JSONL/CSV，字段如 `context`, `reply`）；
2. 先做“从零字符级/词级 tokenizer”而非外部 tokenizer 模型：
   - 以训练语料统计频次构建词表或字符表；
   - 生成 `<pad>, <unk>, <bos>, <eos>` 固定 token；
   - 输出 `vocab.json` 与 `Tokenizer` 编码/解码函数；
3. 生成训练样本为 `(x, y)`：对每条 `context+reply` 构造 `tokens[:-1] -> tokens[1:]` 的 LM 监督；
4. 支持可切换为外部语料（如 Cornell/DailyDialog）但仅作为文本数据源，不加载预训练权重。

### 阶段 3：全自研 Transformer 模型
1. `model.py` 使用 `torch.nn.Transformer` + 自定义 embedding/位置编码 + `lm_head` 组装 tiny decoder-only 或小型 seq2seq（优先实现更简单稳定的 decoder-only）；
2. 参数初始值从头开始随机初始化；
3. 实现 causal mask 与 padding mask，确保训练时无未来信息泄漏；
4. `forward` 只做前向 logits 输出，`generate` 支持 greedy + 可选 top-k/top-p；
5. 提供 CPU-friendly 默认超参：`d_model=128`, `nhead=4`, `n_layers=2`, `ff_dim=256`, `seq_len=64~96`。

### 阶段 4：训练流程（train）
1. `train.py`：
   - 支持配置文件 + 命令行参数（seed、batch_size、lr、max_steps、epochs）；
   - 训练循环 + 验证（loss、perplexity）+ 早停（可选）+ checkpoint 保存；
   - `torch.device('cpu')` 强制 CPU。
2. 输出 artifacts：`checkpoint.pt`, `metrics.jsonl`, `config.json`，用于推理回放与可重复性。

### 阶段 5：推理与聊天测试（infer/chat）
1. `infer.py`：给定 checkpoint + prompt，执行生成并返回文本、耗时与 token 数；
2. `chat_cli.py`：命令行聊天循环（exit/quit 停止）；
3. 增加一条可选 FastAPI 路由（`/chat`）示例文件（可后续扩展）。

### 阶段 6：测试体系（必须覆盖训练和推理）
1. 单元测试：
   - tokenizer 可逆性；
   - 数据 padding/shape 正确；
   - 模型 forward/logits 形状与掩码逻辑；
2. Smoke test：
   - 极小数据/极短步数下完成一次 `train` 并产出 checkpoint；
   - 基于该 checkpoint 完成一次 `infer`，输出非空并带停止 token；
3. 回归测试：
   - 固定 seed 下同一输入在同一模型上输出一致；
   - `chat_cli` 至少一次问答流程可运行。
4. CI（`.github/workflows/ci.yml`）在 CPU runner 上依次跑：`tests/test_train_smoke.py` -> `tests/test_infer_smoke.py` -> `tests/test_chat_cli.py`。

### 阶段 7：交付与验收
1. 在 README 给出最小可复现命令（安装、训练、推理、聊天）；
2. 给出“预期效果”与“CPU 性能范围”（例如 2-4 层、`batch=16` 小样本 5~10 分钟内完成 smoke）；
3. 可选增强：加入 BLEU/人工打分脚本、top-k/top-p 对比与采样温度实验。

## 结果确认标准（DoD）
- 无任何预训练模型加载；
- `python -m pytest tests` 在空环境下可跑通（至少 smoke）；
- 可在 CPU 上完成一次 end-to-end：`训练 -> 保存权重 -> 推理 -> chat`。

如果你认可此方案，我会按该结构直接写出每个文件的具体代码与超参配置。