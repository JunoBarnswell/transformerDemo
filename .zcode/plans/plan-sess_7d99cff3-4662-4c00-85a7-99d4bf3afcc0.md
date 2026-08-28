## 完整实施计划：RoPE + Memory Manager + Thinking Mode

### 选定方案
- **位置编码**：完整 RoPE（自定义 MultiHeadAttention 替代 nn.TransformerEncoder）
- **记忆摘要**：模型自生成摘要（利用训练好的模型本身生成历史摘要句）
- **思考模式**：本次同步实现（`<think>…</think>` 特殊 token）

---

### 改动文件清单

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `src/chatdemo/model.py` | **重写** | 引入 RoPE + 自定义 MultiHeadSelfAttention + TransformerBlock，移除 nn.TransformerEncoder |
| `src/chatdemo/tokenizer.py` | **扩展** | 新增 `<think>`、`</think>` 两个特殊 token |
| `src/chatdemo/data.py` | **扩展** | `encode_chat_pair` 支持可选 `think_text` 字段；新增 `encode_chat_prompt_with_thinking` |
| `src/chatdemo/memory.py` | **新建** | `ConversationMemory` 类（permanent_summary + recent_turns deque + compact()） |
| `src/chatdemo/infer.py` | **扩展** | `generate_text` 解析 `<think>…</think>` 分离推理链与最终回复 |
| `src/chatdemo/chat_cli.py` | **重写** | main() 维护 `ConversationMemory` 实例，接入 memory 和 thinking 显示 |
| `configs/base.yaml` | **扩展** | 新增 `memory` 和 `thinking` 配置节，seq_len 改为 512 |
| `data/sample_conversations.jsonl` | **扩展** | 在 10 条样本中新增 `"think"` 字段，构建思考模式训练样本 |
| `tests/test_memory.py` | **新建** | 测试 push/compact/build_context_prompt 逻辑 |
| `tests/test_thinking.py` | **新建** | 测试 think token 的编解码与推理分离 |

---

### 详细实现说明

#### Step 1 — `model.py` 引入 RoPE

新增 `RoPEEmbedding` 类：
```python
class RoPEEmbedding:
    # 动态计算 sin/cos 旋转矩阵，不存储参数
    def apply(self, q, k):  # q/k shape: [B, H, T, head_dim]
        # 乘旋转矩阵：偶数维 cos，奇数维 sin 交叉
```

新增 `MultiHeadSelfAttention(nn.Module)`：
```python
# 替代 nn.MultiheadAttention，内嵌 RoPE
# 包含：Q/K/V projection → RoPE 旋转 Q/K → scaled dot-product → causal mask → out projection
```

新增 `TransformerBlock(nn.Module)`（= SelfAttn + FFN + 2× LayerNorm，Pre-LN 顺序）

修改 `TinyTransformerLM`：
- 删除 `self.pos_emb = nn.Embedding(max_seq_len, d_model)`
- 删除 `nn.TransformerEncoder`
- 改为 `self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(n_layers)])`
- forward 中去掉 pos_emb，改为 RoPE 在 SelfAttn 内部处理
- **参数量不变**（RoPE 没有参数，只是旋转计算）

#### Step 2 — `tokenizer.py` 新增特殊 token

```python
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>", "<think>", "</think>"]
```
新增 `think_id` / `end_think_id` property，与其他特殊 token 完全对称。

#### Step 3 — `data.py` 训练数据支持 think 字段

`encode_chat_pair` 新增分支：
```
[BOS] hist1 [SEP] hist2 [SEP] prompt [SEP]
<think> think_text </think>
reply [EOS]
```
若样本无 `think` 字段，格式保持原样（向前兼容）。

#### Step 4 — `memory.py` 实现 ConversationMemory

```python
class ConversationMemory:
    def __init__(self, recent_window=8, compact_threshold=8, summary_max_tokens=40):
        self.permanent_summary: str = ""   # 永久摘要，随轮数增长
        self.recent_turns: deque = deque(maxlen=recent_window)  # 原文滑动窗口

    def push(self, user_msg: str, assistant_reply: str):
        # 将一轮对话压入 recent_turns

    def should_compact(self) -> bool:
        # recent_turns 达到 compact_threshold 时返回 True

    def compact(self, model, tokenizer, max_tokens=40):
        # prompt = "请用一句话总结以下对话：\n" + recent_turns 前半部分
        # 调用 generate_text 生成摘要句
        # permanent_summary += "\n" + 摘要句
        # 从 recent_turns 中弹出已摘要的轮次

    def build_context_turns(self) -> List[str]:
        # 返回 [permanent_summary] + list(recent_turns)
        # 如果 permanent_summary 为空则不插入
```

#### Step 5 — `infer.py` 解析 thinking 链

`generate_text` 返回值新增字段：
```python
return {
    "prompt": prompt,
    "response": response_text,     # </think> 之后到 EOS 之前的文字
    "thinking": thinking_text,     # <think>…</think> 之间的文字（可为空字符串）
    "steps": len(response_ids),
    "elapsed_sec": elapsed,
}
```
分离逻辑：找到 `<think>` 和 `</think>` 的位置，切割 response_ids。

#### Step 6 — `chat_cli.py` 接入 memory + thinking 展示

```python
def main():
    memory = ConversationMemory(recent_window=cfg.memory.recent_window)
    while True:
        user_msg = input("你: ")
        context_turns = memory.build_context_turns()
        res = generate_reply(..., context_turns=context_turns)
        
        if args.show_thinking and res["thinking"]:
            print(f"[思考过程] {res['thinking']}")
        print(f"助手: {res['response']}")
        
        memory.push(user_msg, res["response"])
        if memory.should_compact():
            memory.compact(model, tokenizer)
```

#### Step 7 — `configs/base.yaml` 新增配置节

```yaml
model:
  d_model: 128
  nhead: 4
  n_layers: 2
  ff_dim: 192
  dropout: 0.1
  pos_encoding: rope    # ← 新增

data:
  seq_len: 512          # ← 从 128 改为 512（RoPE 支持任意长度，训练用 512）

memory:
  recent_window: 8
  compact_threshold: 8
  summary_max_tokens: 40

thinking:
  enabled: true
  expose_thinking: false
```

#### Step 8 — 扩充训练数据中的 think 样本

在 `sample_conversations.jsonl` 中，挑选 10 条问答加入 `"think"` 字段，例如：
```json
{"context": "什么是 Transformer？", "think": "用户在询问技术概念，需要给出简明的定义并强调核心机制。", "reply": "Transformer 是一种基于自注意力机制的深度学习架构…"}
```

#### Step 9 — 新增测试

`tests/test_memory.py`：
- push 多轮后 `recent_turns` 长度正确
- `should_compact` 触发时机正确
- `build_context_turns` 返回永久摘要 + 近端历史的拼接

`tests/test_thinking.py`：
- tokenizer 正确编解码 `<think>` / `</think>`
- `generate_text` 在有 `<think>` token 时能正确分离 `thinking` 和 `response` 字段

#### Step 10 — 重新训练并端到端验证

删除旧 vocab.json 和 checkpoint，用新配置重新训练 600~800 steps，验证：
- `pytest` 全量 13 项测试通过（新增 2 个测试文件）
- CLI 多轮对话：10 轮后第 1 轮信息依然通过 `permanent_summary` 出现在上下文中
- thinking 字段在 `--show-thinking` 时可见

---

### 执行顺序（实际编码时的依赖顺序）

1. tokenizer.py（新增特殊 token，其他模块依赖）
2. model.py（RoPE + TransformerBlock）
3. data.py（扩展 encode_chat_pair）
4. memory.py（新建）
5. infer.py（分离 thinking/response）
6. chat_cli.py（接入 memory）
7. configs/base.yaml + 数据扩充
8. 新增测试 → 重新训练 → 全量验证
