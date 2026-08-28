from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class ChatPair:
    """One sample for training.

    `context_turns` keeps previous turns to support multi-turn context.
    `think` is an optional reasoning chain inserted between prompt and reply
    during training to teach the model the Thinking Mode pattern.
    """

    context: str
    reply: str
    context_turns: List[str] = field(default_factory=list)
    think: str = ""


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _turn_from_dict(item: Dict[str, Any]) -> List[str]:
    """Parse one history/conversation item into one or two plain text turns."""
    turns: List[str] = []

    role = _as_text(item.get("role"))
    content = _as_text(item.get("content"))
    text = _as_text(item.get("text"))

    if role and content:
        turns.append(f"{role}: {content}")
        return turns

    if "user" in item and "assistant" in item:
        user = _as_text(item.get("user"))
        assistant = _as_text(item.get("assistant"))
        if user:
            turns.append(f"user: {user}")
        if assistant:
            turns.append(f"assistant: {assistant}")
        if turns:
            return turns

    if "user" in item and not turns:
        user = _as_text(item.get("user"))
        if user:
            turns.append(f"user: {user}")
    if "assistant" in item and not turns:
        assistant = _as_text(item.get("assistant"))
        if assistant:
            turns.append(f"assistant: {assistant}")
    if turns:
        return turns

    if role:
        value = content or text
        if value:
            turns.append(f"{role}: {value}")
            return turns

    if content:
        turns.append(content)
        return turns
    if text:
        turns.append(text)
        return turns

    # fallback for generic single-field dict
    for key in ("context", "prompt", "question", "reply", "response", "answer"):
        value = _as_text(item.get(key))
        if value:
            turns.append(value)
            break

    return turns


def _extract_turns(value: object) -> List[str]:
    if not isinstance(value, list):
        return []

    turns: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                turns.append(text)
            continue
        if isinstance(item, dict):
            turns.extend(_turn_from_dict(item))
            continue
    return turns


def _extract_pair_fields(record: Dict[str, Any], max_context_turns: int) -> ChatPair:
    think = _as_text(record.get("think"))

    def _clip(turns: List[str]) -> List[str]:
        return turns[-max_context_turns:] if max_context_turns > 0 else turns

    if "context" in record and "reply" in record:
        context = _as_text(record.get("context"))
        reply = _as_text(record.get("reply"))
        turns = _extract_turns(record.get("history") or record.get("conversation"))
        return ChatPair(context=context, reply=reply, context_turns=_clip(turns), think=think)

    if "user" in record and "assistant" in record:
        context = _as_text(record.get("user"))
        reply = _as_text(record.get("assistant"))
        turns = _extract_turns(record.get("history") or record.get("conversation"))
        return ChatPair(context=context, reply=reply, context_turns=_clip(turns), think=think)

    if "question" in record and "answer" in record:
        context = _as_text(record.get("question"))
        reply = _as_text(record.get("answer"))
        turns = _extract_turns(record.get("history") or record.get("conversation"))
        return ChatPair(context=context, reply=reply, context_turns=_clip(turns), think=think)

    if "prompt" in record and "response" in record:
        context = _as_text(record.get("prompt"))
        reply = _as_text(record.get("response"))
        turns = _extract_turns(record.get("history") or record.get("conversation"))
        return ChatPair(context=context, reply=reply, context_turns=_clip(turns), think=think)

    # fallback: infer pair from a flat conversation-like list
    conv = _extract_turns(record.get("conversation") or record.get("history") or [])
    if len(conv) >= 2:
        return ChatPair(context=conv[-2], reply=conv[-1], context_turns=conv[:-2][-max_context_turns:] if max_context_turns > 0 else conv[:-2], think=think)
    if len(conv) == 1:
        return ChatPair(context=conv[0], reply="", context_turns=[], think=think)

    raise ValueError(f"Unsupported sample fields: {record}")


def _pair_text(pair: ChatPair, sep_token: str) -> str:
    parts: List[str] = []
    if pair.context_turns:
        parts.extend(pair.context_turns)
    if pair.context:
        parts.append(pair.context)
    parts.append(pair.reply)
    return f"{sep_token}".join(parts)


def load_pairs(data_file: str, max_context_turns: int = 4) -> List[ChatPair]:
    """Load dialog pairs from jsonl file.

    Supported keys:
      - context + reply
      - user + assistant
      - question + answer
      - prompt + response
    plus optional:
      - history / conversation fields (list)
    """
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    pairs: List[ChatPair] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if line.startswith("[") or line.startswith("{"):
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"Unsupported JSON row {line_num}: {line[:80]}")
                pair = _extract_pair_fields(obj, max_context_turns=max_context_turns)
            else:
                # Fallback: `context\treply`
                if "\t" not in line:
                    raise ValueError(f"Unsupported plain-text line {line_num}: {line[:80]}")
                context, reply = line.split("\t", 1)
                pair = ChatPair(context=context.strip(), reply=reply.strip(), context_turns=[])

            if not pair.reply:
                # keep only samples with valid supervision targets
                continue
            pairs.append(pair)

    if not pairs:
        raise ValueError(f"No samples found in {path}")
    return pairs



def extract_corpus_for_vocab(pairs: Sequence[ChatPair]) -> List[str]:
    texts: List[str] = []
    for pair in pairs:
        texts.extend(pair.context_turns)
        texts.append(pair.context)
        texts.append(pair.reply)
        if pair.think:
            texts.append(pair.think)
    return texts


def encode_chat_pair(pair: ChatPair, tokenizer: CharTokenizer) -> List[int]:
    """Encode a multi-turn chat pair into token IDs.

    Format (with thinking):
      [BOS] hist1 [SEP] hist2 [SEP] ... prompt [SEP]
      <think> think_text </think> reply [EOS]

    Format (without thinking):
      [BOS] hist1 [SEP] hist2 [SEP] ... prompt [SEP] reply [EOS]
    """
    tokens = [tokenizer.bos_id]
    for turn in pair.context_turns:
        tokens.extend(tokenizer.encode(turn, add_special_tokens=False))
        tokens.append(tokenizer.sep_id)
    if pair.context:
        tokens.extend(tokenizer.encode(pair.context, add_special_tokens=False))
        tokens.append(tokenizer.sep_id)
    # Optional thinking chain
    if pair.think:
        tokens.append(tokenizer.think_id)
        tokens.extend(tokenizer.encode(pair.think, add_special_tokens=False))
        tokens.append(tokenizer.end_think_id)
    tokens.extend(tokenizer.encode(pair.reply, add_special_tokens=False))
    tokens.append(tokenizer.eos_id)
    return tokens


def encode_chat_prompt(prompt: str, tokenizer: CharTokenizer, context_turns: Sequence[str] = ()) -> List[int]:
    """Encode a prompt (plus optional history turns) up to the separator token."""
    tokens = [tokenizer.bos_id]
    for turn in context_turns:
        tokens.extend(tokenizer.encode(turn, add_special_tokens=False))
        tokens.append(tokenizer.sep_id)
    if prompt:
        tokens.extend(tokenizer.encode(prompt, add_special_tokens=False))
        tokens.append(tokenizer.sep_id)
    return tokens


class ChatDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[ChatPair],
        tokenizer: CharTokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for pair in pairs:
            tokens = encode_chat_pair(pair, tokenizer)
            if len(tokens) <= 1:
                continue

            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            x = torch.tensor(tokens[:-1], dtype=torch.long)
            y = torch.tensor(tokens[1:], dtype=torch.long)
            samples.append((x, y))

        if not samples:
            raise ValueError("No valid samples after tokenization")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def collate_chat_batch(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(x) for x, _ in batch)
    batch_size = len(batch)

    x = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)
    y = torch.full((batch_size, max_len), fill_value=pad_id, dtype=torch.long)

    for i, (xs, ys) in enumerate(batch):
        x[i, : len(xs)] = xs
        y[i, : len(ys)] = ys
    return x, y


def build_vocab_if_missing(vocab_path: str, pairs: Sequence[ChatPair], vocab_size: int) -> CharTokenizer:
    path = Path(vocab_path)
    if path.exists():
        return CharTokenizer.load(str(path))
    texts = extract_corpus_for_vocab(pairs)
    tokenizer = CharTokenizer.build_from_texts(texts, vocab_size=vocab_size)
    tokenizer.save(str(path))
    return tokenizer


def select_eval_prompts(pairs: Sequence[ChatPair], max_prompts: int = 3) -> List[str]:
    prompts: List[str] = []
    for pair in pairs:
        if pair.context:
            prompts.append(pair.context)
        if len(prompts) >= max_prompts:
            break
    return prompts
