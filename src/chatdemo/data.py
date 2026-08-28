from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import CharTokenizer


@dataclass
class ChatPair:
    context: str
    reply: str


def load_pairs(data_file: str) -> List[ChatPair]:
    """Load dialog pairs from jsonl/json style files.

    Supported keys:
      - context + reply
      - user + assistant
      - question + answer
      - prompt + response
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
                if "context" in obj and "reply" in obj:
                    context, reply = obj["context"], obj["reply"]
                elif "user" in obj and "assistant" in obj:
                    context, reply = obj["user"], obj["assistant"]
                elif "question" in obj and "answer" in obj:
                    context, reply = obj["question"], obj["answer"]
                elif "prompt" in obj and "response" in obj:
                    context, reply = obj["prompt"], obj["response"]
                else:
                    raise ValueError(f"Unsupported line {line_num}: {line[:80]}")
            else:
                # Fallback: `context\treply`
                if "\t" not in line:
                    raise ValueError(f"Unsupported plain-text line {line_num}: {line[:80]}")
                parts = line.split("\t", 1)
                context, reply = parts

            pairs.append(ChatPair(str(context).strip(), str(reply).strip()))

    if not pairs:
        raise ValueError(f"No samples found in {path}")
    return pairs


def _join_pair(pair: ChatPair, sep_token: str) -> str:
    if pair.context:
        return f"{pair.context}{sep_token}{pair.reply}"
    return pair.reply


def extract_corpus_for_vocab(pairs: Sequence[ChatPair]) -> List[str]:
    return [f"{pair.context}{pair.reply}" for pair in pairs]


class ChatDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[ChatPair],
        tokenizer: CharTokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sep_token = tokenizer.id_to_token[tokenizer.sep_id]

        samples = []
        for pair in pairs:
            text = _join_pair(pair, self.sep_token)
            tokens = tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) <= 1:
                continue

            # Keep sequence length safe for model
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
