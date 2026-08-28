from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]


@dataclass
class CharTokenizer:
    """A tiny character-level tokenizer built from scratch."""

    token_to_id: dict
    id_to_token: list

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def sep_id(self) -> int:
        return self.token_to_id["<sep>"]

    @classmethod
    def build_from_texts(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4000,
    ) -> "CharTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text)

        # Keep the specials fixed.
        tokens = list(SPECIAL_TOKENS)
        common_tokens = [ch for ch, _ in counter.most_common(max(0, vocab_size - len(tokens)))]
        tokens.extend([t for t in common_tokens if t not in tokens])

        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        return cls(token_to_id=token_to_id, id_to_token=tokens)

    @classmethod
    def build_from_file(
        cls,
        file_path: str,
        vocab_size: int = 4000,
    ) -> "CharTokenizer":
        text = Path(file_path).read_text(encoding="utf-8")
        texts = [line.strip() for line in text.splitlines() if line.strip()]
        return cls.build_from_texts(texts, vocab_size=vocab_size)

    @classmethod
    def load(cls, file_path: str) -> "CharTokenizer":
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return cls(token_to_id=data["token_to_id"], id_to_token=data["id_to_token"])

    def save(self, file_path: str) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        Path(file_path).write_text(
            json.dumps({
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int]) -> str:
        out_tokens = []
        for idx in ids:
            token = self.id_to_token[idx]
            if token in SPECIAL_TOKENS:
                continue
            out_tokens.append(token)
        return "".join(out_tokens)
