from pathlib import Path

import torch

from chatdemo.data import ChatDataset, ChatPair, build_vocab_if_missing, collate_chat_batch, load_pairs
from chatdemo.tokenizer import CharTokenizer


def test_data_loading_and_collate(tmp_path: Path):
    src = tmp_path / "data.jsonl"
    src.write_text(
        '{"context":"你好","reply":"你也好"}\n'
        '{"context":"早上好","reply":"早安"}\n',
        encoding="utf-8",
    )

    pairs = load_pairs(str(src))
    assert len(pairs) == 2

    toks = CharTokenizer.build_from_texts(["你好你也好早上好早安"], vocab_size=64)
    ds = ChatDataset(pairs=pairs, tokenizer=toks, max_length=32)
    assert len(ds) == 2

    x, y = ds[0]
    assert x.dim() == 1 and y.dim() == 1
    assert len(x) == len(y)

    bx, by = collate_chat_batch([ds[0], ds[1]], pad_id=toks.pad_id)
    assert bx.shape == by.shape
    assert bx.shape[0] == 2
    assert bx.dtype == torch.long


def test_data_with_history_context_turns(tmp_path: Path):
    src = tmp_path / "data_history.jsonl"
    src.write_text(
        '{"user":"你是谁","assistant":"我是模型","history":["你好","你在吗" ,{"user":"今天好吗","assistant":"今天很好"}] ,"reply":"可以和你聊天"}\n'
        '{"history":[{"user":"上次你说什么"}],"question":"你还在吗","answer":"我还在"}\n',
        encoding="utf-8",
    )

    pairs = load_pairs(str(src), max_context_turns=1)
    assert len(pairs) == 2
    assert isinstance(pairs[0], ChatPair)
    assert len(pairs[0].context_turns) >= 1
    assert len(pairs[1].context_turns) >= 1


def test_build_vocab_if_missing(tmp_path: Path):
    corpus = tmp_path / "chat.jsonl"
    corpus.write_text('{"context":"a","reply":"b"}\n', encoding="utf-8")

    pairs = [ChatPair("a", "b")]
    vocab_path = tmp_path / "vocab.json"
    tokenizer = build_vocab_if_missing(str(vocab_path), pairs=pairs, vocab_size=64)
    assert vocab_path.exists()
    assert tokenizer.bos_id == 2
    assert tokenizer.eos_id == 3
