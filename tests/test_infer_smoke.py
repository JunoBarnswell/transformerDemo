from pathlib import Path

from .conftest import run_chat_module, write_jsonl
from .test_train_smoke import _write_yaml_config


def test_infer_smoke(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    vocab_file = tmp_path / "vocab.json"
    out_dir = tmp_path / "out"

    rows = [
        {"context": "今天怎么样", "reply": "今天很适合写代码"},
        {"context": "最近好吗", "reply": "最近还不错，多亏有你在"},
        {"context": "你会记住我吗", "reply": "在本轮对话里我会根据上下文回答"},
        {"context": "你喜欢什么", "reply": "我喜欢学习和聊天"},
    ]
    write_jsonl(data_file, rows)

    cfg = tmp_path / "config.yaml"
    _write_yaml_config(cfg, data_file, vocab_file, out_dir)

    train_res = run_chat_module("chatdemo.train", ["--config", str(cfg)])
    assert train_res.returncode == 0, train_res.stderr

    ckpt = out_dir / "checkpoint.pt"
    infer_res = run_chat_module(
        "chatdemo.infer",
        [
            "--checkpoint", str(ckpt),
            "--prompt", "今天心情不好",
            "--max-new-tokens", "8",
            "--return-json",
        ],
    )
    assert infer_res.returncode == 0, infer_res.stderr
    payload = __import__("json").loads(infer_res.stdout.strip())
    assert "response" in payload
    assert isinstance(payload["response"], str)
