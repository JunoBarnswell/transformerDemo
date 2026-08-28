from pathlib import Path

from chatdemo.chat_cli import chat_once
from .conftest import run_chat_module, write_jsonl
from .test_train_smoke import _write_yaml_config


def test_chat_cli_once(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    vocab_file = tmp_path / "vocab.json"
    out_dir = tmp_path / "out"

    rows = [
        {"context": "你是谁", "reply": "我正在测试 CLI 接口"},
        {"context": "你能聊天吗", "reply": "可以，我能基于CPU模型生成回复"},
        {"context": "再见", "reply": "再见，祝你今天愉快"},
        {"context": "帮我复习", "reply": "我们可以一起复习"},
    ]
    write_jsonl(data_file, rows)

    cfg = tmp_path / "config.yaml"
    _write_yaml_config(cfg, data_file, vocab_file, out_dir)

    train_res = run_chat_module("chatdemo.train", ["--config", str(cfg)])
    assert train_res.returncode == 0, train_res.stderr

    ckpt = out_dir / "checkpoint.pt"
    reply = chat_once(str(ckpt), "你是谁")
    assert isinstance(reply, str)
    assert len(reply) > 0
