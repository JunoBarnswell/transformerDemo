from pathlib import Path

import json

from .conftest import run_chat_module, write_jsonl



def _write_yaml_config(path: Path, data_path: Path, vocab_path: Path, output_dir: Path) -> None:
    cfg = f"""
seed: 7

data:
  data_file: {data_path.as_posix()}
  vocab_path: {vocab_path.as_posix()}
  vocab_size: 200
  seq_len: 48

train:
  output_dir: {output_dir.as_posix()}
  batch_size: 2
  num_workers: 0
  max_steps: 2
  max_epochs: 1
  learning_rate: 0.001
  grad_clip: 1.0
  eval_interval: 1
  save_interval: 1
  log_interval: 1

model:
  d_model: 32
  nhead: 2
  n_layers: 1
  ff_dim: 64
  dropout: 0.1
"""
    path.write_text(cfg.strip() + "\n", encoding="utf-8")


def test_train_smoke(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    vocab_file = tmp_path / "vocab.json"
    out_dir = tmp_path / "out"

    rows = [
        {"context": "你是谁", "reply": "我是一个聊天模型"},
        {"context": "你可以做什么", "reply": "我可以在 CPU 上对话"},
        {"context": "你好", "reply": "你好呀"},
        {"context": "再见", "reply": "再见！"},
    ]
    write_jsonl(data_file, rows)

    cfg = tmp_path / "config.yaml"
    _write_yaml_config(cfg, data_file, vocab_file, out_dir)

    res = run_chat_module("chatdemo.train", ["--config", str(cfg)])
    assert res.returncode == 0, res.stderr
    assert (out_dir / "checkpoint.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "train_log.jsonl").exists()
    assert (out_dir / "loss_curve.json").exists()
    assert (out_dir / "inference_log.jsonl").exists()
