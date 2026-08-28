from pathlib import Path

import json

from .conftest import run_chat_module, write_jsonl
from .test_train_smoke import _write_yaml_config


def test_evaluate_smoke(tmp_path: Path):
    data_file = tmp_path / "train.jsonl"
    vocab_file = tmp_path / "vocab.json"
    out_dir = tmp_path / "out"
    eval_out = tmp_path / "eval_out"

    rows = [
        {"context": "今天天气怎么样", "reply": "今天天气很好"},
        {"user": "你是谁", "assistant": "我是一个 CPU 聊天模型"},
        {"question": "你会记住我吗", "answer": "我只会记住最近的上下文"},
        {
            "context": "帮我复习吧",
            "reply": "可以，我们先从基础开始",
            "history": [
                {"user": "你是谁", "assistant": "我是你的模型"},
                {"user": "你能做什么", "assistant": "我可以回答并聊天"},
            ],
        },
    ]
    write_jsonl(data_file, rows)

    cfg = tmp_path / "config.yaml"
    _write_yaml_config(cfg, data_file, vocab_file, out_dir)

    train_res = run_chat_module("chatdemo.train", ["--config", str(cfg)])
    assert train_res.returncode == 0, train_res.stderr

    ckpt = out_dir / "checkpoint.pt"
    assert ckpt.exists()

    eval_res = run_chat_module(
        "chatdemo.evaluate",
        [
            "--checkpoint",
            str(ckpt),
            "--eval-data",
            str(data_file),
            "--output-dir",
            str(eval_out),
            "--max-samples",
            "2",
            "--max-context-turns",
            "2",
            "--max-new-tokens",
            "6",
            "--bleu-max-n",
            "4",
            "--human-template",
            "human.csv",
        ],
    )
    assert eval_res.returncode == 0, eval_res.stderr

    payload = json.loads(eval_res.stdout.strip())
    assert payload["num_samples"] == 2
    assert payload["bleu"] >= 0.0

    metrics = Path(payload["metrics"])
    human = Path(payload["human_template"])
    preds = Path(payload["predictions"])

    assert metrics.exists() and metrics.is_file()
    assert human.exists() and human.is_file()
    assert preds.exists() and preds.is_file()

    metric_obj = json.loads(metrics.read_text(encoding="utf-8"))
    assert "bleu" in metric_obj
    assert metric_obj["num_samples"] == 2
    assert len(metric_obj.get("samples", [])) == 2
