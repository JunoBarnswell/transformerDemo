from pathlib import Path

import csv

from .conftest import run_chat_module


def test_plot_loss_smoke(tmp_path: Path):
    curve = {
        "train": [
            {"step": 1, "split": "train", "loss": 2.0},
            {"step": 2, "split": "train", "loss": 1.5},
        ],
        "val": [
            {"step": 1, "split": "val", "loss": 2.2},
            {"step": 2, "split": "val", "loss": 1.8},
        ],
    }
    curve_path = tmp_path / "loss_curve.json"
    curve_path.write_text(__import__("json").dumps(curve, ensure_ascii=False), encoding="utf-8")

    csv_path = tmp_path / "curve.csv"
    res = run_chat_module(
        "chatdemo.plot_loss",
        [
            "--loss-curve",
            str(curve_path),
            "--output-csv",
            str(csv_path),
        ],
    )
    assert res.returncode == 0, res.stderr
    assert csv_path.exists()

    rows = csv_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("split,step,loss,ppl")
    assert len(rows) == 5
