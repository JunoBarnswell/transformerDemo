from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = str(REPO_ROOT / "src")


def run_chat_module(module: str, args: List[str], cwd: Path | None = None):
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_PATH
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=str(cwd or REPO_ROOT),
        text=True,
        capture_output=True,
        env=env,
    )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
