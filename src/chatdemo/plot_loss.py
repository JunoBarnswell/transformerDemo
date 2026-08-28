from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot or export training loss curves")
    parser.add_argument("--loss-curve", required=True, dest="loss_curve")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot", action="store_true", help="Generate a PNG plot with matplotlib if available")
    parser.add_argument("--output-plot", default=None)
    parser.add_argument("--title", default="Training Loss Curves")
    return parser.parse_args()


def _ensure_path(path: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_loss_curve(loss_curve_path: str) -> Dict[str, Any]:
    path = Path(loss_curve_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported loss curve format: {path}")
    return payload


def _normalize_series(payload: Dict[str, Any], split: str) -> List[Tuple[int, float]]:
    points: List[Tuple[int, float]] = []

    if split in payload and isinstance(payload[split], list):
        rows = payload[split]
    elif isinstance(payload.get("train_log"), list) and all(isinstance(x, dict) for x in payload["train_log"]):
        # compatible with a raw training log file
        rows = [x for x in payload["train_log"] if x.get("split") == split]
    else:
        rows = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        if "step" not in row or "loss" not in row:
            continue
        try:
            step = int(row["step"])
            loss = float(row["loss"])
        except (TypeError, ValueError):
            continue
        points.append((step, loss))

    points.sort(key=lambda item: item[0])
    return points


def export_loss_csv(payload: Dict[str, Any], output_csv: Path) -> Path:
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "step", "loss", "ppl"])

        for split in ("train", "val"):
            rows = payload.get(split, []) if isinstance(payload.get(split), list) else []
            if not rows and split == "train":
                # fallback for raw logs
                rows = [x for x in payload.get("train_log", []) if x.get("split") == "train"]
            if not rows and split == "val":
                rows = [x for x in payload.get("train_log", []) if x.get("split") == "val"]

            for row in rows:
                if not isinstance(row, dict):
                    continue
                if "step" not in row or "loss" not in row:
                    continue
                writer.writerow(
                    [
                        split,
                        row.get("step", ""),
                        row.get("loss", ""),
                        row.get("ppl", ""),
                    ]
                )
    return output_csv


def plot_curve(train_points: List[Tuple[int, float]], val_points: List[Tuple[int, float]], output_plot: Path, title: str = "Training Loss Curves") -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is not installed") from exc

    fig, ax = plt.subplots()
    if train_points:
        x_train, y_train = zip(*train_points)
        ax.plot(x_train, y_train, label="train")
    if val_points:
        x_val, y_val = zip(*val_points)
        ax.plot(x_val, y_val, label="val")

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(output_plot))
    plt.close(fig)
    return output_plot


def plot_loss_curve(
    loss_curve_path: str,
    output_csv: str | None = None,
    do_plot: bool = False,
    output_plot: str | None = None,
    title: str = "Training Loss Curves",
) -> Dict[str, str]:
    payload = _load_loss_curve(loss_curve_path)

    if output_csv is None:
        curve_path = Path(loss_curve_path)
        output_csv = str(curve_path.parent / f"{curve_path.stem}.csv")

    output_csv_path = _ensure_path(output_csv)
    csv_path = export_loss_csv(payload, output_csv_path)

    train_points = _normalize_series(payload, "train")
    val_points = _normalize_series(payload, "val")

    result: Dict[str, str] = {
        "curve_csv": str(csv_path),
    }

    if do_plot:
        if output_plot is None:
            stem_path = Path(loss_curve_path)
            output_plot = str(stem_path.with_suffix(".png"))
        try:
            plot_path = plot_curve(train_points, val_points, _ensure_path(output_plot), title=title)
            result["loss_plot"] = str(plot_path)
        except RuntimeError:
            # Keep script working even when plotting deps are missing.
            result["loss_plot"] = ""
            print("plot skipped because matplotlib is not available")

    return result


def main() -> None:
    args = parse_args()
    result = plot_loss_curve(
        loss_curve_path=args.loss_curve,
        output_csv=args.output_csv,
        do_plot=args.plot,
        output_plot=args.output_plot,
        title=args.title,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
