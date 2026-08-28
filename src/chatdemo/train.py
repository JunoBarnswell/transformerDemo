from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import ChatDataset, build_vocab_if_missing, collate_chat_batch, load_pairs
from .model import TinyTransformerLM, TransformerConfig
from .tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny transformer from scratch")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data-file", dest="data_file")
    parser.add_argument("--vocab-path", dest="vocab_path")
    parser.add_argument("--vocab-size", type=int, dest="vocab_size")
    parser.add_argument("--seq-len", type=int, dest="seq_len")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--max-steps", type=int, dest="max_steps")
    parser.add_argument("--max-epochs", type=int, dest="max_epochs")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--grad-clip", type=float, dest="grad_clip")
    parser.add_argument("--eval-interval", type=int, dest="eval_interval")
    parser.add_argument("--save-interval", type=int, dest="save_interval")
    parser.add_argument("--log-interval", type=int, dest="log_interval")
    parser.add_argument("--d-model", type=int, dest="d_model")
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--n-layers", type=int, dest="n_layers")
    parser.add_argument("--ff-dim", type=int, dest="ff_dim")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--num-workers", type=int, dest="num_workers", default=0)
    return parser.parse_args()


def _merge_config(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(base)

    # Ensure base sections exist.
    merged.setdefault("data", {})
    merged.setdefault("train", {})
    merged.setdefault("model", {})

    # scalar override
    if args.seed is not None:
        merged["seed"] = args.seed

    # dict overrides
    overrides = {
        "data": {
            "data_file": args.data_file,
            "vocab_path": args.vocab_path,
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
        },
        "train": {
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "grad_clip": args.grad_clip,
            "eval_interval": args.eval_interval,
            "save_interval": args.save_interval,
            "log_interval": args.log_interval,
            "output_dir": args.output_dir,
            "num_workers": args.num_workers,
        },
        "model": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "n_layers": args.n_layers,
            "ff_dim": args.ff_dim,
            "dropout": args.dropout,
        },
    }

    for section, values in overrides.items():
        if section not in merged or merged.get(section) is None or not isinstance(merged.get(section), dict):
            merged[section] = {}
        for k, v in values.items():
            if v is not None:
                merged[section][k] = v

    if not isinstance(merged.get("train", None), dict):
        merged["train"] = {}
    if not merged["train"].get("output_dir"):
        merged["train"]["output_dir"] = merged.get("output_dir", "outputs/run")

    return merged


def evaluate(model: TinyTransformerLM, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    num_tokens = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            mask = y.reshape(-1) != model.cfg.pad_id
            total_loss += float(loss.item() * mask.sum().item())
            num_tokens += int(mask.sum().item())

    if num_tokens == 0:
        return 0.0
    return total_loss / max(1, num_tokens)


def save_checkpoint(model: TinyTransformerLM, tokenizer: CharTokenizer, cfg: Dict[str, Any], step: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "token_to_id": tokenizer.token_to_id,
            "id_to_token": tokenizer.id_to_token,
            "config": cfg,
        },
        path,
    )


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = cfg["seed"]
    set_seed(seed)

    device = torch.device("cpu")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    data_file = data_cfg["data_file"]
    pairs = load_pairs(data_file)

    vocab_path = data_cfg["vocab_path"]
    tokenizer = build_vocab_if_missing(
        vocab_path=vocab_path,
        pairs=pairs,
        vocab_size=data_cfg["vocab_size"],
    )

    seq_len = int(data_cfg["seq_len"])
    dataset = ChatDataset(pairs=pairs, tokenizer=tokenizer, max_length=seq_len)

    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
        collate_fn=lambda batch: collate_chat_batch(batch, pad_id=tokenizer.pad_id),
    )

    tcfg = TransformerConfig(
        vocab_size=len(tokenizer.token_to_id),
        max_seq_len=seq_len,
        d_model=int(model_cfg["d_model"]),
        nhead=int(model_cfg["nhead"]),
        n_layers=int(model_cfg["n_layers"]),
        ff_dim=int(model_cfg["ff_dim"]),
        dropout=float(model_cfg["dropout"]),
        pad_id=tokenizer.pad_id,
    )
    model = TinyTransformerLM(tcfg).to(device)

    opt = AdamW(model.parameters(), lr=float(train_cfg["learning_rate"]))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    max_steps = int(train_cfg["max_steps"])
    log_interval = int(train_cfg["log_interval"])
    eval_interval = int(train_cfg["eval_interval"])
    save_interval = int(train_cfg["save_interval"])
    grad_clip = float(train_cfg["grad_clip"])

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    best_loss = float("inf")
    final_ckpt = output_dir / "checkpoint.pt"

    for epoch in range(int(train_cfg["max_epochs"])):
        model.train()
        for x, y in loader:
            if step >= max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            step += 1

            if step % log_interval == 0:
                ppl = torch.exp(loss.detach().cpu()).item()
                print(f"step={step} loss={loss.item():.4f} ppl={ppl:.2f}")

            if eval_interval > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, loader, criterion, device)
                print(f"eval step={step} val_loss={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(model, tokenizer, cfg, step, output_dir / "best.pt")

            if save_interval > 0 and step % save_interval == 0:
                save_checkpoint(model, tokenizer, cfg, step, output_dir / f"checkpoint_step_{step}.pt")

            if step >= max_steps:
                break

        if step >= max_steps:
            break

    save_checkpoint(model, tokenizer, cfg, step, final_ckpt)
    print(f"training finished. checkpoint={final_ckpt}")

    return {
        "checkpoint": str(final_ckpt),
        "steps": step,
        "best_loss": best_loss if best_loss < float("inf") else None,
    }


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)

    # Fill missing sections for deterministic behavior when config file不完整。
    raw_cfg.setdefault("seed", 42)
    raw_cfg.setdefault("data", {})
    raw_cfg.setdefault("train", {})
    raw_cfg.setdefault("model", {})

    cfg = _merge_config(raw_cfg, args)

    result = train(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
