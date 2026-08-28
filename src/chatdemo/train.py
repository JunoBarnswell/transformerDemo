from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import (
    ChatDataset,
    build_vocab_if_missing,
    collate_chat_batch,
    load_pairs,
    select_eval_prompts,
)
from .infer import generate_text
from .model import TinyTransformerLM, TransformerConfig
from .tokenizer import CharTokenizer


def set_seed(seed: int) -> None:
    import random

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
    parser.add_argument("--early-stop-patience", type=int, dest="early_stop_patience")
    parser.add_argument("--max-context-turns", type=int, dest="max_context_turns")
    parser.add_argument("--train-log-file", dest="train_log_file")
    parser.add_argument("--loss-curve-file", dest="loss_curve_file")
    parser.add_argument("--inference-log-file", dest="inference_log_file")
    parser.add_argument("--inference-prompts-file", dest="inference_prompts_file")
    parser.add_argument("--num-infer-prompts", type=int, dest="num_infer_prompts")
    parser.add_argument("--d-model", type=int, dest="d_model")
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--n-layers", type=int, dest="n_layers")
    parser.add_argument("--ff-dim", type=int, dest="ff_dim")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", dest="output_dir")
    parser.add_argument("--num-workers", type=int, dest="num_workers", default=0)
    return parser.parse_args()


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_prompt_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value.strip()]
    return []


def _read_prompt_file(path: str) -> List[str]:
    prompts: List[str] = []
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "context" in obj:
                prompts.append(str(obj["context"]).strip())
            elif isinstance(obj, str):
                prompts.append(obj.strip())
            else:
                prompts.append(stripped)
        except json.JSONDecodeError:
            prompts.append(stripped)
    return [p for p in prompts if p]


def _resolve_train_log_paths(output_dir: Path, train_cfg: Dict[str, Any]) -> Dict[str, Path]:
    return {
        "train": output_dir / str(train_cfg.get("train_log_file", "train_log.jsonl")),
        "loss_curve": output_dir / str(train_cfg.get("loss_curve_file", "loss_curve.json")),
        "inference": output_dir / str(train_cfg.get("inference_log_file", "inference_log.jsonl")),
    }


def _resolve_inference_prompts(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    default_pairs: List[Any],
) -> List[str]:
    prompts = _load_prompt_list(data_cfg.get("inference_prompts"))

    prompt_file = data_cfg.get("inference_prompts_file") or train_cfg.get("inference_prompts_file")
    if not prompt_file and train_cfg.get("inference_prompts_file"):
        prompt_file = train_cfg.get("inference_prompts_file")

    if isinstance(prompt_file, str) and prompt_file.strip():
        prompts.extend(_read_prompt_file(prompt_file))

    if not prompts:
        prompts.extend(select_eval_prompts(default_pairs, max_prompts=int(train_cfg.get("num_infer_prompts", 3) or 3)))
    else:
        # dedupe while keep order
        seen = set()
        unique_prompts: List[str] = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)
        if int(train_cfg.get("num_infer_prompts", 0) or 0) > 0:
            unique_prompts = unique_prompts[: int(train_cfg["num_infer_prompts"])]
        prompts = unique_prompts

    return prompts[: int(train_cfg.get("num_infer_prompts", 3) or 3)]


def _merge_config(base: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = dict(base)

    merged.setdefault("seed", 42)
    merged.setdefault("data", {})
    merged.setdefault("train", {})
    merged.setdefault("model", {})

    # scalar override
    if args.seed is not None:
        merged["seed"] = args.seed

    overrides = {
        "data": {
            "data_file": args.data_file,
            "vocab_path": args.vocab_path,
            "vocab_size": args.vocab_size,
            "seq_len": args.seq_len,
            "max_context_turns": args.max_context_turns,
            "inference_prompts": merged.get("data", {}).get("inference_prompts"),
            "inference_prompts_file": merged.get("data", {}).get("inference_prompts_file") or args.inference_prompts_file,
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
                "early_stop_patience": args.early_stop_patience,
                "output_dir": args.output_dir,
                "num_workers": args.num_workers,
                "train_log_file": args.train_log_file,
                "loss_curve_file": args.loss_curve_file,
                "inference_log_file": args.inference_log_file,
                "inference_prompts_file": args.inference_prompts_file,
                "num_infer_prompts": args.num_infer_prompts,
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


def _generation_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gen = cfg.get("generation", {})
    return {
        "max_new_tokens": int(gen.get("max_new_tokens", 24)),
        "temperature": float(gen.get("temperature", 1.0)),
        "top_k": int(gen.get("top_k", 0)),
        "top_p": float(gen.get("top_p", 1.0)),
    }


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    seed = cfg["seed"]
    set_seed(seed)

    device = torch.device("cpu")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]

    data_file = data_cfg["data_file"]
    pairs = load_pairs(data_file, max_context_turns=int(data_cfg.get("max_context_turns", 4)))

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
    early_stop_patience = int(train_cfg.get("early_stop_patience", 10) or 10)
    if early_stop_patience < 1:
        early_stop_patience = 10

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_paths = _resolve_train_log_paths(output_dir, train_cfg)
    for p in log_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            p.unlink()

    gen_cfg = _generation_settings(cfg)
    inference_prompts = _resolve_inference_prompts(data_cfg, train_cfg, pairs)

    step = 0
    best_loss = float("inf")
    final_ckpt = output_dir / "checkpoint.pt"
    no_improve_steps = 0
    should_stop = False

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
            ppl = torch.exp(loss.detach().cpu()).item()
            _append_jsonl(log_paths["train"], {"step": step, "split": "train", "loss": float(loss.item()), "ppl": float(ppl)})

            if step % log_interval == 0:
                print(f"step={step} loss={loss.item():.4f} ppl={ppl:.2f}")

            if eval_interval > 0 and step % eval_interval == 0:
                val_loss = evaluate(model, loader, criterion, device)
                _append_jsonl(log_paths["train"], {"step": step, "split": "val", "loss": float(val_loss)})
                print(f"eval step={step} val_loss={val_loss:.4f}")

                improved = False
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improve_steps = 0
                    improved = True
                    save_checkpoint(model, tokenizer, cfg, step, output_dir / "best.pt")

                if not improved:
                    no_improve_steps += 1

                if no_improve_steps >= early_stop_patience:
                    _append_jsonl(
                        log_paths["train"],
                        {
                            "step": step,
                            "split": "early_stop",
                            "patience": early_stop_patience,
                            "no_improve_steps": no_improve_steps,
                            "best_loss": best_loss if best_loss < float("inf") else None,
                        },
                    )
                    should_stop = True
                    break

                for prompt in inference_prompts:
                    gen = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=gen_cfg["max_new_tokens"],
                        temperature=gen_cfg["temperature"],
                        top_k=gen_cfg["top_k"],
                        top_p=gen_cfg["top_p"],
                    )
                    _append_jsonl(
                        log_paths["inference"],
                        {
                            "step": step,
                            "type": "eval_sample",
                            **gen,
                        },
                    )

            if save_interval > 0 and step % save_interval == 0:
                save_checkpoint(model, tokenizer, cfg, step, output_dir / f"checkpoint_step_{step}.pt")

            if step >= max_steps:
                break
            if should_stop:
                break

        if step >= max_steps or should_stop:
            break

    save_checkpoint(model, tokenizer, cfg, step, final_ckpt)
    print(f"training finished. checkpoint={final_ckpt}")

    # rebuild train log file into one JSON curve for easy plotting
    # (the raw jsonl is enough for streaming, this summary file is for convenience)
    train_curve = []
    val_curve = []
    for line in log_paths["train"].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("split") == "train":
            train_curve.append(payload)
        elif payload.get("split") == "val":
            val_curve.append(payload)

    with log_paths["loss_curve"].open("w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_curve,
                "val": val_curve,
                "best_loss": best_loss if best_loss < float("inf") else None,
                "steps": step,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "checkpoint": str(final_ckpt),
        "steps": step,
        "best_loss": best_loss if best_loss < float("inf") else None,
        "train_log": str(log_paths["train"]),
        "loss_curve": str(log_paths["loss_curve"]),
        "inference_log": str(log_paths["inference"]),
    }


def main() -> None:
    args = parse_args()
    raw_cfg = load_yaml(args.config)

    # Fill missing sections for deterministic behavior when config is incomplete.
    raw_cfg.setdefault("seed", 42)
    raw_cfg.setdefault("data", {})
    raw_cfg.setdefault("train", {})
    raw_cfg.setdefault("model", {})
    raw_cfg.setdefault("generation", {})

    # keep compatibility: allow max_context_turns in root train/data sections
    cfg = _merge_config(raw_cfg, args)

    result = train(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
