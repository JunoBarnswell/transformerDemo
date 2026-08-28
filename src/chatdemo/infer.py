from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch

from .model import TinyTransformerLM, TransformerConfig
from .tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate response from a trained tiny chat model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--return-json", action="store_true")
    return parser.parse_args()


def load_checkpoint_bundle(checkpoint: str):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg: Dict[str, Any] = payload["config"]
    token = CharTokenizer(token_to_id=payload["token_to_id"], id_to_token=payload["id_to_token"])

    mcfg = cfg["model"]
    tcfg = TransformerConfig(
        vocab_size=len(token.token_to_id),
        max_seq_len=cfg["data"]["seq_len"],
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        n_layers=mcfg["n_layers"],
        ff_dim=mcfg["ff_dim"],
        dropout=mcfg["dropout"],
        pad_id=token.pad_id,
    )
    model = TinyTransformerLM(tcfg)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, token, cfg


def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    input_ids = torch.tensor([[tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)
    if input_ids.size(1) >= model.cfg.max_seq_len:
        input_ids = input_ids[:, -model.cfg.max_seq_len :]

    start = time.time()
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    elapsed = time.time() - start

    generated_ids = output[0].tolist()
    prompt_len = input_ids.size(1)
    response_ids = generated_ids[prompt_len:]
    response_text = tokenizer.decode(response_ids)

    return {
        "prompt": prompt,
        "response": response_text,
        "response_ids": response_ids,
        "steps": len(response_ids),
        "elapsed_sec": elapsed,
    }


def generate_reply(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict:
    model, tokenizer, _ = load_checkpoint_bundle(checkpoint)
    return generate_text(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p)



def main() -> None:
    args = parse_args()
    result = generate_reply(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    if args.return_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["response"])


if __name__ == "__main__":
    main()
