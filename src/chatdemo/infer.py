from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

from .data import encode_chat_prompt
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
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


def _split_thinking(
    response_ids: List[int],
    tokenizer: CharTokenizer,
) -> tuple[str, str]:
    """Split generated ids into (thinking_text, response_text).

    If the output contains <think>…</think>, the content inside is the
    thinking chain and the content after </think> is the final reply.
    If only <think> is present (e.g. stopped mid-thinking), it is treated as thinking.
    Otherwise thinking_text is empty and response_text is the full output.
    """
    think_id = tokenizer.think_id
    end_think_id = tokenizer.end_think_id

    if think_id in response_ids and end_think_id in response_ids:
        t_start = response_ids.index(think_id)
        t_end = response_ids.index(end_think_id)
        if t_start < t_end:
            thinking_ids = response_ids[t_start + 1 : t_end]
            reply_ids = response_ids[t_end + 1 :]
            thinking_text = tokenizer.decode(thinking_ids)
            response_text = tokenizer.decode(reply_ids)
            return thinking_text, response_text
    elif think_id in response_ids:
        t_start = response_ids.index(think_id)
        return tokenizer.decode(response_ids[t_start + 1 :]), ""

    # No thinking block — whole output is the reply
    return "", tokenizer.decode(response_ids)


def generate_text(
    model: TinyTransformerLM,
    tokenizer: CharTokenizer,
    prompt: str,
    context_turns: Sequence[str] = (),
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Dict[str, Any]:
    prompt_ids = encode_chat_prompt(prompt, tokenizer, context_turns=context_turns)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    # RoPE has no hard length limit — no forced truncation needed.
    # We still guard against pathologically long prompts.
    if input_ids.size(1) >= model.cfg.max_seq_len:
        input_ids = input_ids[:, -model.cfg.max_seq_len:]

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

    # Hard stop at earliest EOS or SEP (before think-splitting)
    stop_indices = [
        response_ids.index(stop_id)
        for stop_id in (tokenizer.eos_id, tokenizer.sep_id)
        if stop_id in response_ids
    ]
    if stop_indices:
        response_ids = response_ids[: min(stop_indices)]

    # Split <think>…</think> from final reply
    thinking_text, response_text = _split_thinking(response_ids, tokenizer)

    return {
        "prompt": prompt,
        "response": response_text,
        "thinking": thinking_text,
        "response_ids": response_ids,
        "steps": len(response_ids),
        "elapsed_sec": elapsed,
    }


def generate_reply(
    checkpoint: str,
    prompt: str,
    context_turns: Sequence[str] = (),
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> dict:
    model, tokenizer, _ = load_checkpoint_bundle(checkpoint)
    return generate_text(
        model,
        tokenizer,
        prompt,
        context_turns=context_turns,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


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
