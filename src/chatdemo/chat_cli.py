from __future__ import annotations

import argparse

from .infer import generate_reply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with tiny transformer model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def chat_once(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    return generate_reply(
        checkpoint=checkpoint,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )["response"]


def main() -> None:
    args = parse_args()

    if args.prompt is not None:
        print(chat_once(
            checkpoint=args.checkpoint,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        ))
        return

    print("Tiny Transformer Chat. 输入 quit 或 exit 退出。")
    while True:
        user = input("你: ")
        if user.strip().lower() in {"quit", "exit"}:
            break
        reply = chat_once(
            checkpoint=args.checkpoint,
            prompt=user,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"模型: {reply}")


if __name__ == "__main__":
    main()
