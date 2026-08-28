from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .infer import generate_text, load_checkpoint_bundle
from .memory import ConversationMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with a tiny Transformer model (supports persistent memory & thinking mode)"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--prompt", default=None, help="Single-shot prompt (non-interactive)")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Print <think>…</think> reasoning chain when present")
    parser.add_argument("--recent-window", type=int, default=8,
                        help="Number of recent turns to keep verbatim in memory")
    parser.add_argument("--compact-threshold", type=int, default=8,
                        help="Trigger memory compaction after this many recent turns")
    parser.add_argument("--memory-file", default=None,
                        help="JSON file to persist/resume memory across sessions")
    return parser.parse_args()


def chat_once(
    checkpoint: str,
    prompt: str,
    context_turns: tuple = (),
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    """Single-shot generation helper (used by tests and programmatic callers)."""
    from .infer import generate_reply
    return generate_reply(
        checkpoint=checkpoint,
        prompt=prompt,
        context_turns=context_turns,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )["response"]


def main() -> None:
    args = parse_args()

    # Load model once for the whole session
    model, tokenizer, cfg = load_checkpoint_bundle(args.checkpoint)
    gen_cfg = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    # Restore or create memory
    mem_file = Path(args.memory_file) if args.memory_file else None
    if mem_file and mem_file.exists():
        memory = ConversationMemory.from_dict(
            json.loads(mem_file.read_text(encoding="utf-8"))
        )
        print(f"[记忆已恢复: {memory}]")
    else:
        memory = ConversationMemory(
            recent_window=args.recent_window,
            compact_threshold=args.compact_threshold,
        )

    def _reply(user_msg: str) -> dict:
        context_turns = memory.build_context_turns()
        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=user_msg,
            context_turns=context_turns,
            **gen_cfg,
        )
        return result

    def _after_reply(user_msg: str, result: dict) -> None:
        memory.push(user_msg, result["response"])
        if memory.should_compact():
            print("[记忆压缩中…]")
            summary = memory.compact(model, tokenizer)
            print(f"[摘要生成] {summary}")
        if mem_file:
            mem_file.write_text(
                json.dumps(memory.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    # ----- Single-shot mode -----
    if args.prompt is not None:
        result = _reply(args.prompt)
        if args.show_thinking and result.get("thinking"):
            print(f"[思考过程]\n{result['thinking']}\n")
        print(result["response"])
        _after_reply(args.prompt, result)
        return

    # ----- Interactive loop -----
    print("Tiny Transformer Chat  —  输入 quit / exit 退出，/reset 清空记忆，/memory 查看记忆状态")
    while True:
        try:
            user_msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_msg:
            continue
        if user_msg.lower() in {"quit", "exit"}:
            print("再见！")
            break
        if user_msg == "/reset":
            memory.reset()
            print("[记忆已清空]")
            continue
        if user_msg == "/memory":
            print(f"[永久摘要] {memory.permanent_summary or '(空)'}")
            print(f"[近端轮次] {len(memory.recent_turns)} / {memory.recent_window} 轮")
            continue

        result = _reply(user_msg)
        if args.show_thinking and result.get("thinking"):
            print(f"[思考过程] {result['thinking']}")
        print(f"助手: {result['response']}")
        _after_reply(user_msg, result)


if __name__ == "__main__":
    main()
