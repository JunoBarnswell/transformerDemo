"""ConversationMemory — rolling-window + permanent-summary memory for multi-turn chat.

Architecture
------------
permanent_summary : str
    Grows monotonically.  Every time ``compact()`` is called the oldest
    half of *recent_turns* is condensed into one sentence and appended here.
    This text is *always* prepended to the context sent to the model, so the
    very first turn is never truly forgotten — it lives in the summary.

recent_turns : collections.deque
    Keeps the last ``recent_window`` (user, assistant) string pairs verbatim.
    When the deque fills up ``compact()`` is called automatically.

Usage
-----
    memory = ConversationMemory(recent_window=8, compact_threshold=8)
    context = memory.build_context_turns()          # list[str] → pass to generate_reply
    res = generate_reply(checkpoint, prompt, context_turns=context, ...)
    memory.push(user_msg, res["response"])
    if memory.should_compact():
        memory.compact(model, tokenizer)
"""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .infer import generate_text  # noqa: F401 — only for type hints
    from .model import TinyTransformerLM
    from .tokenizer import CharTokenizer


class ConversationMemory:
    """Dual-layer conversation memory that never forgets the first turn."""

    def __init__(
        self,
        recent_window: int = 8,
        compact_threshold: int = 8,
        summary_max_tokens: int = 40,
    ) -> None:
        self.recent_window = recent_window
        self.compact_threshold = compact_threshold
        self.summary_max_tokens = summary_max_tokens

        self.permanent_summary: str = ""
        # Each element is a plain string like "user: ...\nassistant: ..."
        self.recent_turns: deque[str] = deque(maxlen=recent_window)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push(self, user_msg: str, assistant_reply: str) -> None:
        """Record one exchange into recent utterances."""
        if user_msg:
            self.recent_turns.append(user_msg.strip())
        if assistant_reply:
            self.recent_turns.append(assistant_reply.strip())

    def should_compact(self) -> bool:
        """True when enough turns have accumulated to warrant a compaction."""
        return len(self.recent_turns) >= self.compact_threshold

    def compact(
        self,
        model: "TinyTransformerLM",
        tokenizer: "CharTokenizer",
        temperature: float = 0.3,
        top_k: int = 5,
    ) -> str:
        """Summarise the oldest half of *recent_turns* and fold it into
        *permanent_summary*.  Returns the generated summary sentence."""
        from .infer import generate_text  # local import to avoid circular

        half = max(1, len(self.recent_turns) // 2)
        turns_to_summarise = [self.recent_turns[i] for i in range(half)]
        dialogue_text = "\n".join(turns_to_summarise)
        prompt = f"请用一句话总结以下对话内容：\n{dialogue_text}\n总结："

        result = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=self.summary_max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        summary_sentence = result["response"].strip()

        if summary_sentence:
            sep = "\n" if self.permanent_summary else ""
            self.permanent_summary += sep + summary_sentence

        # Remove the summarised turns from the deque
        for _ in range(half):
            if self.recent_turns:
                self.recent_turns.popleft()

        return summary_sentence

    def build_context_turns(self) -> List[str]:
        """Return the list of context strings to pass as *context_turns* to
        ``generate_reply`` / ``generate_text``.

        Format:
            [permanent_summary (if non-empty)] + recent turns (verbatim strings)
        """
        turns: List[str] = []
        if self.permanent_summary:
            turns.append(f"[历史摘要] {self.permanent_summary}")
        turns.extend(self.recent_turns)
        return turns

    # ------------------------------------------------------------------
    # Persistence helpers (optional)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "permanent_summary": self.permanent_summary,
            "recent_turns": list(self.recent_turns),
            "recent_window": self.recent_window,
            "compact_threshold": self.compact_threshold,
            "summary_max_tokens": self.summary_max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        mem = cls(
            recent_window=data.get("recent_window", 8),
            compact_threshold=data.get("compact_threshold", 8),
            summary_max_tokens=data.get("summary_max_tokens", 40),
        )
        mem.permanent_summary = data.get("permanent_summary", "")
        for turn in data.get("recent_turns", []):
            mem.recent_turns.append(turn)
        return mem

    def reset(self) -> None:
        """Clear all memory (start a new session)."""
        self.permanent_summary = ""
        self.recent_turns.clear()

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={len(self.recent_turns)}/{self.recent_window}, "
            f"summary_len={len(self.permanent_summary)})"
        )
