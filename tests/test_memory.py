"""Tests for ConversationMemory — unit-level, no model needed."""
from __future__ import annotations

from collections import deque

import pytest

from chatdemo.memory import ConversationMemory


def _make_mem(recent_window: int = 4, compact_threshold: int = 4) -> ConversationMemory:
    return ConversationMemory(recent_window=recent_window, compact_threshold=compact_threshold)


class TestPush:
    def test_push_single_turn(self):
        mem = _make_mem()
        mem.push("你好", "你好啊")
        assert len(mem.recent_turns) == 2
        assert list(mem.recent_turns)[0] == "你好"
        assert list(mem.recent_turns)[1] == "你好啊"

    def test_push_respects_maxlen(self):
        mem = _make_mem(recent_window=4, compact_threshold=10)
        for i in range(5):
            mem.push(f"问{i}", f"答{i}")
        # deque maxlen=4, so only last 4 utterances remain
        assert len(mem.recent_turns) == 4
        assert list(mem.recent_turns)[-1] == "答4"


class TestShouldCompact:
    def test_not_triggered_below_threshold(self):
        mem = _make_mem(compact_threshold=6)
        mem.push("a", "b")
        mem.push("c", "d")
        assert not mem.should_compact()

    def test_triggered_at_threshold(self):
        mem = _make_mem(compact_threshold=4)
        mem.push("a", "b")
        mem.push("c", "d")
        assert mem.should_compact()


class TestBuildContextTurns:
    def test_empty_memory_returns_empty(self):
        mem = _make_mem()
        assert mem.build_context_turns() == []

    def test_only_recent_turns_when_no_summary(self):
        mem = _make_mem()
        mem.push("你好", "你好")
        mem.push("再见", "再见")
        turns = mem.build_context_turns()
        assert len(turns) == 4
        assert not any("[历史摘要]" in t for t in turns)

    def test_permanent_summary_prepended(self):
        mem = _make_mem()
        mem.permanent_summary = "用户叫张三，喜欢科幻"
        mem.push("今天天气好", "是的")
        turns = mem.build_context_turns()
        assert turns[0].startswith("[历史摘要]")
        assert "张三" in turns[0]
        assert len(turns) == 3  # summary + 2 utterances

    def test_summary_always_first(self):
        mem = _make_mem()
        mem.permanent_summary = "早期摘要"
        for i in range(3):
            mem.push(f"问{i}", f"答{i}")
        turns = mem.build_context_turns()
        assert "[历史摘要]" in turns[0]


class TestPersistence:
    def test_round_trip(self):
        mem = _make_mem(recent_window=6, compact_threshold=6)
        mem.permanent_summary = "历史摘要内容"
        mem.push("问题", "回答")

        data = mem.to_dict()
        mem2 = ConversationMemory.from_dict(data)

        assert mem2.permanent_summary == mem.permanent_summary
        assert list(mem2.recent_turns) == list(mem.recent_turns)
        assert mem2.recent_window == mem.recent_window
        assert mem2.compact_threshold == mem.compact_threshold

    def test_reset_clears_all(self):
        mem = _make_mem()
        mem.permanent_summary = "some summary"
        mem.push("a", "b")
        mem.reset()
        assert mem.permanent_summary == ""
        assert len(mem.recent_turns) == 0
