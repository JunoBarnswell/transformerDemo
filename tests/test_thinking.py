"""Tests for Thinking Mode — tokenizer special tokens and generate_text splitting."""
from __future__ import annotations

from pathlib import Path

import pytest

from chatdemo.tokenizer import CharTokenizer, SPECIAL_TOKENS
from chatdemo.infer import _split_thinking


# ---------------------------------------------------------------------------
# Tokenizer: think / end_think tokens
# ---------------------------------------------------------------------------

class TestThinkTokens:
    def _tok(self) -> CharTokenizer:
        return CharTokenizer.build_from_texts(["你好世界"], vocab_size=64)

    def test_think_token_in_special_tokens(self):
        assert "<think>" in SPECIAL_TOKENS
        assert "</think>" in SPECIAL_TOKENS

    def test_think_id_accessible(self):
        tok = self._tok()
        assert isinstance(tok.think_id, int)
        assert isinstance(tok.end_think_id, int)
        assert tok.think_id != tok.end_think_id

    def test_think_ids_are_deterministic(self):
        tok = self._tok()
        assert tok.think_id == tok.token_to_id["<think>"]
        assert tok.end_think_id == tok.token_to_id["</think>"]

    def test_think_id_after_sep(self):
        """<think> must come after <sep> in the special-token list."""
        tok = self._tok()
        assert tok.think_id > tok.sep_id

    def test_save_and_load_preserves_think_tokens(self, tmp_path: Path):
        tok = self._tok()
        tok.save(str(tmp_path / "vocab.json"))
        tok2 = CharTokenizer.load(str(tmp_path / "vocab.json"))
        assert tok2.think_id == tok.think_id
        assert tok2.end_think_id == tok.end_think_id

    def test_decode_strips_think_markers_by_default(self):
        tok = self._tok()
        ids = [tok.think_id, *tok.encode("你好", add_special_tokens=False), tok.end_think_id]
        decoded = tok.decode(ids)
        assert "<think>" not in decoded
        assert "</think>" not in decoded
        assert "你好" in decoded

    def test_decode_with_special_preserves_think_markers(self):
        tok = self._tok()
        ids = [tok.think_id, *tok.encode("你好", add_special_tokens=False), tok.end_think_id]
        decoded = tok.decode_with_special(ids)
        assert "<think>" in decoded
        assert "</think>" in decoded


# ---------------------------------------------------------------------------
# _split_thinking helper
# ---------------------------------------------------------------------------

class TestSplitThinking:
    def _tok(self) -> CharTokenizer:
        return CharTokenizer.build_from_texts(["好回复推理"], vocab_size=64)

    def test_no_think_block_returns_empty_thinking(self):
        tok = self._tok()
        reply_ids = tok.encode("好", add_special_tokens=False)
        thinking, response = _split_thinking(reply_ids, tok)
        assert thinking == ""
        assert response == tok.decode(reply_ids)

    def test_think_block_is_split_correctly(self):
        tok = self._tok()
        think_content = tok.encode("推理", add_special_tokens=False)
        reply_content = tok.encode("回复", add_special_tokens=False)
        ids = [tok.think_id] + think_content + [tok.end_think_id] + reply_content
        thinking, response = _split_thinking(ids, tok)
        assert "推理" in thinking
        assert "回复" in response
        assert "<think>" not in thinking
        assert "<think>" not in response

    def test_empty_think_block(self):
        tok = self._tok()
        reply_content = tok.encode("好", add_special_tokens=False)
        ids = [tok.think_id, tok.end_think_id] + reply_content
        thinking, response = _split_thinking(ids, tok)
        assert thinking == ""
        assert "好" in response

    def test_think_after_end_think_ignored(self):
        """If end_think comes before think (malformed), treat as no thinking."""
        tok = self._tok()
        ids = [tok.end_think_id, tok.think_id, *tok.encode("好", add_special_tokens=False)]
        thinking, response = _split_thinking(ids, tok)
        # end_think_id index (0) < think_id index (1), condition t_start < t_end fails
        assert thinking == ""
