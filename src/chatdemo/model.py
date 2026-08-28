from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 64
    nhead: int = 2
    n_layers: int = 2
    ff_dim: int = 128
    dropout: float = 0.1
    pad_id: int = 0


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# No learnable parameters — works for any sequence length at inference time.
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute sin/cos tables for RoPE up to seq_len."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    half = head_dim // 2
    # Frequencies: theta_i = 1 / (10000 ^ (2i / d))
    theta = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    # Positions
    positions = torch.arange(seq_len, device=device).float()
    # Outer product → [seq_len, half]
    freqs = torch.outer(positions, theta)
    # Expand to [seq_len, head_dim] by repeating: [cos0, cos0, cos1, cos1, ...]
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate x by 90°: split in half, negate second half, swap."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape [B, H, T, head_dim]."""
    # cos/sin: [T, head_dim] → [1, 1, T, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with RoPE
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                       # [B, T, d_model]
        cos: torch.Tensor,                     # [T, head_dim]
        sin: torch.Tensor,                     # [T, head_dim]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool, True = pad
    ) -> torch.Tensor:
        B, T, _ = x.shape
        H, hd = self.nhead, self.head_dim

        def _split(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]

        q, k, v = _split(self.q_proj), _split(self.k_proj), _split(self.v_proj)

        # Apply RoPE to queries and keys
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Scaled dot-product — causal mask via is_causal=True (requires PyTorch ≥ 2.0)
        # We build the mask manually for compatibility.
        scale = 1.0 / math.sqrt(hd)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Causal mask: upper triangle = -inf
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        attn = attn + causal_mask

        # Padding mask: True positions get -inf
        # Use torch.where to avoid 0.0 * -inf = NaN (IEEE 754 undefined)
        if key_padding_mask is not None:
            # [B, T] → [B, 1, 1, T]  broadcast across heads and query positions
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN that arises when every key in a row is -inf (e.g. a query
        # position that can only attend to itself but that position is also masked).
        # This is benign: the zero attention weight means the row contributes nothing.
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)             # [B, H, T, hd]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Feed-Forward Network (SwiGLU-style or standard GELU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Single Transformer Block (Pre-LayerNorm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.nhead, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg.d_model, cfg.ff_dim, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin, key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# TinyTransformerLM  — decoder-only language model with RoPE
# ---------------------------------------------------------------------------

class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.d_model % cfg.nhead == 0, "d_model must be divisible by nhead"
        self.head_dim = cfg.d_model // cfg.nhead

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding with lm_head
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass. input_ids: [B, T] → logits: [B, T, vocab_size]"""
        B, T = input_ids.shape
        device = input_ids.device

        # Build RoPE cache dynamically for length T (no stored parameters)
        cos, sin = _build_rope_cache(T, self.head_dim, device)

        x = self.token_emb(input_ids)           # [B, T, d_model]
        x = self.drop(x)

        key_padding_mask = input_ids.eq(self.cfg.pad_id)  # [B, T]

        for block in self.blocks:
            x = block(x, cos, sin, key_padding_mask)

        x = self.ln_f(x)
        return self.lm_head(x)                  # [B, T, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 40,
        eos_id: int = 3,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Auto-regressive generation. RoPE supports arbitrary sequence length."""
        self.eval()
        generated = input_ids

        for _ in range(max_new_tokens):
            # RoPE has no hard length limit — run on full generated sequence
            logits = self(generated)[:, -1, :] / max(temperature, 1e-6)
            # Guard against NaN/Inf from degenerate states
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            probs = torch.softmax(logits, dim=-1)

            if top_k > 0:
                k = min(top_k, probs.size(-1))
                topk_values, topk_indices = torch.topk(probs, k=k, dim=-1)
                filtered = torch.zeros_like(probs)
                filtered.scatter_(1, topk_indices, topk_values)
                probs = filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                keep = cdf <= top_p
                keep[:, 0] = True
                trimmed_sorted = sorted_probs * keep.to(sorted_probs.dtype)
                probs = torch.zeros_like(probs)
                probs.scatter_(1, sorted_idx, trimmed_sorted)
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Final safety: ensure valid distribution before sampling
            probs = probs.clamp(min=0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True)
            if (prob_sum == 0).any():
                probs = torch.ones_like(probs) / probs.size(-1)
            else:
                probs = probs / prob_sum

            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if (next_id == eos_id).all():
                break

        return generated
