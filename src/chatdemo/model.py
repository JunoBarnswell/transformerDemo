from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


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


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, device=device) * float("-inf"), diagonal=1)


class TinyTransformerLM(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        key_padding_mask = input_ids.eq(self.cfg.pad_id)
        attn_mask = _causal_mask(seq_len, device)

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        return logits

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
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.size(1) >= self.cfg.max_seq_len:
                break

            logits = self(generated)[:, -1, :] / max(temperature, 1e-6)
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

            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)

            if (next_id == eos_id).all():
                break

        return generated
