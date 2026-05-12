# -*- coding: utf-8 -*-
"""SASRec: Self-Attentive Sequential Recommendation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt_utils


class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation model.

    Encodes a padded item-ID sequence into a single user representation
    using stacked Transformer encoder layers with causal masking.

    Reference: Kang & McAuley, "Self-Attentive Sequential Recommendation",
    ICDM 2018.
    """

    def __init__(
        self,
        n_items: int,
        dim: int = 256,
        n_heads: int = 4,
        n_blocks: int = 1,
        max_len: int = 30,
        dropout: float = 0.2,
        use_grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.use_grad_ckpt = use_grad_ckpt

        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=n_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.normal_(self.pos_emb.weight, std=0.01)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    # ------------------------------------------------------------------
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: LongTensor of shape (B, L) — zero-padded item-ID sequences.

        Returns:
            Tensor of shape (B, dim) — last non-padding position representation.
        """
        B, L = seq.shape
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)
        x = self.drop(self.item_emb(seq) + self.pos_emb(pos))
        for blk in self.blocks:
            if self.use_grad_ckpt and self.training:
                x = ckpt_utils.checkpoint(
                    lambda _x: blk(_x), x, use_reentrant=False
                )
            else:
                x = blk(x)
        x = self.norm(x)
        lens = (seq != 0).sum(dim=1).clamp(1, L) - 1
        return x[torch.arange(B, device=seq.device), lens]