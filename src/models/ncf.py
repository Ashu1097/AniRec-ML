# -*- coding: utf-8 -*-
"""Neural Collaborative Filtering head and GatedFusion module."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults — overridden at construction time when values are passed in.
_DEFAULT_MLP_DIMS = [256, 128, 64]
_TEXT_EMBED_DIM = 384
_TEXT_PROJ_DIM = 64
_TONE_DIM_IN = 6
_TONE_PROJ_DIM = 16
_DOMAIN_DIM = 32
_CONTENT_DIM = 64
_N_DOMAINS = 3


class GatedFusion(nn.Module):
    """
    Gated combination of GNN user embedding and SASRec sequence embedding.

    Learns a per-dimension gate g ∈ (0,1)::

        output = g * gnn_u + (1 - g) * seq_u
    """

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, gnn_u: torch.Tensor,
                seq_u: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([gnn_u, seq_u], dim=-1))
        return g * gnn_u + (1 - g) * seq_u


class NCF(nn.Module):
    """
    Neural Collaborative Filtering head.

    Combines user and item embeddings with domain, content (genre), text,
    and tone features through a GMF path and an MLP path, then fuses them.
    """

    def __init__(
        self,
        dim: int = 256,
        domain_dim: int = _DOMAIN_DIM,
        content_dim: int = _CONTENT_DIM,
        n_domains: int = _N_DOMAINS,
        n_genres: int = 51,
        mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_text: bool = True,
        use_tone: bool = True,
        text_emb_dim: int = _TEXT_EMBED_DIM,
        text_proj_dim: int = _TEXT_PROJ_DIM,
        tone_dim_in: int = _TONE_DIM_IN,
        tone_proj_dim: int = _TONE_PROJ_DIM,
    ) -> None:
        super().__init__()
        if mlp_dims is None:
            mlp_dims = list(_DEFAULT_MLP_DIMS)
        self.use_text = use_text
        self.use_tone = use_tone

        self.domain_emb = nn.Embedding(n_domains, domain_dim)
        self.content_proj = nn.Linear(n_genres, content_dim)
        self.gmf_u = nn.Linear(dim, dim, bias=False)
        self.gmf_i = nn.Linear(dim, dim, bias=False)

        _text_dim = 0
        if use_text:
            self.text_proj = nn.Sequential(
                nn.Linear(text_emb_dim, text_proj_dim * 2),
                nn.LayerNorm(text_proj_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(text_proj_dim * 2, text_proj_dim),
                nn.LayerNorm(text_proj_dim),
            )
            _text_dim = text_proj_dim

        _tone_dim = 0
        if use_tone:
            self.tone_proj = nn.Sequential(
                nn.Linear(tone_dim_in, tone_proj_dim),
                nn.LayerNorm(tone_proj_dim),
                nn.GELU(),
            )
            _tone_dim = tone_proj_dim

        mlp_in = dim + dim + domain_dim + content_dim + _text_dim + _tone_dim
        layers: list = []
        prev = mlp_in
        for d in mlp_dims:
            layers += [
                nn.Linear(prev, d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = d
        self.mlp = nn.Sequential(*layers)
        self.fusion = nn.Sequential(
            nn.Linear(dim + mlp_dims[-1], 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.domain_emb.weight, std=0.01)

    # ------------------------------------------------------------------
    def _zero_feat(self, layer: nn.Module, B: int,
                   device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        d = layer[-1].normalized_shape[0]
        return torch.zeros(B, d, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        u_emb: torch.Tensor,
        i_emb: torch.Tensor,
        domain_ids: torch.Tensor,
        item_content: torch.Tensor,
        item_text_emb: Optional[torch.Tensor] = None,
        item_tone: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dom = self.domain_emb(domain_ids)
        cont = F.gelu(self.content_proj(item_content))
        gmf = self.gmf_u(u_emb) * self.gmf_i(i_emb)
        parts = [u_emb, i_emb, dom, cont]
        if self.use_text:
            parts.append(
                self.text_proj(item_text_emb)
                if item_text_emb is not None
                else self._zero_feat(
                    self.text_proj, u_emb.shape[0], u_emb.device, u_emb.dtype
                )
            )
        if self.use_tone:
            parts.append(
                self.tone_proj(item_tone)
                if item_tone is not None
                else self._zero_feat(
                    self.tone_proj, u_emb.shape[0], u_emb.device, u_emb.dtype
                )
            )
        mlp_out = self.mlp(torch.cat(parts, dim=-1))
        return self.fusion(torch.cat([gmf, mlp_out], dim=-1))