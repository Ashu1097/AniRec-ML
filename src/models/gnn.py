# -*- coding: utf-8 -*-
"""LightGCN graph neural network for collaborative filtering."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    Simplified Graph Convolutional Network for recommendation.

    Propagates user/item embeddings over a normalised bipartite interaction
    graph for ``n_layers`` hops, then averages all hop representations.

    Reference: He et al., "LightGCN: Simplifying and Powering Graph
    Convolution Network for Recommendation", SIGIR 2020.
    """

    def __init__(self, n_users: int, n_items: int, dim: int = 256,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_layers = n_layers
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        self._A: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def load_adj(self, A_tilde: sp.spmatrix,
                 target_device: torch.device | None = None) -> None:
        """Convert a scipy sparse adjacency matrix to a CUDA sparse tensor."""
        A = A_tilde.tocoo()
        idx = torch.LongTensor(np.vstack([A.row, A.col]))
        val = torch.FloatTensor(A.data)
        dev = target_device if target_device is not None \
            else self.user_emb.weight.device
        self._A = torch.sparse_coo_tensor(idx, val, A.shape).to(dev)

    # ------------------------------------------------------------------
    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return aggregated (user_embeddings, item_embeddings)."""
        with torch.amp.autocast("cuda", enabled=False):
            E = torch.cat(
                [self.user_emb.weight, self.item_emb.weight], dim=0
            ).float()
            A_fp32 = self._A.float()
            E_mean = E / (self.n_layers + 1)
            E_cur = E
            for _ in range(self.n_layers):
                E_cur = torch.sparse.mm(A_fp32, E_cur)
                E_mean = E_mean + E_cur / (self.n_layers + 1)
        E_final = E_mean.to(self.user_emb.weight.dtype)
        return E_final[: self.n_users], E_final[self.n_users :]