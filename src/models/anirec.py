# -*- coding: utf-8 -*-
"""AniRecV20: full composite model (LightGCN + SASRec + GatedFusion + NCF)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.gnn import LightGCN
from src.models.sasrec import SASRec
from src.models.ncf import GatedFusion, NCF


class AniRecV20(nn.Module):
    """
    AniRec v22 end-to-end model.

    Combines:
      * **LightGCN** — graph-based collaborative filtering embeddings.
      * **SASRec**   — sequential self-attention over interaction history.
      * **GatedFusion** — learned blend of GNN and sequential representations.
      * **NCF** — final scoring head with content, text, and tone features.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        dim: int = 256,
        use_text: bool = True,
        use_tone: bool = True,
        use_grad_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.gnn = LightGCN(n_users, n_items, dim=dim)
        self.sasrec = SASRec(n_items, dim=dim, use_grad_ckpt=use_grad_ckpt)
        self.fusion = GatedFusion(dim=dim)
        self.ncf = NCF(
            dim=dim,
            n_genres=n_genres,
            use_text=use_text,
            use_tone=use_tone,
        )

    # ------------------------------------------------------------------
    def get_user_rep(
        self,
        u_ids: torch.Tensor,
        seqs: torch.Tensor,
        gnn_u: torch.Tensor,
        gnn_i: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse GNN and sequential user representations."""
        return self.fusion(gnn_u[u_ids], self.sasrec(seqs))

    # ------------------------------------------------------------------
    def score(
        self,
        u_rep: torch.Tensor,
        i_ids: torch.Tensor,
        domain_ids: torch.Tensor,
        item_content: torch.Tensor,
        gnn_i: torch.Tensor,
        item_text_emb: torch.Tensor | None = None,
        item_tone: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw (un-sigmoid) interaction scores."""
        return self.ncf(
            u_rep,
            gnn_i[i_ids],
            domain_ids,
            item_content,
            item_text_emb=item_text_emb,
            item_tone=item_tone,
        )

    # ------------------------------------------------------------------
    def encode_seq(self, seqs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sequences into L2-normalised vectors."""
        return F.normalize(self.sasrec(seqs), dim=-1)


# Backward-compat alias used in some checkpoints
AniRecV19 = AniRecV20