# -*- coding: utf-8 -*-
"""
AniRec v22 — Inference scorer.

The full InferenceScorer implementation lives in
``notebooks/experimentation.ipynb`` (Sections 11-14.7).

This module re-exports the class and provides a ``build_scorer_from_checkpoint``
factory used by ``infer.py`` and the FastAPI routes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_scorer_from_checkpoint(
    anime_catalog: Dict[int, dict],
    movie_catalog: Dict[int, dict],
    anime_interactions: Dict[str, List],
    movie_interactions: Dict[str, List],
    checkpoint_path: Optional[str] = None,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> "InferenceScorer":
    """Instantiate an InferenceScorer from a saved checkpoint."""
    from src.training.dataset import load_dataset_v19
    from src.models.anirec import AniRecV20

    cfg = config or {}
    _device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    embed_dir = Path(cfg.get("embed_dir", "./AniRec_output/v22/embeddings"))
    data = load_dataset_v19(
        anime_catalog, movie_catalog,
        anime_interactions, movie_interactions,
        embedding_dir=embed_dir if embed_dir.exists() else None,
    )

    n_u = data["n_users"]; n_i = data["n_items"]; n_g = data["n_genres"]
    has_text = data["item_text_embeddings"] is not None
    has_tone = data["tone_scores"] is not None

    model = AniRecV20(n_u, n_i, n_g,
                      use_text=has_text, use_tone=has_tone).to(_device)
    model.gnn.load_adj(data["A_tilde"], target_device=_device)

    if checkpoint_path is None:
        default_ckpt = Path(cfg.get("ckpt_dir", "./AniRec_output/v22")) / "best_v22.pt"
        checkpoint_path = str(default_ckpt)

    if os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        try:
            model.load_state_dict(ck["model"])
        except Exception:
            from collections import OrderedDict
            sd = OrderedDict({k.replace("_orig_mod.", ""): v
                              for k, v in ck["model"].items()})
            model.load_state_dict(sd, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("Checkpoint not found at %s — using untrained model.",
                       checkpoint_path)

    item_content_cpu = torch.FloatTensor(data["item_content"])
    seqs_cpu         = torch.LongTensor(data["seqs"])
    item_text_cpu    = (torch.FloatTensor(data["item_text_embeddings"])
                        if has_text else None)

    from src.data.feedback import FeedbackStore
    fb_path = Path(cfg.get("feedback_db", "./AniRec_output/v22/data/feedback.db"))
    fb_path.parent.mkdir(parents=True, exist_ok=True)
    feedback_store = FeedbackStore(fb_path)

    return InferenceScorer(
        model=model, device=_device,
        anime_catalog=anime_catalog,
        movie_catalog=movie_catalog,
        item_content_cpu=item_content_cpu, seqs_cpu=seqs_cpu,
        item_text_cpu=item_text_cpu,
        tone_scores_np=data.get("tone_scores"),
        item_popularity_norm=data["item_popularity_norm"],
        item_genres=data["item_genres"],
        item_popularity=data["item_popularity"],
        movie_names=data["movie_names"], anime_names=data["anime_names"],
        n_anime=data["n_anime"], item_domain=data["item_domain"],
        feedback_store=feedback_store,
        item_recency_norm=data.get("item_recency_norm"),
        item_year=data.get("item_year"),
        ani2i=data.get("ani2i", {}),
        item_sequel_ids=data.get("item_sequel_ids", {}),
    )


class InferenceScorer:
    """
    AniRec v22 inference scorer.

    The full scoring pipeline is in notebooks/experimentation.ipynb (Sec 11-14.7).
    This class exposes the public interface used by infer.py and the API routes.
    """

    def __init__(self,anime_catalog, movie_catalog, model, device, item_content_cpu, seqs_cpu,
                 item_text_cpu, tone_scores_np, item_popularity_norm,
                 item_genres, item_popularity, movie_names, anime_names,
                 n_anime, item_domain, feedback_store=None,
                 item_recency_norm=None, item_year=None,
                 ani2i=None, item_sequel_ids=None,
                 top_k_cands=2000, pop_floor=0.0,
                 score_weights=None, use_amp=True,
                 rng_=None) -> None:
        self.model                = model
        self.device               = device
        self.item_content_cpu     = item_content_cpu
        self.seqs_cpu             = seqs_cpu
        self.item_text_cpu        = item_text_cpu
        self.tone_scores_np       = tone_scores_np
        self.item_popularity_norm = item_popularity_norm
        self.item_genres          = item_genres
        self.item_popularity      = item_popularity
        self.movie_names          = movie_names
        self.anime_names          = anime_names
        self.n_anime              = n_anime
        self.item_domain          = item_domain
        self.feedback_store       = feedback_store
        self.item_text_np         = (item_text_cpu.numpy()
                                     if item_text_cpu is not None else None)
        self.item_recency_norm    = item_recency_norm
        self.item_year            = item_year or {}
        self.ani2i                = ani2i or {}
        self._item_sequel_ids     = item_sequel_ids or {}
        self.top_k_cands          = top_k_cands
        self.rng_                 = rng_ or np.random.default_rng(42)
        self._smart_search        = None
        self.anime_catalog = anime_catalog
        self.movie_catalog = movie_catalog

    def find_item(self, query: str):
        from src.inference.search import SmartSearch
        if self._smart_search is None:
            self._smart_search = SmartSearch(
                self.anime_names, self.movie_names, self.item_text_np)
        return self._smart_search.find(query)

    def recommend_from_items(self, item_ids, k=10, user_id=None,
                              ui_context="all", epsilon=0.0):
        scores = []

        seed_set = set(item_ids)

        for item_id, item in self.anime_catalog.items():
            if item_id in seed_set:
                continue
            
            popularity = float(item.get("popularity", 0))
            score = popularity

            scores.append((item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        top_items = scores[:k]

        results = [
            self.anime_catalog[item_id]
            for item_id, _ in top_items
        ]

        return results, top_items

    def recommend_by_names(self, names, k=10, user_id=None,
                            ui_context="all", epsilon=0.0):
        ids = []
        for nm in names:
            idx, _ = self.find_item(nm)
            if idx is not None:
                ids.append(idx)
        if not ids:
            return [], []
        return self.recommend_from_items(ids, k=k, user_id=user_id,
                                         ui_context=ui_context, epsilon=epsilon)

    def recommend_from_watchlist(self, username, watchlist=None,
                                  k=10, ui_context="all"):
        raise NotImplementedError(
            "Full watchlist scoring is in notebooks/experimentation.ipynb.")