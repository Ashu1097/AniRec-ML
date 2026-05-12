"""
AniRec v22 — training loop entry point.

Delegates the heavy lifting to the full implementation in
``notebooks/experimentation.ipynb``.  This module exposes a clean
``run_training`` function that can be called from ``train.py`` or imported
by tests.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_training(
    anime_catalog: Dict[int, dict],
    movie_catalog: Dict[int, dict],
    anime_interactions: Dict[str, List],
    movie_interactions: Dict[str, List],
    config: Optional[dict] = None,
    ckpt_dir: Optional[str] = None,
    n_epochs: Optional[int] = None,
    resume: bool = True,
) -> None:
    """
    Launch the AniRec v22 training pipeline.

    Args:
        anime_catalog:       ``{anilist_id: meta_dict}`` mapping.
        movie_catalog:       ``{imdb_id: meta_dict}`` mapping.
        anime_interactions:  ``{username: [interaction_records]}`` dict.
        movie_interactions:  ``{ml_user_key: [interaction_records]}`` dict.
        config:              Optional dict of hyper-parameter overrides.
        ckpt_dir:            Directory to save checkpoints.
        n_epochs:            Override the default epoch count.
        resume:              Whether to resume from an existing checkpoint.
    """
    # Local import keeps startup fast when only inference is needed.
    try:
        import numpy as np
        import torch
        from torch.amp import GradScaler
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install it with: pip install torch"
        ) from exc

    from src.models.anirec import AniRecV20
    from src.training.dataset import load_dataset_v19
    from src.utils.progress import print_ok, print_section, print_warn

    cfg = config or {}
    _n_epochs = n_epochs or cfg.get("n_epochs", 40)
    _ckpt_dir = ckpt_dir or cfg.get("ckpt_dir", "./AniRec_output/v22")
    os.makedirs(_ckpt_dir, exist_ok=True)

    print_section("BUILDING DATASET")
    embed_dir = Path(_ckpt_dir) / "embeddings"
    data = load_dataset_v19(
        anime_catalog,
        movie_catalog,
        anime_interactions,
        movie_interactions,
        embedding_dir=embed_dir if embed_dir.exists() else None,
    )

    n_u = data["n_users"]
    n_i = data["n_items"]
    n_g = data["n_genres"]
    has_text = data["item_text_embeddings"] is not None
    has_tone = data["tone_scores"] is not None

    logger.info("Users: %d  Items: %d  Genres: %d", n_u, n_i, n_g)
    logger.info("Text embeddings: %s  Tone: %s", has_text, has_tone)

    N_TRAIN = len(data["train"]["users"])
    if N_TRAIN < 100:
        print_warn(
            f"Only {N_TRAIN} training triples — skipping training."
            " Re-run preprocess with more users."
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_section("INITIALISING MODEL")
    model = AniRecV20(
        n_u,
        n_i,
        n_g,
        use_text=has_text,
        use_tone=has_tone,
    ).to(device)
    model.gnn.load_adj(data["A_tilde"], target_device=device)

    total_params = sum(p.numel() for p in model.parameters())
    print_ok(f"Parameters: {total_params / 1e6:.2f}M")
    print_ok("Training loop is implemented in notebooks/experimentation.ipynb.")
    print_ok("Use RUN_MODE='train' in the notebook for the full training run.")
    logger.info("Trainer stub executed successfully — model initialised.")
