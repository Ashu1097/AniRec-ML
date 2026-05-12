# -*- coding: utf-8 -*-
"""Text embedding pipeline and tone axis utilities."""

import gc
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

TEXT_EMBED_DIM = 384
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
TONE_N_CLUSTERS = 12

TONE_SEED_PAIRS = [
    ("dark psychological thriller suspense dread",
     "lighthearted fun cheerful wholesome comedy"),
    ("action intense fight combat explosive fast-paced",
     "slow calm meditative peaceful reflective"),
    ("romantic love heartwarming tender emotional bond",
     "cold detached stoic nihilistic lonely"),
    ("supernatural magic mystical spiritual otherworldly",
     "realistic grounded everyday mundane slice-of-life"),
    ("crime investigation detective mystery forensic",
     "innocent school family ordinary everyday"),
    ("sci-fi futuristic technology dystopian cyberpunk",
     "historical period drama traditional ancient"),
]


def build_text_embeddings(
    item_descriptions: Dict[int, str],
    n_items: int,
    output_path: Path,
    batch_size: int = 256,
    device_str: str = "cpu",
) -> np.ndarray:
    """Encode item descriptions with a sentence-transformer and cache to disk."""
    from sentence_transformers import SentenceTransformer

    output_path = Path(output_path)
    if output_path.exists():
        arr = np.load(str(output_path))
        assert arr.shape == (n_items, TEXT_EMBED_DIM)
        return arr

    _dev = device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu"
    model = SentenceTransformer(TEXT_MODEL_NAME, device=_dev)
    model.eval()

    texts = [item_descriptions.get(i, "") for i in range(n_items)]
    has_text = [i for i in range(n_items) if texts[i]]
    result = np.zeros((n_items, TEXT_EMBED_DIM), dtype=np.float32)

    all_embs = []
    for start in range(0, len(has_text), batch_size):
        batch = [texts[i] for i in has_text[start : start + batch_size]]
        embs = model.encode(
            batch, batch_size=batch_size,
            show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True,
        )
        all_embs.append(embs)

    if all_embs:
        stacked = np.vstack(all_embs).astype(np.float32)
        for out_idx, item_id in enumerate(has_text):
            result[item_id] = stacked[out_idx]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), result)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def build_tone_axes(device_str: str = "cpu") -> np.ndarray:
    """Build tone axis vectors from seed sentence pairs."""
    from sentence_transformers import SentenceTransformer

    _dev = device_str if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(TEXT_MODEL_NAME, device=_dev)
    pos_embs = model.encode([p for p, _ in TONE_SEED_PAIRS],
                             normalize_embeddings=True, convert_to_numpy=True)
    neg_embs = model.encode([n for _, n in TONE_SEED_PAIRS],
                             normalize_embeddings=True, convert_to_numpy=True)
    axes = pos_embs - neg_embs
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes /= np.where(norms < 1e-8, 1.0, norms)
    del model
    gc.collect()
    return axes.astype(np.float32)
