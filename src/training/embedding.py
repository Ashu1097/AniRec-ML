"""Text-embedding and tone-clustering pipeline for AniRec v22."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict

import numpy as np
import torch

TEXT_EMBED_DIM = 384
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
TONE_N_CLUSTERS = 12

TONE_SEED_PAIRS = [
    ("dark psychological thriller suspense dread", "lighthearted fun cheerful wholesome comedy"),
    (
        "action intense fight combat explosive fast-paced",
        "slow calm meditative peaceful reflective",
    ),
    ("romantic love heartwarming tender emotional bond", "cold detached stoic nihilistic lonely"),
    (
        "supernatural magic mystical spiritual otherworldly",
        "realistic grounded everyday mundane slice-of-life",
    ),
    ("crime investigation detective mystery forensic", "innocent school family ordinary everyday"),
    (
        "sci-fi futuristic technology dystopian cyberpunk",
        "historical period drama traditional ancient",
    ),
]


# ---------------------------------------------------------------------------
def build_text_embeddings(
    item_descriptions: Dict[int, str],
    n_items: int,
    output_path: Path,
    batch_size: int = 256,
    device_str: str = "cpu",
) -> np.ndarray:
    """Encode item descriptions with a sentence-transformer model."""
    from sentence_transformers import SentenceTransformer

    output_path = Path(output_path)
    if output_path.exists():
        arr = np.load(str(output_path))
        assert arr.shape == (n_items, TEXT_EMBED_DIM)
        return arr

    _dev = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    model = SentenceTransformer(TEXT_MODEL_NAME, device=_dev)
    model.eval()

    texts = [item_descriptions.get(i, "") for i in range(n_items)]
    has_text = [i for i in range(n_items) if texts[i]]
    result = np.zeros((n_items, TEXT_EMBED_DIM), dtype=np.float32)

    all_embs = []
    for start in range(0, len(has_text), batch_size):
        batch = [texts[i] for i in has_text[start : start + batch_size]]
        embs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
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
    return result


# ---------------------------------------------------------------------------
def build_tone_axes(device_str: str = "cpu") -> np.ndarray:
    """Build tone projection axes from seed phrase pairs."""
    from sentence_transformers import SentenceTransformer

    _dev = device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    model = SentenceTransformer(TEXT_MODEL_NAME, device=_dev)
    pos_embs = model.encode(
        [p for p, _ in TONE_SEED_PAIRS], normalize_embeddings=True, convert_to_numpy=True
    )
    neg_embs = model.encode(
        [n for _, n in TONE_SEED_PAIRS], normalize_embeddings=True, convert_to_numpy=True
    )
    axes = pos_embs - neg_embs
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.where(norms < 1e-8, 1.0, norms)
    del model
    gc.collect()
    return axes.astype(np.float32)


# ---------------------------------------------------------------------------
def build_embedding_pipeline(
    item_descriptions: Dict[int, str],
    n_items: int,
    output_dir: Path,
    device_str: str = "cpu",
) -> dict:
    """Run the full embedding pipeline and persist results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    emb_path = output_dir / "item_text_embeddings.npy"
    embeddings = build_text_embeddings(item_descriptions, n_items, emb_path, device_str=device_str)

    axes_path = output_dir / "tone_axes.npy"
    if axes_path.exists():
        tone_axes = np.load(str(axes_path))
    else:
        tone_axes = build_tone_axes(device_str)
        np.save(str(axes_path), tone_axes)

    tsc_path = output_dir / "tone_scores.npy"
    if tsc_path.exists():
        tone_scores = np.load(str(tsc_path))
    else:
        tone_scores = (embeddings @ tone_axes.T).astype(np.float32)
        np.save(str(tsc_path), tone_scores)

    lbl_path = output_dir / "tone_clusters_labels.npy"
    cen_path = output_dir / "tone_clusters_centroids.npy"
    if lbl_path.exists() and cen_path.exists():
        labels = np.load(str(lbl_path))
        centroids = np.load(str(cen_path))
    else:
        from sklearn.cluster import MiniBatchKMeans

        valid_mask = np.linalg.norm(embeddings, axis=1) > 1e-6
        km = MiniBatchKMeans(
            n_clusters=TONE_N_CLUSTERS, init="k-means++", n_init=3, random_state=42
        )
        km.fit(embeddings[valid_mask])
        labels = np.full(len(embeddings), -1, dtype=np.int16)
        labels[valid_mask] = km.labels_.astype(np.int16)
        centroids = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids /= np.where(norms < 1e-8, 1.0, norms)
        np.save(str(lbl_path), labels)
        np.save(str(cen_path), centroids)

    return dict(
        item_text_embeddings=embeddings,
        tone_axes=tone_axes,
        tone_scores=tone_scores,
        tone_labels=labels,
        tone_centroids=centroids,
    )


# ---------------------------------------------------------------------------
def load_precomputed_embeddings(output_dir: Path) -> dict:
    """Load all pre-computed embedding artefacts from disk."""
    output_dir = Path(output_dir)
    files = {
        "item_text_embeddings": "item_text_embeddings.npy",
        "tone_axes": "tone_axes.npy",
        "tone_scores": "tone_scores.npy",
        "tone_labels": "tone_clusters_labels.npy",
        "tone_centroids": "tone_clusters_centroids.npy",
    }
    return {
        key: np.load(str(output_dir / fname)) if (output_dir / fname).exists() else None
        for key, fname in files.items()
    }
