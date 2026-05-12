"""
Dataset builder for AniRec v22.

Merges AniList anime interactions with MovieLens movie interactions into a
single shared item/user space, constructs train/val/test splits, builds the
normalised LightGCN adjacency matrix, and loads pre-computed embeddings.
"""

from __future__ import annotations

import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import scipy.sparse as sp

from src.data.cleaning import normalize_popularity

# ---------------------------------------------------------------------------
# Constants (defaults; override by passing kwargs or setting before import)
# ---------------------------------------------------------------------------
MIN_WEIGHT_THRESHOLD = 0.1
MAX_SEQ_LEN = 30
SEED = 42
ML_USER_PREFIX = "ml_"


# ---------------------------------------------------------------------------
def _build_genre_vocab(items: List[dict]) -> Dict[str, int]:
    all_g: set = set()
    for it in items:
        all_g.update(it.get("genres", []))
    all_g.discard("")
    return {g: i for i, g in enumerate(sorted(all_g))}


def _encode_genres_list(items: List[dict], g2i: Dict[str, int]) -> np.ndarray:
    mat = np.zeros((len(items), len(g2i)), dtype=np.float32)
    for i, it in enumerate(items):
        for g in it.get("genres", []):
            if g in g2i:
                mat[i, g2i[g]] = 1.0
    return mat


def _build_item_sequel_ids(
    anime_catalog: Dict[int, dict],
    al2i: Dict[int, int],
    ani2i: Dict[int, int],
) -> Dict[int, Set[int]]:
    result: Dict[int, Set[int]] = {}
    for mal_id, meta in anime_catalog.items():
        src_idx = ani2i.get(mal_id, -1)
        if src_idx < 0:
            continue
        for seq_al_id in meta.get("sequel_ids") or []:
            dst_idx = al2i.get(int(seq_al_id), -1)
            if dst_idx < 0:
                dst_idx = ani2i.get(int(seq_al_id), -1)
            if dst_idx >= 0 and dst_idx != src_idx:
                result.setdefault(src_idx, set()).add(dst_idx)
                result.setdefault(dst_idx, set()).add(src_idx)
    return result


# ---------------------------------------------------------------------------
def load_dataset_v19(
    anime_catalog: Dict[int, dict],
    movie_catalog: Dict[int, dict],
    anime_interactions: Dict[str, List],
    movie_interactions: Dict[str, List],
    embedding_dir: Optional[Path] = None,
    max_seq_len: int = MAX_SEQ_LEN,
    rng_: Optional[np.random.Generator] = None,
) -> dict:
    """
    Build the full training dataset from catalogs and interaction dicts.

    Returns a flat dict consumed by the training loop and InferenceScorer.
    """
    if rng_ is None:
        rng_ = np.random.default_rng(SEED)

    all_users = sorted(set(anime_interactions) | set(movie_interactions))
    u2i = {u: i for i, u in enumerate(all_users)}
    n_u = len(all_users)

    anime_list = sorted(anime_catalog.keys())
    movie_list = sorted(movie_catalog.keys())
    n_ani = len(anime_list)
    n_mov = len(movie_list)
    n_i = n_ani + n_mov

    ani2i = {a: i for i, a in enumerate(anime_list)}
    mov2i = {m: i + n_ani for i, m in enumerate(movie_list)}
    item_domain = {**dict.fromkeys(range(n_ani), 0), **dict.fromkeys(range(n_ani, n_i), 1)}

    al2i: Dict[int, int] = {}
    for mal_id, meta in anime_catalog.items():
        al_id = meta.get("anilist_id")
        if al_id and int(al_id) not in al2i:
            al2i[int(al_id)] = ani2i.get(mal_id, -1)
    for mal_id, idx in ani2i.items():
        if mal_id not in al2i:
            al2i[mal_id] = idx

    n_ml_users = sum(1 for u in all_users if u.startswith(ML_USER_PREFIX))

    # ── Build interaction arrays ──────────────────────────────────────────
    tr_u, tr_i, tr_d, tr_w, tr_t = [], [], [], [], []

    for uname, records in anime_interactions.items():
        uid = u2i.get(uname)
        if uid is None:
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            item_id = int(rec.get("item_id") or rec.get("anilist_id") or 0)
            weight = float(rec.get("weight", 1.0))
            ts = int(rec.get("ts", 0))
            idx = al2i.get(item_id) or ani2i.get(item_id)
            if idx is None or weight < MIN_WEIGHT_THRESHOLD:
                continue
            tr_u.append(uid)
            tr_i.append(idx)
            tr_d.append(0)
            tr_w.append(weight)
            tr_t.append(ts)

    for uname, mlist in movie_interactions.items():
        uid = u2i.get(uname)
        if uid is None:
            continue
        for entry in mlist:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("imdb_num") or entry.get("item_id") or entry.get("imdb_id")
            weight = float(entry.get("weight", 1.0))
            ts = int(entry.get("ts", 0))
            if mid is None or int(mid) not in mov2i:
                continue
            if weight < MIN_WEIGHT_THRESHOLD:
                continue
            tr_u.append(uid)
            tr_i.append(mov2i[int(mid)])
            tr_d.append(1)
            tr_w.append(weight)
            tr_t.append(ts)

    tr_u = np.array(tr_u, dtype=np.int32)
    tr_i = np.array(tr_i, dtype=np.int32)
    tr_d = np.array(tr_d, dtype=np.int32)
    tr_w = np.array(tr_w, dtype=np.float32)
    tr_t = np.array(tr_t, dtype=np.int64)

    # ── Filter sparse ─────────────────────────────────────────────────────
    if len(tr_u) > 0:
        uc = defaultdict(float)
        for u, w in zip(tr_u.tolist(), tr_w.tolist()):
            uc[u] += w
        mask = np.array([uc[u] >= MIN_WEIGHT_THRESHOLD * 3 for u in tr_u])
        tr_u, tr_i, tr_d, tr_w, tr_t = (tr_u[mask], tr_i[mask], tr_d[mask], tr_w[mask], tr_t[mask])

        ic = Counter(tr_i.tolist())
        mask = np.array([ic[i] >= 2 for i in tr_i])
        tr_u, tr_i, tr_d, tr_w, tr_t = (tr_u[mask], tr_i[mask], tr_d[mask], tr_w[mask], tr_t[mask])

    # ── Train / val / test splits ─────────────────────────────────────────
    user_hist: Dict[int, List] = {}
    for u, i, d, w, t in zip(tr_u, tr_i, tr_d, tr_w, tr_t):
        user_hist.setdefault(int(u), []).append((int(i), int(d), float(w), int(t)))
    for uid in user_hist:
        user_hist[uid].sort(key=lambda x: (x[3], x[0]))

    train_triples, val_data, test_data = [], [], []
    user_train_items: Dict[int, set] = {}

    for uid, hist in user_hist.items():
        if len(hist) < 3:
            continue
        for i, d, w, _ in hist[:-2]:
            train_triples.append((uid, i, d, w))
            user_train_items.setdefault(uid, set()).add(i)
        vi, vd, _, _ = hist[-2]
        val_data.append((uid, vi, vd))
        ti, td, _, _ = hist[-1]
        test_data.append((uid, ti, td))

    train_arr = (
        np.array([(t[0], t[1], t[2]) for t in train_triples], dtype=np.int32)
        if train_triples
        else np.zeros((0, 3), dtype=np.int32)
    )
    train_w = (
        np.array([t[3] for t in train_triples], dtype=np.float32)
        if train_triples
        else np.zeros(0, dtype=np.float32)
    )
    val_arr = np.array(val_data, dtype=np.int32) if val_data else np.zeros((0, 3), dtype=np.int32)
    test_arr = (
        np.array(test_data, dtype=np.int32) if test_data else np.zeros((0, 3), dtype=np.int32)
    )

    tr_u2 = train_arr[:, 0] if len(train_arr) else np.zeros(0, dtype=np.int32)
    tr_i2 = train_arr[:, 1] if len(train_arr) else np.zeros(0, dtype=np.int32)
    tr_d2 = train_arr[:, 2] if len(train_arr) else np.zeros(0, dtype=np.int32)

    anime_train_idx = (
        np.where(tr_d2 == 0)[0].astype(np.int32) if len(tr_d2) else np.zeros(0, dtype=np.int32)
    )
    movie_train_idx = (
        np.where(tr_d2 == 1)[0].astype(np.int32) if len(tr_d2) else np.zeros(0, dtype=np.int32)
    )

    # ── Content ───────────────────────────────────────────────────────────
    all_items_meta = [anime_catalog[a] for a in anime_list] + [movie_catalog[m] for m in movie_list]
    g2i = _build_genre_vocab(all_items_meta)
    item_content = _encode_genres_list(all_items_meta, g2i)

    # ── Sequences ─────────────────────────────────────────────────────────
    seqs = np.zeros((n_u, max_seq_len), dtype=np.int64)
    for uid, hist in user_hist.items():
        if uid >= n_u:
            continue
        items = [h[0] for h in hist][-max_seq_len:]
        seqs[uid, max_seq_len - len(items) :] = [it + 1 for it in items]
    user_seq_snapshot = {
        uid: [h[0] for h in hist][-max_seq_len:] for uid, hist in user_hist.items()
    }

    # ── Adjacency matrix ─────────────────────────────────────────────────
    n = n_u + n_i
    if len(tr_u2) > 0:
        row = np.concatenate([tr_u2, tr_i2 + n_u])
        col = np.concatenate([tr_i2 + n_u, tr_u2])
        wts = np.concatenate([train_w, train_w])
        A = sp.csr_matrix((wts, (row, col)), shape=(n, n))
    else:
        A = sp.csr_matrix((n, n))
    deg = np.array(A.sum(1)).flatten()
    di = np.where(deg > 0, deg**-0.5, 0.0).astype(np.float32)
    A_tilde = (sp.diags(di) @ A @ sp.diags(di)).tocsr()
    A_tilde.sort_indices()
    A_tilde.eliminate_zeros()
    del A, di, deg
    gc.collect()

    # ── Popularity / recency ──────────────────────────────────────────────
    item_pop_raw = np.zeros(n_i, dtype=np.float32)
    if len(tr_u2) > 0:
        for i_idx, w in zip(tr_i2.tolist(), train_w.tolist()):
            item_pop_raw[i_idx] += w
    # Catalog floor for cold items
    for a in anime_list:
        ai = ani2i[a]
        if item_pop_raw[ai] < 1.0:
            item_pop_raw[ai] = max(float(anime_catalog[a].get("popularity") or 0) / 1000.0, 1.0)
    for m in movie_list:
        mi = mov2i[m]
        if item_pop_raw[mi] < 1.0:
            vc = float(movie_catalog[m].get("vote_count", 0) or 0)
            va = float(movie_catalog[m].get("vote_average", 0) or 0)
            tp = float(movie_catalog[m].get("popularity", 0) or 0)
            item_pop_raw[mi] = max(vc / 1000.0 + va * 10 + tp, 1.0)

    item_pop = item_pop_raw / (item_pop_raw.sum() + 1e-8)
    item_pop_norm = normalize_popularity(item_pop_raw)
    item_popularity = dict(enumerate(item_pop_raw.astype(np.float64).tolist()))

    # Year-based recency
    item_year: Dict[int, int] = {}
    for a in anime_list:
        yr_str = (anime_catalog[a].get("aired_from") or "")[:4]
        try:
            item_year[ani2i[a]] = int(yr_str)
        except ValueError:
            item_year[ani2i[a]] = 0
    for m in movie_list:
        yr_str = str(movie_catalog[m].get("release_year", "") or "")[:4]
        try:
            item_year[mov2i[m]] = int(yr_str)
        except ValueError:
            item_year[mov2i[m]] = 0

    # ── Descriptions + embeddings ─────────────────────────────────────────
    item_descriptions: Dict[int, str] = {}
    for a in anime_list:
        item_descriptions[ani2i[a]] = anime_catalog[a].get("synopsis", "")
    for m in movie_list:
        overview = movie_catalog[m].get("overview", "")
        if not overview:
            title = movie_catalog[m].get("title", "")
            genres = " ".join(movie_catalog[m].get("genres", []))
            yr = str(movie_catalog[m].get("release_year", "") or "")
            overview = f"{title} {genres} {yr}".strip()
        item_descriptions[mov2i[m]] = overview

    item_text_embeddings = None
    tone_scores = None
    tone_labels = None
    if embedding_dir and Path(embedding_dir).exists():
        from src.training.embeddings import load_precomputed_embeddings

        emb_dir = Path(embedding_dir)
        if (emb_dir / "item_text_embeddings.npy").exists():
            arts = load_precomputed_embeddings(emb_dir)
            item_text_embeddings = arts.get("item_text_embeddings")
            tone_scores = arts.get("tone_scores")
            tone_labels = arts.get("tone_labels")

    # Names
    anime_names = {ani2i[a]: anime_catalog[a].get("title", f"anime_{a}") for a in anime_list}
    movie_names = {mov2i[m]: movie_catalog[m].get("title", f"movie_{m}") for m in movie_list}
    item_genres = {ani2i[a]: anime_catalog[a].get("genres", []) for a in anime_list}
    item_genres.update({mov2i[m]: movie_catalog[m].get("genres", []) for m in movie_list})

    return dict(
        n_users=n_u,
        n_items=n_i,
        n_anime=n_ani,
        n_movies=n_mov,
        n_genres=len(g2i),
        item_content=item_content,
        item_pop=item_pop,
        item_popularity=item_popularity,
        item_popularity_norm=item_pop_norm,
        item_year=item_year,
        train=dict(users=tr_u2, items=tr_i2, domains=tr_d2, weights=train_w),
        anime_train_idx=anime_train_idx,
        movie_train_idx=movie_train_idx,
        val=val_arr,
        test=test_arr,
        A_tilde=A_tilde,
        seqs=seqs,
        user_seq_full=user_seq_snapshot,
        anime_names=anime_names,
        movie_names=movie_names,
        item_genres=item_genres,
        genre_vocab=g2i,
        user_train_items=user_train_items,
        item_text_embeddings=item_text_embeddings,
        tone_scores=tone_scores,
        tone_labels=tone_labels,
        item_descriptions=item_descriptions,
        user_id_map=u2i,
        item_id_map={**ani2i, **{m: mov2i[m] for m in movie_list}},
        n_ml_users=n_ml_users,
        item_domain=item_domain,
        ani2i=ani2i,
        mov2i=mov2i,
        al2i=al2i,
        item_sequel_ids=_build_item_sequel_ids(anime_catalog, al2i, ani2i),
    )
