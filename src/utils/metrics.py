# -*- coding: utf-8 -*-
"""Ranking evaluation metrics: HR@K, NDCG@K, MRR."""

from __future__ import annotations

import numpy as np


def compute_hr_at_k(ranked_items: np.ndarray, ground_truth: int,
                    k: int) -> float:
    """Hit-Rate@K: 1 if ground_truth appears in the top-k, else 0."""
    return 1.0 if ground_truth in ranked_items[:k] else 0.0


def compute_ndcg_at_k(ranked_items: np.ndarray, ground_truth: int,
                      k: int) -> float:
    """Normalised Discounted Cumulative Gain@K."""
    hits = np.where(ranked_items[:k] == ground_truth)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / np.log2(int(hits[0]) + 2)


def compute_mrr(ranked_items: np.ndarray, ground_truth: int) -> float:
    """Mean Reciprocal Rank."""
    hits = np.where(ranked_items == ground_truth)[0]
    if len(hits) == 0:
        return 0.0
    return 1.0 / (int(hits[0]) + 1)