"""
Shared pytest fixtures and configuration for the AniRec test suite.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

# ── Reproducibility ────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def set_seeds():
    """Fix random seeds for deterministic tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


# ── Dimension constants ────────────────────────────────────────────────────────


@pytest.fixture
def dims():
    """Standard small dimensions used across tests."""
    return {"n_u": 50, "n_i": 100, "dim": 64, "n_genres": 20, "batch": 8, "seq_len": 10}


# ── Sparse adjacency ───────────────────────────────────────────────────────────


@pytest.fixture
def random_adj():
    """Factory fixture: returns a function to build a random normalised adj matrix."""

    def _build(n_u: int, n_i: int, nnz: int = 200) -> sp.csr_matrix:
        n = n_u + n_i
        rows = np.random.randint(0, n, nnz)
        cols = np.random.randint(0, n, nnz)
        data = np.ones(nnz, dtype=np.float32)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        A = A + A.T
        deg = np.array(A.sum(1)).flatten()
        di = np.where(deg > 0, deg**-0.5, 0.0).astype(np.float32)
        A_tilde = sp.diags(di) @ A @ sp.diags(di)
        return A_tilde.tocsr()

    return _build


# ── Item metadata ──────────────────────────────────────────────────────────────


@pytest.fixture
def item_genres():
    """Minimal item_genres dict for 100 items."""
    genre_pool = [
        "Action",
        "Drama",
        "Comedy",
        "Sci-Fi",
        "Fantasy",
        "Romance",
        "Horror",
        "Psychological",
        "Thriller",
        "Slice of Life",
    ]
    rng = np.random.default_rng(42)
    return {
        i: list(rng.choice(genre_pool, size=rng.integers(1, 4), replace=False)) for i in range(100)
    }


@pytest.fixture
def item_popularity_norm():
    """Random popularity scores in [0, 1] for 100 items."""
    rng = np.random.default_rng(42)
    return rng.random(100).astype(np.float32)
