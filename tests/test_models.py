"""Tests for model architectures: LightGCN, SASRec, NCF, AniRecV20."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src.models.anirec import AniRecV20, GatedFusion
from src.models.lightgcn import LightGCN
from src.models.ncf import NCF
from src.models.sasrec import SASRec

# ── Fixtures ──────────────────────────────────────────────────────────────────

N_U, N_I, DIM = 50, 100, 256  # DIM=256 matches default model config
N_GENRES = 20
BATCH = 8
SEQ_LEN = 10


def _random_adj(n_u, n_i, nnz=200):
    """Build a small random symmetric normalised adjacency matrix."""
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


# ── LightGCN ──────────────────────────────────────────────────────────────────


class TestLightGCN:
    def test_forward_shapes(self):
        model = LightGCN(N_U, N_I, dim=DIM)
        A = _random_adj(N_U, N_I)
        model.load_adj(A, target_device=torch.device("cpu"))
        gnn_u, gnn_i = model()
        assert gnn_u.shape == (N_U, DIM)
        assert gnn_i.shape == (N_I, DIM)

    def test_embeddings_differ(self):
        model = LightGCN(N_U, N_I, dim=DIM)
        A = _random_adj(N_U, N_I)
        model.load_adj(A, target_device=torch.device("cpu"))
        gnn_u, gnn_i = model()
        # Not all identical
        assert gnn_u.std().item() > 1e-6

    def test_grad_flows(self):
        model = LightGCN(N_U, N_I, dim=DIM)
        A = _random_adj(N_U, N_I)
        model.load_adj(A, target_device=torch.device("cpu"))
        gnn_u, gnn_i = model()
        loss = gnn_u.sum() + gnn_i.sum()
        loss.backward()
        assert model.user_emb.weight.grad is not None
        assert model.item_emb.weight.grad is not None


# ── SASRec ────────────────────────────────────────────────────────────────────


class TestSASRec:
    def test_output_shape(self):
        model = SASRec(N_I, dim=DIM, max_len=SEQ_LEN)
        seq = torch.randint(0, N_I + 1, (BATCH, SEQ_LEN))
        out = model(seq)
        assert out.shape == (BATCH, DIM)

    def test_padding_handled(self):
        """All-zero (padding) sequence should not crash."""
        model = SASRec(N_I, dim=DIM, max_len=SEQ_LEN)
        seq = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
        out = model(seq)
        assert out.shape == (BATCH, DIM)

    def test_different_lengths(self):
        model = SASRec(N_I, dim=DIM, max_len=SEQ_LEN)
        seq = torch.randint(0, N_I + 1, (4, SEQ_LEN))
        # Mask first half as padding
        seq[:, : SEQ_LEN // 2] = 0
        out = model(seq)
        assert out.shape == (4, DIM)

    def test_grad_flows(self):
        model = SASRec(N_I, dim=DIM, max_len=SEQ_LEN)
        seq = torch.randint(1, N_I + 1, (BATCH, SEQ_LEN))
        out = model(seq)
        out.sum().backward()
        assert model.item_emb.weight.grad is not None


# ── NCF ───────────────────────────────────────────────────────────────────────


class TestNCF:
    def test_output_shape(self):
        model = NCF(n_genres=N_GENRES, use_text=False, use_tone=False)
        u = torch.randn(BATCH, DIM)
        i = torch.randn(BATCH, DIM)
        d = torch.zeros(BATCH, dtype=torch.long)
        c = torch.randn(BATCH, N_GENRES)
        out = model(u, i, d, c)
        assert out.shape == (BATCH, 1)

    def test_with_text(self):
        TEXT_DIM = 384
        model = NCF(n_genres=N_GENRES, use_text=True, use_tone=False)
        u = torch.randn(BATCH, DIM)
        i = torch.randn(BATCH, DIM)
        d = torch.zeros(BATCH, dtype=torch.long)
        c = torch.randn(BATCH, N_GENRES)
        txt = torch.randn(BATCH, TEXT_DIM)
        out = model(u, i, d, c, item_text_emb=txt)
        assert out.shape == (BATCH, 1)

    def test_with_text_and_tone(self):
        model = NCF(n_genres=N_GENRES, use_text=True, use_tone=True)
        u = torch.randn(BATCH, DIM)
        i = torch.randn(BATCH, DIM)
        d = torch.zeros(BATCH, dtype=torch.long)
        c = torch.randn(BATCH, N_GENRES)
        txt = torch.randn(BATCH, 384)
        tone = torch.randn(BATCH, 6)
        out = model(u, i, d, c, item_text_emb=txt, item_tone=tone)
        assert out.shape == (BATCH, 1)

    def test_domain_embedding_different_domains(self):
        model = NCF(n_genres=N_GENRES, use_text=False, use_tone=False)
        u = torch.randn(BATCH, DIM)
        i = torch.randn(BATCH, DIM)
        d_anime = torch.zeros(BATCH, dtype=torch.long)
        d_movie = torch.ones(BATCH, dtype=torch.long)
        c = torch.randn(BATCH, N_GENRES)
        out_anime = model(u, i, d_anime, c)
        out_movie = model(u, i, d_movie, c)
        # Scores should differ due to different domain embeddings
        assert not torch.allclose(out_anime, out_movie)


# ── GatedFusion ───────────────────────────────────────────────────────────────


class TestGatedFusion:
    def test_output_shape(self):
        fusion = GatedFusion(dim=DIM)
        gnn_u = torch.randn(BATCH, DIM)
        seq_u = torch.randn(BATCH, DIM)
        out = fusion(gnn_u, seq_u)
        assert out.shape == (BATCH, DIM)

    def test_gate_values_in_zero_one(self):
        """Gate activations are sigmoid so output should be a smooth blend."""
        fusion = GatedFusion(dim=DIM)
        gnn_u = torch.ones(BATCH, DIM)
        seq_u = torch.zeros(BATCH, DIM)
        out = fusion(gnn_u, seq_u)
        # Output should be strictly between 0 and 1 (smooth blend)
        assert out.max().item() <= 1.0 + 1e-5
        assert out.min().item() >= -1e-5


# ── AniRecV20 (full model) ────────────────────────────────────────────────────


class TestAniRecV20:
    @pytest.fixture(autouse=True)
    def build_model(self):
        self.model = AniRecV20(N_U, N_I, N_GENRES, use_text=False, use_tone=False)
        A = _random_adj(N_U, N_I)
        self.model.gnn.load_adj(A, target_device=torch.device("cpu"))
        self.gnn_u, self.gnn_i = self.model.gnn()

    def test_get_user_rep_shape(self):
        u_ids = torch.arange(BATCH)
        seqs = torch.randint(0, N_I + 1, (BATCH, SEQ_LEN))
        out = self.model.get_user_rep(u_ids, seqs, self.gnn_u, self.gnn_i)
        assert out.shape == (BATCH, DIM)

    def test_score_shape(self):
        u_ids = torch.arange(BATCH)
        seqs = torch.randint(0, N_I + 1, (BATCH, SEQ_LEN))
        u_rep = self.model.get_user_rep(u_ids, seqs, self.gnn_u, self.gnn_i)
        i_ids = torch.randint(0, N_I, (BATCH,))
        d_ids = torch.zeros(BATCH, dtype=torch.long)
        content = torch.randn(BATCH, N_GENRES)
        sc = self.model.score(u_rep, i_ids, d_ids, content, self.gnn_i)
        assert sc.shape == (BATCH, 1)

    def test_encode_seq_normalized(self):
        seqs = torch.randint(0, N_I + 1, (BATCH, SEQ_LEN))
        out = self.model.encode_seq(seqs)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)

    def test_parameter_count(self):
        n_params = sum(p.numel() for p in self.model.parameters())
        assert n_params > 10_000  # sanity: should have at least 10k params
