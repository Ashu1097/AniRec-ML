"""Tests for training loss functions."""

import torch
import torch.nn.functional as F

from src.training.losses import (
    combined_training_loss,
    info_nce_loss,
    semantic_cl_loss,
    vicreg_variance_loss,
    weighted_bpr_loss,
)


class TestWeightedBprLoss:
    def test_basic_shape(self):
        s_pos = torch.randn(8)
        s_neg = torch.randn(8, 4)
        w = torch.ones(8)
        loss = weighted_bpr_loss(s_pos, s_neg, w)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_positive_dominates(self):
        """When positive scores >> negative, loss should be near zero."""
        s_pos = torch.ones(16) * 5.0
        s_neg = torch.ones(16, 4) * -5.0
        w = torch.ones(16)
        loss = weighted_bpr_loss(s_pos, s_neg, w)
        assert loss.item() < 0.01

    def test_negative_dominates(self):
        """When negative scores >> positive, loss should be high."""
        s_pos = torch.ones(16) * -5.0
        s_neg = torch.ones(16, 4) * 5.0
        w = torch.ones(16)
        loss = weighted_bpr_loss(s_pos, s_neg, w)
        assert loss.item() > 4.0

    def test_weights_scale_loss(self):
        s_pos = torch.zeros(8)
        s_neg = torch.zeros(8, 4)
        w_high = torch.ones(8) * 3.0
        w_low = torch.ones(8) * 0.5
        loss_high = weighted_bpr_loss(s_pos, s_neg, w_high)
        loss_low = weighted_bpr_loss(s_pos, s_neg, w_low)
        assert loss_high.item() > loss_low.item()

    def test_gradient_flows(self):
        s_pos = torch.randn(8, requires_grad=True)
        s_neg = torch.randn(8, 4, requires_grad=True)
        w = torch.ones(8)
        loss = weighted_bpr_loss(s_pos, s_neg, w)
        loss.backward()
        assert s_pos.grad is not None
        assert s_neg.grad is not None


class TestInfoNceLoss:
    def test_perfect_alignment(self):
        """Identical z1 and z2 should give low loss."""
        z = F.normalize(torch.randn(16, 64), dim=-1)
        loss = info_nce_loss(z, z.clone())
        assert loss.item() < 0.1

    def test_random_alignment(self):
        """Random z1 and z2 should give high loss (near log(B))."""
        z1 = F.normalize(torch.randn(32, 64), dim=-1)
        z2 = F.normalize(torch.randn(32, 64), dim=-1)
        loss = info_nce_loss(z1, z2)
        # Expect ≈ log(32) ≈ 3.47
        assert loss.item() > 2.0

    def test_batch_size_1_returns_zero(self):
        z1 = torch.randn(1, 64)
        z2 = torch.randn(1, 64)
        loss = info_nce_loss(z1, z2)
        assert loss.item() == 0.0

    def test_temperature_scaling(self):
        """Lower temperature should increase loss on random pairs."""
        z1 = F.normalize(torch.randn(16, 64), dim=-1)
        z2 = F.normalize(torch.randn(16, 64), dim=-1)
        loss_low_temp = info_nce_loss(z1, z2, temp=0.05)
        loss_high_temp = info_nce_loss(z1, z2, temp=0.5)
        assert loss_low_temp.item() > loss_high_temp.item()

    def test_symmetric(self):
        """Loss should be the same regardless of which pair is z1 vs z2."""
        z1 = F.normalize(torch.randn(8, 32), dim=-1)
        z2 = F.normalize(torch.randn(8, 32), dim=-1)
        loss_fwd = info_nce_loss(z1, z2)
        loss_bwd = info_nce_loss(z2, z1)
        assert abs(loss_fwd.item() - loss_bwd.item()) < 1e-5


class TestVicregVarianceLoss:
    def test_collapsed_embeddings_penalised(self):
        """Constant embeddings (all same) should yield high variance loss."""
        z = torch.ones(32, 64)
        loss = vicreg_variance_loss(z, gamma=1.0)
        assert loss.item() > 0.5

    def test_spread_embeddings_low_loss(self):
        """Embeddings with std >> gamma should give near-zero loss."""
        z = torch.randn(128, 64) * 5.0
        loss = vicreg_variance_loss(z, gamma=1.0)
        assert loss.item() < 0.1

    def test_gradient_flows(self):
        z = torch.randn(32, 64, requires_grad=True)
        loss = vicreg_variance_loss(z)
        loss.backward()
        assert z.grad is not None


class TestSemanticClLoss:
    def test_basic(self):
        item_embs = F.normalize(torch.randn(100, 64), dim=-1)
        pairs = torch.randint(0, 100, (20, 2))
        # Ensure no pair has i == j
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        if len(pairs) >= 2:
            loss = semantic_cl_loss(item_embs, pairs)
            assert loss.item() >= 0.0

    def test_too_few_pairs_returns_zero(self):
        item_embs = torch.randn(50, 32)
        pairs = torch.zeros(1, 2, dtype=torch.long)
        loss = semantic_cl_loss(item_embs, pairs)
        assert loss.item() == 0.0


class TestCombinedTrainingLoss:
    def test_combined_returns_all_components(self):
        B, D, N_NEG = 8, 64, 4
        s_pos = torch.randn(B)
        s_neg = torch.randn(B, N_NEG)
        w = torch.ones(B)
        gnn_u = F.normalize(torch.randn(B, D), dim=-1)
        sasr = F.normalize(torch.randn(B, D), dim=-1)
        item_embs = F.normalize(torch.randn(200, D), dim=-1)
        pairs = torch.randint(0, 200, (30, 2))

        total, parts = combined_training_loss(s_pos, s_neg, w, gnn_u, sasr, item_embs, pairs)
        assert total.item() > 0
        for key in ("bpr", "cl_user", "cl_item", "variance", "total"):
            assert key in parts
            assert isinstance(parts[key], float)

    def test_no_sem_pairs(self):
        B, D, N_NEG = 8, 64, 4
        s_pos = torch.randn(B)
        s_neg = torch.randn(B, N_NEG)
        w = torch.ones(B)
        gnn_u = F.normalize(torch.randn(B, D), dim=-1)
        sasr = F.normalize(torch.randn(B, D), dim=-1)
        item_embs = F.normalize(torch.randn(50, D), dim=-1)

        total, parts = combined_training_loss(
            s_pos, s_neg, w, gnn_u, sasr, item_embs, sem_pairs_batch=None
        )
        assert total.item() > 0
        assert parts["cl_item"] == 0.0

    def test_backward_pass(self):
        B, D, N_NEG = 4, 32, 2
        s_pos = torch.randn(B, requires_grad=True)
        s_neg = torch.randn(B, N_NEG, requires_grad=True)
        w = torch.ones(B)
        gnn_u = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        sasr = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
        item_embs = F.normalize(torch.randn(20, D), dim=-1).requires_grad_(True)

        total, _ = combined_training_loss(
            s_pos, s_neg, w, gnn_u, sasr, item_embs, sem_pairs_batch=None
        )
        total.backward()
        assert s_pos.grad is not None
        assert gnn_u.grad is not None
