"""Training loss functions for AniRec v22."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Hyper-parameters (imported from config at runtime; defaults here) ─────────
_CL_TEMP = 0.07
_CL_LAMBDA = 0.40
_VICREG_LAMBDA = 0.02
_VICREG_GAMMA = 1.0


def weighted_bpr_loss(
    s_pos: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Weighted Bayesian Personalised Ranking loss.

    Args:
        s_pos:      (B,) positive item scores.
        neg_scores: (B, N_NEG) negative item scores.
        weights:    (B,) per-interaction importance weights.
    Returns:
        Scalar loss.
    """
    w = weights.unsqueeze(1)
    return -(w * F.logsigmoid(s_pos.unsqueeze(1) - neg_scores)).mean()


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temp: float = _CL_TEMP,
) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) contrastive loss with cosine similarity.

    EMB-1/5/7: Explicit L2-normalisation; cosine sim used throughout.

    Args:
        z1, z2: (B, D) embedding pairs.
        temp:   Temperature scaling factor.
    Returns:
        Scalar loss.
    """
    B = z1.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
    z1n = F.normalize(z1, dim=-1)
    z2n = F.normalize(z2, dim=-1)
    sim = torch.mm(z1n, z2n.T) / temp
    labels = torch.arange(B, device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2.0


def semantic_cl_loss(
    item_embs: torch.Tensor,
    sem_pairs: torch.Tensor,
    temp: float = _CL_TEMP,
) -> torch.Tensor:
    """
    EMB-3: InfoNCE over pre-mined semantic positive pairs.

    Args:
        item_embs: (N_items, D) item embedding matrix.
        sem_pairs: (K, 2) int tensor of (item_i, item_j) positive pairs.
        temp:      Temperature.
    Returns:
        Scalar loss.
    """
    if sem_pairs.shape[0] < 2:
        return torch.tensor(0.0, device=item_embs.device, dtype=item_embs.dtype)
    idx_i = sem_pairs[:, 0]
    idx_j = sem_pairs[:, 1]
    z_i = item_embs[idx_i]
    z_j = item_embs[idx_j]
    return info_nce_loss(z_i, z_j, temp=temp)


def vicreg_variance_loss(
    z: torch.Tensor,
    gamma: float = _VICREG_GAMMA,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    EMB-4: VICReg variance regularisation term.

    Penalises dimensions whose standard deviation falls below `gamma`,
    preventing embedding collapse.

    Args:
        z:     (B, D) batch of embeddings.
        gamma: Target minimum std per dimension.
        eps:   Numerical stability constant.
    Returns:
        Scalar loss.
    """
    z_norm = z - z.mean(dim=0)
    std = torch.sqrt(z_norm.var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def combined_training_loss(
    s_pos: torch.Tensor,
    neg_scores: torch.Tensor,
    weights: torch.Tensor,
    gnn_u_pos: torch.Tensor,
    sasrec_pos: torch.Tensor,
    item_embs: torch.Tensor,
    sem_pairs_batch: Optional[torch.Tensor],
    cl_lambda: float = _CL_LAMBDA,
    vicreg_lambda: float = _VICREG_LAMBDA,
    cl_temp: float = _CL_TEMP,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Full combined training loss:
        BPR  +  user_CL  +  0.5 × item_CL  +  VICReg variance

    Args:
        s_pos:           (B,) positive scores.
        neg_scores:      (B, N_NEG) negative scores.
        weights:         (B,) sample weights.
        gnn_u_pos:       (B, D) L2-normalised GNN user embeddings.
        sasrec_pos:      (B, D) L2-normalised SASRec user embeddings.
        item_embs:       (N_items, D) L2-normalised item embeddings.
        sem_pairs_batch: (K, 2) semantic positive pairs, or None.
        cl_lambda:       Weight for contrastive terms.
        vicreg_lambda:   Weight for variance regularisation.
        cl_temp:         InfoNCE temperature.
    Returns:
        (total_loss, dict of component values)
    """
    loss_bpr = weighted_bpr_loss(s_pos, neg_scores, weights)
    loss_cl_user = info_nce_loss(gnn_u_pos, sasrec_pos, temp=cl_temp)
    loss_cl_item = torch.tensor(0.0, device=s_pos.device, dtype=s_pos.dtype)
    if sem_pairs_batch is not None and sem_pairs_batch.shape[0] >= 2:
        loss_cl_item = semantic_cl_loss(item_embs, sem_pairs_batch, temp=cl_temp)
    loss_var = vicreg_variance_loss(item_embs)

    total = loss_bpr + cl_lambda * (loss_cl_user + 0.5 * loss_cl_item) + vicreg_lambda * loss_var
    return total, {
        "bpr": loss_bpr.item(),
        "cl_user": loss_cl_user.item(),
        "cl_item": loss_cl_item.item(),
        "variance": loss_var.item(),
        "total": total.item(),
    }
