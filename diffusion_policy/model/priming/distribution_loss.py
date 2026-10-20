"""Conditional action-space Sinkhorn surrogate for one-step Priming Policy.

Each observation is a separate matching group.  With one expert action
per observation, generated->real OT has a uniform Bx1 plan, so it is
computed directly; generated->independent-reference OT uses Sinkhorn.
This is a *research adaptation* of the Priming Models image loss, not
an action-space configuration validated by the original paper.
"""
import math

import torch
from torch import nn


def _half_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """x [B,K,D], y [B,M,D] -> 0.5 ||x-y||^2 [B,K,M]."""
    return 0.5 * (x.unsqueeze(2) - y.unsqueeze(1)).square().sum(dim=-1)


@torch.no_grad()
def uniform_sinkhorn_plan(cost: torch.Tensor, epsilon: float = 0.05,
                          iterations: int = 30) -> torch.Tensor:
    """Log-domain entropic OT with uniform row and column masses.

    The plan is intentionally detached: gradient flows only through the
    transport costs, matching the Priming Models' cost-only surrogate.
    """
    if cost.ndim != 3:
        raise ValueError("cost must have shape [conditions, live, reference]")
    if epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon must be positive and iterations >= 1")
    if not torch.isfinite(cost).all():
        raise FloatingPointError("non-finite Sinkhorn cost")
    _, n, m = cost.shape
    log_kernel = -cost.detach().float() / float(epsilon)
    log_u = torch.zeros_like(log_kernel[..., 0])
    log_v = torch.zeros_like(log_kernel[..., 0, :])
    for _ in range(iterations):
        log_u = -math.log(n) - torch.logsumexp(
            log_kernel + log_v.unsqueeze(-2), dim=-1)
        log_v = -math.log(m) - torch.logsumexp(
            log_kernel + log_u.unsqueeze(-1), dim=-2)
    plan = torch.exp(log_u.unsqueeze(-1) + log_kernel + log_v.unsqueeze(-2))
    return plan.to(cost.dtype).detach()


class ConditionalActionOTLoss(nn.Module):
    """Attraction to demonstrations minus generated-reference self-interaction.

    Inputs:
       live: [B, K, H, Da] (requires gradient)
       reference: [B, K, H, Da] (independent noise; stop gradient)
       real: [B, 1, H, Da] (one paired expert action per observation)

    Raw actions and temporal action differences are matched in *separate*
    feature spaces. Their scale is the detached RMS pairwise distance
    of active-to-real and active-to-reference features in the minibatch.
    """

    def __init__(self, epsilon=0.05, sinkhorn_iterations=30,
                 delta_weight=0.0, min_feature_scale=1e-3):
        super().__init__()
        if not (0.0 <= delta_weight):
            raise ValueError("delta_weight must be nonnegative")
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.delta_weight = float(delta_weight)
        self.min_feature_scale = float(min_feature_scale)
        if self.min_feature_scale <= 0:
            raise ValueError("min_feature_scale must be positive")

    def _branch_loss(self, live: torch.Tensor, reference: torch.Tensor,
                     real: torch.Tensor):
        # Features are already flattened: [B,K,D], [B,K,D], [B,1,D].
        # The scale is computed from features before normalization, is
        # shared across groups, and does not receive gradients.
        with torch.no_grad():
            raw_cross = _half_squared_distances(live.detach(), real.detach())
            raw_self = _half_squared_distances(live.detach(), reference.detach())
            squared_rms = raw_cross.mean() + raw_self.mean()
            scale = squared_rms.clamp_min(self.min_feature_scale ** 2).sqrt()

        live_scaled = live / scale
        reference_scaled = reference.detach() / scale
        real_scaled = real.detach() / scale
        cross_cost = _half_squared_distances(live_scaled, real_scaled)
        self_cost = _half_squared_distances(live_scaled, reference_scaled)

        # With exactly one real action per condition, cross OT is the
        # uniform mean regardless of entropy regularization.
        attraction = cross_cost.mean()
        plan = uniform_sinkhorn_plan(self_cost, self.epsilon,
                                     self.sinkhorn_iterations)
        repulsion = (plan * self_cost).sum(dim=(-2, -1)).mean()
        return attraction - repulsion, attraction.detach(), repulsion.detach()

    def forward(self, live: torch.Tensor, reference: torch.Tensor,
                real: torch.Tensor):
        if (live.ndim != 4 or reference.shape != live.shape or
                real.shape != (live.shape[0], 1, *live.shape[-2:])):
            raise ValueError("expected live/ref [B,K,H,Da] and real [B,1,H,Da]")
        if live.shape[1] < 2:
            raise ValueError("use at least 2 generated samples per condition")

        B, K = live.shape[:2]
        raw_live = live.reshape(B, K, -1)
        raw_ref = reference.reshape(B, K, -1)
        raw_real = real.reshape(B, 1, -1)
        raw_loss, attraction, repulsion = self._branch_loss(
            raw_live, raw_ref, raw_real)
        total = raw_loss
        details = {
            "ot_raw": raw_loss.detach(),
            "ot_attraction": attraction,
            "ot_repulsion": repulsion,
        }
        if self.delta_weight:
            if live.shape[2] < 2:
                raise ValueError("temporal differences require horizon >= 2")
            diff = lambda x: (x[:, :, 1:] - x[:, :, :-1]).reshape(x.shape[0], x.shape[1], -1)
            delta_loss, _, _ = self._branch_loss(
                diff(live), diff(reference), diff(real))
            total = total + self.delta_weight * delta_loss
            details["ot_delta"] = delta_loss.detach()
        details["ot_total"] = total.detach()
        return total, details
