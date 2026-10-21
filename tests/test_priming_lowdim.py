"""CPU smoke/unit tests; run from the root of the Diffusion Policy fork.

Full policy tests need the original repository's environment (zarr etc.).
"""
import torch
import torch.nn as nn

from diffusion_policy.model.priming.distribution_loss import (
    ConditionalActionOTLoss, uniform_sinkhorn_plan,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.priming_unet_lowdim_policy import PrimingUnetLowdimPolicy


class TinyConditionalGenerator(nn.Module):
    def __init__(self, horizon=16, action_dim=2, cond_dim=40):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, horizon * action_dim),
        )
        self.calls = 0

    def forward(self, sample, timestep, global_cond=None):
        self.calls += 1
        x = torch.cat((sample.flatten(1), global_cond), dim=-1)
        return self.net(x).reshape(-1, self.horizon, self.action_dim)


def make_policy(stage):
    torch.manual_seed(3)
    m = TinyConditionalGenerator()
    policy = PrimingUnetLowdimPolicy(
        model=m, horizon=16, obs_dim=20, action_dim=2,
        n_action_steps=8, n_obs_steps=2, stage=stage,
        samples_per_condition=3, delta_weight=0.25,
        ot_iterations=40,
    )
    normalizer = LinearNormalizer()
    normalizer.fit({
        'obs': torch.randn(32, 16, 20),
        'action': torch.randn(32, 16, 2),
    })
    policy.set_normalizer(normalizer)
    return policy


def batch():
    return {
        'obs': torch.randn(4, 16, 20),
        'action': torch.randn(4, 16, 2),
    }


def test_sinkhorn_uniform_marginals():
    cost = torch.tensor([[[0.02, 0.04, 0.08],
                          [0.04, 0.01, 0.06],
                          [0.04, 0.06, 0.01]]])
    plan = uniform_sinkhorn_plan(cost, epsilon=0.1, iterations=100)
    assert not plan.requires_grad
    assert torch.allclose(plan.sum(-1), torch.full((1, 3), 1/3), atol=1e-5)
    assert torch.allclose(plan.sum(-2), torch.full((1, 3), 1/3), atol=1e-5)


def test_ot_has_finite_grad_and_does_not_update_reference():
    live = torch.randn(2, 3, 8, 2, requires_grad=True)
    ref = torch.randn(2, 3, 8, 2, requires_grad=True)
    real = torch.randn(2, 1, 8, 2)
    loss, details = ConditionalActionOTLoss(delta_weight=0.2)(live, ref, real)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(live.grad).all()
    assert live.grad.abs().sum() > 0
    assert ref.grad is None
    assert 'ot_delta' in details


def test_pretrain_backprop_and_one_step_inference():
    p = make_policy('pretrain')
    loss = p.compute_loss(batch())
    assert torch.isfinite(loss)
    loss.backward()
    assert p.model.net[0].weight.grad is not None
    p.model.calls = 0
    out = p.predict_action({'obs': torch.randn(2, 2, 20)})
    assert p.model.calls == 1  # true one-NFE inference
    assert out['action'].shape == (2, 8, 2)
    assert out['action_pred'].shape == (2, 16, 2)
    # slicing respects OA convention (start = n_obs_steps - 1)
    assert torch.equal(out['action'], out['action_pred'][:, 1:9])


def test_posttrain_backprop_and_reference_no_grad():
    p = make_policy('posttrain')
    loss = p.compute_loss(batch())
    assert torch.isfinite(loss)
    assert p.model.calls == 2  # live and independent reference batches
    loss.backward()
    assert p.model.net[-1].weight.grad is not None
    assert torch.isfinite(p.model.net[-1].weight.grad).all()
    assert 'ot_total' in p.last_loss_metrics

def test_optional_loss_metrics_are_scalar_and_detached():
    """Shared workspace exports Priming metrics without changing baseline training."""
    from types import SimpleNamespace
    from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
        TrainDiffusionUnetLowdimWorkspace,
    )

    metric_tensor = torch.tensor(1.25, requires_grad=True)
    policy = SimpleNamespace(last_loss_metrics={
        'recon_mse': metric_tensor,
        'ot_repulsion': 0.375,
        'non_scalar': torch.ones(3),
    })
    extracted = TrainDiffusionUnetLowdimWorkspace._policy_loss_metrics(
        policy, prefix='train')
    assert extracted == {
        'train/recon_mse': 1.25,
        'train/ot_repulsion': 0.375,
    }
    assert metric_tensor.grad is None
    assert TrainDiffusionUnetLowdimWorkspace._policy_loss_metrics(
        SimpleNamespace(), prefix='val') == {}


def test_priming_stage_metrics_available_for_workspace():
    from diffusion_policy.workspace.train_diffusion_unet_lowdim_workspace import (
        TrainDiffusionUnetLowdimWorkspace,
    )

    pretrain_policy = make_policy('pretrain')
    pretrain_policy.compute_loss(batch())
    pretrain_metrics = TrainDiffusionUnetLowdimWorkspace._policy_loss_metrics(
        pretrain_policy, prefix='train')
    assert 'train/recon_mse' in pretrain_metrics

    posttrain_policy = make_policy('posttrain')
    posttrain_policy.compute_loss(batch())
    posttrain_metrics = TrainDiffusionUnetLowdimWorkspace._policy_loss_metrics(
        posttrain_policy, prefix='val')
    for key in ('ot_raw', 'ot_attraction', 'ot_repulsion', 'ot_total', 'ot_delta'):
        assert 'val/' + key in posttrain_metrics
