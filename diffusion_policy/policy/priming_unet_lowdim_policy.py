"""One-step conditional action generator for PushT lowdim.

Stage 1: MSE denoising of the *clean action* at fixed corruption s.
Stage 2: direct noise->action conditional distribution matching.
No diffusion scheduler is used at inference or during training.
"""
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.priming.distribution_loss import ConditionalActionOTLoss
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class PrimingUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, model: ConditionalUnet1D, horizon: int,
                 obs_dim: int, action_dim: int, n_action_steps: int,
                 n_obs_steps: int, stage: str = "pretrain",
                 noise_level: float = 0.8, samples_per_condition: int = 4,
                 ot_epsilon: float = 0.05, ot_iterations: int = 30,
                 delta_weight: float = 0.0,
                 distribution_objective: str = "ot",
                 distribution_window: str = "full",
                 clip_sample: bool = True,
                 oa_step_convention: bool = True):
        super().__init__()
        if stage not in ("pretrain", "posttrain"):
            raise ValueError("stage must be 'pretrain' or 'posttrain'")
        if not (0.0 < noise_level < 1.0):
            raise ValueError("noise_level must be between 0 and 1")
        if samples_per_condition < 2:
            raise ValueError("samples_per_condition must be >= 2")
        if distribution_window not in ("full", "executed"):
            raise ValueError("distribution_window must be 'full' or 'executed'")
        if n_obs_steps < 1 or n_action_steps < 1:
            raise ValueError("n_obs_steps and n_action_steps must be positive")
        start = n_obs_steps - 1 if oa_step_convention else n_obs_steps
        if start + n_action_steps > horizon:
            raise ValueError("executed action window exceeds prediction horizon")
        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = int(horizon)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.n_action_steps = int(n_action_steps)
        self.n_obs_steps = int(n_obs_steps)
        self.stage = stage
        self.noise_level = float(noise_level)
        self.samples_per_condition = int(samples_per_condition)
        self.distribution_window = distribution_window
        self.clip_sample = bool(clip_sample)
        self.oa_step_convention = bool(oa_step_convention)
        self.distribution_loss = ConditionalActionOTLoss(
            epsilon=ot_epsilon, sinkhorn_iterations=ot_iterations,
            delta_weight=delta_weight, objective=distribution_objective)
        self.last_loss_metrics = {}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _condition(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or obs.shape[1] < self.n_obs_steps or obs.shape[2] != self.obs_dim:
            raise ValueError("obs must have shape [B, To>=n_obs_steps, obs_dim]")
        return obs[:, :self.n_obs_steps, :].reshape(obs.shape[0], -1)

    def _generate(self, noise: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # Reuse the diffusion U-Net with fixed time embedding (no diffusion steps).
        fixed_t = torch.zeros(noise.shape[0], device=noise.device, dtype=torch.long)
        return self.model(noise, fixed_t, global_cond=global_cond)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")
        nobs = self.normalizer["obs"].normalize(obs_dict["obs"])
        global_cond = self._condition(nobs)
        B = nobs.shape[0]
        z = torch.randn(B, self.horizon, self.action_dim,
                        dtype=nobs.dtype, device=nobs.device)
        sample = self._generate(z, global_cond)
        if self.clip_sample:
            sample = sample.clamp(-1.0, 1.0)
        action_pred = self.normalizer["action"].unnormalize(sample)
        start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
        action = action_pred[:, start:start+self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "valid_mask" in batch:
            raise NotImplementedError("valid_mask is not supported by PushT lowdim")
        nobs = self.normalizer["obs"].normalize(batch["obs"])
        action = self.normalizer["action"].normalize(batch["action"])
        if action.ndim != 3 or action.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError("action must have shape [B, horizon, action_dim]")
        condition = self._condition(nobs)
        if self.stage == "pretrain":
            z = torch.randn_like(action)
            noisy = (1.0 - self.noise_level) * action + self.noise_level * z
            pred = self._generate(noisy, condition)
            loss = F.mse_loss(pred, action)
            self.last_loss_metrics = {"recon_mse": loss.detach()}
            return loss

        B, H, Da = action.shape
        K = self.samples_per_condition
        repeated_cond = condition.repeat_interleave(K, dim=0)
        live_noise = torch.randn(B*K, H, Da, device=action.device, dtype=action.dtype)
        ref_noise = torch.randn_like(live_noise)  # independent reference noises
        live = self._generate(live_noise, repeated_cond).reshape(B, K, H, Da)
        with torch.no_grad():
            reference = self._generate(ref_noise, repeated_cond).reshape(B, K, H, Da)
        # Only a subset of the prediction horizon reaches the robot during
        # receding-horizon control. Keep full-horizon matching as the default
        # for regression compatibility, but enable an executed-action ablation.
        if self.distribution_window == "executed":
            start = self.n_obs_steps - 1 if self.oa_step_convention else self.n_obs_steps
            window = slice(start, start + self.n_action_steps)
            live = live[:, :, window, :]
            reference = reference[:, :, window, :]
            action = action[:, window, :]
        loss, metrics = self.distribution_loss(live, reference, action.unsqueeze(1))
        self.last_loss_metrics = metrics
        return loss
