"""One-step Priming policy for PushT RGB image + proprioception.
Independent image Stage-1 initialization; lowdim weights are not transferable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.common.pytorch_util import dict_apply


class PrimingUnetImagePolicy(BaseImagePolicy):
    def __init__(self, obs_encoder, shape_meta, horizon=16, n_obs_steps=2,
                 n_action_steps=8, noise_level=0.8, clip_sample=True,
                 diffusion_step_embed_dim=128, down_dims=(256, 512, 1024),
                 kernel_size=5, n_groups=8, cond_predict_scale=True):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.obs_feature_dim = int(obs_encoder.output_shape()[0])
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.n_obs_steps = int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.horizon = int(horizon)
        self.noise_level = float(noise_level)
        self.clip_sample = bool(clip_sample)
        self.oa_step_convention = True
        assert self.n_obs_steps-1+self.n_action_steps <= self.horizon
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim, local_cond_dim=None,
            global_cond_dim=self.obs_feature_dim * self.n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=list(down_dims), kernel_size=kernel_size,
            n_groups=n_groups, cond_predict_scale=cond_predict_scale)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def encode_condition(self, obs):
        nobs = self.normalizer.normalize(obs)
        first = next(iter(nobs.values()))
        b = first.shape[0]
        to = self.n_obs_steps
        sliced = dict_apply(nobs, lambda x: x[:, :to].reshape(-1, *x.shape[2:]))
        features = self.obs_encoder(sliced)
        return features.reshape(b, -1)

    def _generate(self, noise, global_cond):
        t = torch.zeros(noise.shape[0], dtype=torch.long, device=noise.device)
        return self.model(noise, t, global_cond=global_cond)

    def predict_action(self, obs_dict):
        cond = self.encode_condition(obs_dict)
        z = torch.randn(cond.shape[0], self.horizon, self.action_dim,
                        device=cond.device, dtype=cond.dtype)
        pred = self._generate(z, cond)
        if self.clip_sample:
            pred = pred.clamp(-1.0, 1.0)
        actions = self.normalizer["action"].unnormalize(pred)
        start = self.n_obs_steps - 1
        return {
            "action": actions[:, start:start+self.n_action_steps],
            "action_pred": actions,
        }

    def compute_loss(self, batch):
        na = self.normalizer["action"].normalize(batch["action"])
        cond = self.encode_condition(batch["obs"])
        z = torch.randn_like(na)
        noisy = (1-self.noise_level) * na + self.noise_level * z
        return F.mse_loss(self._generate(noisy, cond), na)
