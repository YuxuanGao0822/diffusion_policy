import torch
import torch.nn.functional as F
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.priming.distribution_loss import ConditionalActionOTLoss
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

class PrimingUnetImagePolicy(BaseImagePolicy):
    """One-NFE image-conditioned Priming policy for PushT Image."""
    def __init__(self, shape_meta, obs_encoder, horizon=16, n_obs_steps=2,
                 n_action_steps=8, stage="pretrain", noise_level=0.8,
                 samples_per_condition=4, distribution_objective="ot",
                 distribution_window="full", ot_epsilon=0.05,
                 ot_iterations=30, delta_weight=0.0,
                 down_dims=(256,512,1024), diffusion_step_embed_dim=256,
                 kernel_size=5, n_groups=8, cond_predict_scale=True,
                 clip_sample=True):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_dim = int(shape_meta["action"]["shape"][0])
        self.obs_dim = int(obs_encoder.output_shape()[0])
        self.model = ConditionalUnet1D(
            input_dim=self.action_dim, local_cond_dim=None,
            global_cond_dim=self.obs_dim*n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims, kernel_size=kernel_size,
            n_groups=n_groups, cond_predict_scale=cond_predict_scale)
        self.normalizer = LinearNormalizer()
        self.horizon, self.n_obs_steps = int(horizon), int(n_obs_steps)
        self.n_action_steps = int(n_action_steps)
        self.stage, self.noise_level = stage, float(noise_level)
        self.samples_per_condition = int(samples_per_condition)
        self.distribution_window, self.clip_sample = distribution_window, bool(clip_sample)
        self.distribution_loss = ConditionalActionOTLoss(
            epsilon=ot_epsilon, sinkhorn_iterations=ot_iterations,
            delta_weight=delta_weight, objective=distribution_objective)
        self.last_loss_metrics = {}

    def set_normalizer(self, normalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _condition(self, obs):
        b = next(iter(obs.values())).shape[0]
        features = {k: v[:,:self.n_obs_steps].reshape(
            b*self.n_obs_steps,*v.shape[2:]) for k,v in obs.items()}
        return self.obs_encoder(features).reshape(b,-1)

    def _generate(self, noise, global_cond):
        t = torch.zeros(noise.shape[0],device=noise.device,dtype=torch.long)
        return self.model(noise,t,global_cond=global_cond)

    def predict_action(self, obs_dict):
        nobs = self.normalizer.normalize(obs_dict)
        cond = self._condition(nobs)
        z = torch.randn(cond.shape[0], self.horizon, self.action_dim,
                        device=cond.device,dtype=cond.dtype)
        actions = self._generate(z,cond)
        if self.clip_sample:
            actions = actions.clamp(-1,1)
        full = self.normalizer["action"].unnormalize(actions)
        start = self.n_obs_steps-1
        return {"action":full[:,start:start+self.n_action_steps],
                "action_pred":full}

    def compute_loss(self,batch):
        cond = self._condition(self.normalizer.normalize(batch["obs"]))
        real = self.normalizer["action"].normalize(batch["action"])
        if self.stage == "pretrain":
            z = torch.randn_like(real)
            pred = self._generate((1-self.noise_level)*real+self.noise_level*z,cond)
            loss = F.mse_loss(pred,real)
            self.last_loss_metrics = {"recon_mse":loss.detach()}
            return loss
        b,h,da = real.shape
        k = self.samples_per_condition
        cc = cond.repeat_interleave(k,dim=0)
        live = self._generate(torch.randn(b*k,h,da,device=real.device),cc).reshape(b,k,h,da)
        with torch.no_grad():
            ref = self._generate(torch.randn(b*k,h,da,device=real.device),cc).reshape(b,k,h,da)
        if self.distribution_window == "executed":
            sl = slice(self.n_obs_steps-1,self.n_obs_steps-1+self.n_action_steps)
            live,ref,real = live[:,:,sl,:],ref[:,:,sl,:],real[:,sl,:]
        loss,details = self.distribution_loss(live,ref,real.unsqueeze(1))
        self.last_loss_metrics = details
        return loss
