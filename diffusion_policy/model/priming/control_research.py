"""Shared PushT chunk-level online RL primitives for Priming research.

Strictly separates training seeds, model-selection seeds and held-out seeds.
Each transition stores actual discounted per-step reward and chunk duration.
State is a frozen Priming observation embedding: image encoder stays frozen.
"""
import collections
import copy
import dill
import math
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(checkpoint, device):
    # ONLY load locally produced, trusted dill checkpoints.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    policy = hydra.utils.instantiate(cfg.policy)
    if "ema_model" not in payload["state_dicts"]:
        raise ValueError("teacher checkpoint has no ema_model")
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy.to(device).eval().requires_grad_(False)
    return policy, cfg


def _stack_history(history, image):
    if image:
        return {key: np.stack([x[key] for x in history]).astype(np.float32)
                for key in history[-1]}
    return np.stack([x[:20] for x in history]).astype(np.float32)


class PushTChunkEnv:
    """Uses raw Gym env to preserve rewards from ALL steps in a chunk."""
    def __init__(self, modality, n_obs_steps=2, n_action_steps=8,
                 max_steps=300, gamma=0.99):
        if modality == "lowdim":
            from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
            kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
            self.env = PushTKeypointsEnv(legacy=True, keypoint_visible_rate=1.,
                                          agent_keypoints=False, **kwargs)
        elif modality == "image":
            from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
            self.env = PushTImageEnv(legacy=True, render_size=96)
        else:
            raise ValueError("modality must be lowdim or image")
        self.image = modality == "image"
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.gamma = gamma
        self.steps = 0
        self.history = collections.deque(maxlen=n_obs_steps)

    def reset(self, seed):
        self.env.seed(int(seed))
        obs = self.env.reset()
        self.steps = 0
        self.history = collections.deque([obs]*self.n_obs_steps,
                                          maxlen=self.n_obs_steps)
        return _stack_history(self.history, self.image)

    def step(self, action_chunk):
        if len(action_chunk) != self.n_action_steps:
            raise ValueError("action chunk size mismatch")
        discounted, max_reward, duration, terminated = 0., 0., 0, False
        for action in action_chunk:
            obs, reward, done, _ = self.env.step(np.asarray(action,dtype=np.float32))
            self.history.append(obs)
            discounted += (self.gamma ** duration) * float(reward)
            max_reward = max(max_reward, float(reward))
            duration += 1
            self.steps += 1
            terminated = bool(done)
            if terminated or self.steps >= self.max_steps:
                break
        return (_stack_history(self.history,self.image), discounted,
                bool(terminated or self.steps >= self.max_steps),
                duration, max_reward)

    def close(self):
        self.env.close()


def encode_observation(teacher, observation, device):
    if isinstance(observation, dict):
        tensors = {k:torch.from_numpy(v[None]).to(device) for k,v in observation.items()}
        normalized = teacher.normalizer.normalize(tensors)
    else:
        normalized = teacher.normalizer["obs"].normalize(
            torch.from_numpy(observation[None]).to(device))
    with torch.no_grad():
        return teacher._condition(normalized).detach()


class SmallMLP(nn.Module):
    def __init__(self, input_dim, output_dim, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,width),
                 nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,output_dim))

    def forward(self,x):
        return self.net(x)


class Actor(nn.Module):
    """Freeze the pre-trained teacher. Train either a bounded residual or a copy of its U-Net."""
    def __init__(self, teacher, phase="residual", residual_scale=0.1):
        super().__init__()
        self.teacher = teacher
        self.teacher.requires_grad_(False)
        self.phase = phase
        self.residual_scale = residual_scale
        self.start = teacher.n_obs_steps-1
        self.steps = teacher.n_action_steps
        self.horizon = teacher.horizon
        self.action_dim = teacher.action_dim
        # frozen teacher may be image conditioned; feature dim inferred outside.
        self.trainable = None

    def initialize(self, condition_dim):
        if self.phase == "residual":
            self.trainable = SmallMLP(
                condition_dim+self.steps*self.action_dim,
                self.steps*self.action_dim)
        elif self.phase in ("q","task_ot"):
            self.trainable = copy.deepcopy(self.teacher.model)
            self.trainable.requires_grad_(True)
        elif self.phase != "teacher":
            raise ValueError(self.phase)
        return self

    def _generate_with(self, network, cond, noise):
        t = torch.zeros(cond.shape[0],device=cond.device,dtype=torch.long)
        return network(noise,t,global_cond=cond)

    def actions(self, cond, noise, network=None):
        with torch.no_grad():
            base = self.teacher._generate(noise,cond)[:,self.start:self.start+self.steps]
        if self.phase == "teacher":
            return base.clamp(-1,1),base.clamp(-1,1)
        net = self.trainable if network is None else network
        if self.phase == "residual":
            residual = net(torch.cat([cond,base.reshape(cond.shape[0],-1)],dim=-1))
            residual = self.residual_scale * torch.tanh(
                residual.reshape(cond.shape[0],self.steps,self.action_dim))
            output = base + residual
        else:
            output = self._generate_with(net,cond,noise)[:,self.start:self.start+self.steps]
        return output.clamp(-1,1),base.clamp(-1,1)


class TwinQ(nn.Module):
    def __init__(self, condition_dim, steps=8, action_dim=2):
        super().__init__()
        inp = condition_dim + steps*action_dim
        self.q1 = SmallMLP(inp,1)
        self.q2 = SmallMLP(inp,1)

    def forward(self,cond,action):
        x = torch.cat([cond,action.reshape(cond.shape[0],-1)],dim=-1)
        return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)


class Dynamics(nn.Module):
    """Frozen-feature transition model trained from online chunk transitions."""
    def __init__(self,condition_dim,steps=8,action_dim=2):
        super().__init__()
        self.net = SmallMLP(condition_dim+steps*action_dim,condition_dim)

    def forward(self,cond,action):
        x = torch.cat([cond,action.reshape(cond.shape[0],-1)],dim=-1)
        return cond + self.net(x)


class Replay:
    def __init__(self,capacity=50000):
        self.transitions=collections.deque(maxlen=capacity)

    def add(self,cond,action,reward,next_cond,done,duration):
        self.transitions.append((cond.squeeze(0).detach().cpu().float(),
                   action.squeeze(0).detach().cpu().float(),float(reward),
                   next_cond.squeeze(0).detach().cpu().float(),
                   bool(done),int(duration)))

    def __len__(self):
        return len(self.transitions)

    def sample(self,n,device):
        elements=random.sample(self.transitions,n)
        c,a,r,nc,d,k=zip(*elements)
        return (torch.stack(c).to(device),torch.stack(a).to(device),
                torch.tensor(r,device=device,dtype=torch.float32),
                torch.stack(nc).to(device),
                torch.tensor(d,device=device,dtype=torch.float32),
                torch.tensor(k,device=device,dtype=torch.float32))


def sample_noise(actor,b,device):
    return torch.randn(b,actor.horizon,actor.action_dim,device=device)


@torch.no_grad()
def evaluate(actor,teacher,modality,device,seeds,max_steps=300,gamma=0.99):
    env=PushTChunkEnv(modality,teacher.n_obs_steps,teacher.n_action_steps,
                      max_steps,gamma)
    rewards=[]
    try:
        for seed in seeds:
            # Same seed -> same environment init + Gaussian noise across methods.
            set_seed(int(seed))
            obs=env.reset(seed)
            best=0.
            done=False
            while not done:
                cond=encode_observation(teacher,obs,device)
                action,_=actor.actions(cond,sample_noise(actor,1,device))
                physical=teacher.normalizer["action"].unnormalize(action)
                obs,_,done,_,step_max=env.step(physical[0].cpu().numpy())
                best=max(best,step_max)
            rewards.append(best)
    finally:
        env.close()
    return {"mean_score":float(np.mean(rewards)),
            "per_seed":{str(s):float(r) for s,r in zip(seeds,rewards)}}


def soft_update(source,target,tau=0.005):
    with torch.no_grad():
        for p,q in zip(source.parameters(),target.parameters()):
            q.mul_(1-tau).add_(p,alpha=tau)


def task_ot_features(dynamics,condition,actions,effect_scale):
    """[B,K,T,Da] -> [B,K,1,T*Da + latent_dim]."""
    b,k,t,da = actions.shape
    ctx=condition[:,None,:].expand(b,k,-1).reshape(b*k,-1)
    acts=actions.reshape(b*k,t,da)
    # Use learned predicted transition, in units scaled by 1/sqrt(feature dim).
    effect = dynamics(ctx,acts) - ctx
    d=effect.shape[-1]
    return torch.cat([acts.reshape(b*k,-1),
                      effect_scale*effect/math.sqrt(max(d,1))],dim=-1).reshape(b,k,1,-1)
