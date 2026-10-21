"""Core networks and correctly-discounted action-chunk RL primitives.

Research implementation, not a reproduction of Policy Decorator or FQL.
Observations are frozen teacher features. Critic actions are normalized
executed chunks, not full predicted horizons.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from diffusion_policy.model.priming.distribution_loss import ConditionalActionOTLoss


def mlp(input_dim, output_dim, hidden=256):
    return nn.Sequential(
        nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, output_dim))


def chunk_return(rewards, gamma):
    """Semi-Markov reward and discount across *actually executed* env steps."""
    assert len(rewards) > 0
    return (sum((gamma ** i)*float(r) for i,r in enumerate(rewards)),
            gamma ** len(rewards))


class Replay:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity, self.index, self.size = int(capacity), 0, 0
        self.arrays = dict(
            state=np.empty((capacity,state_dim), np.float32),
            next_state=np.empty((capacity,state_dim), np.float32),
            action=np.empty((capacity,action_dim), np.float32),
            reward=np.empty((capacity,1), np.float32),
            discount=np.empty((capacity,1), np.float32),
            terminal=np.empty((capacity,1), np.float32))

    def add(self, state, action, reward, next_state, discount, terminal):
        i = self.index
        data = (state,next_state,action,[reward],[discount],[float(terminal)])
        for key,value in zip(self.arrays, data):
            self.arrays[key][i] = np.asarray(value, np.float32)
        self.index = (i+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, count, device, high_reward=False):
        assert self.size > 0, "replay is empty"
        if high_reward and self.size > count:
            rewards = self.arrays["reward"][:self.size,0]
            cutoff = np.quantile(rewards, 0.65)
            candidates = np.flatnonzero(rewards >= cutoff)
            index = np.random.choice(candidates, count, replace=True)
        else:
            index = np.random.randint(self.size, size=count)
        return {k:torch.from_numpy(v[index]).to(device)
                for k,v in self.arrays.items()}

    def state_dict(self):
        return {"capacity": self.capacity, "index":self.index, "size":self.size,
                "arrays":{k:v[:self.size].copy() for k,v in self.arrays.items()}}

    @classmethod
    def from_state_dict(cls, payload):
        ar = payload["arrays"]
        self = cls(payload["capacity"], ar["state"].shape[-1],
                   ar["action"].shape[-1])
        self.index, self.size = payload["index"],payload["size"]
        for k in self.arrays:
            self.arrays[k][:self.size] = ar[k]
        return self


class ResidualActor(nn.Module):
    def __init__(self, state_dim, action_dim, bound=0.1):
        super().__init__()
        self.net = mlp(state_dim+action_dim, 2*action_dim)
        # The first deterministic residual policy must reproduce its
        # frozen teacher exactly; exploration noise is still stochastic.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.action_dim = action_dim
        self.bound = float(bound)

    def forward(self, state, base, deterministic=False):
        x = self.net(torch.cat([state,base],-1))
        mu, log_std = x.chunk(2,-1)
        log_std = log_std.clamp(-5,-1)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        pre = mu if deterministic else dist.rsample()
        v = torch.tanh(pre)
        # Log-prob accounts for tanh and fixed residual scale.
        logp = (dist.log_prob(pre)-torch.log(1-v.square()+1e-6)
                -math.log(max(self.bound,1e-8))).sum(-1,keepdim=True)
        action = (base+self.bound*v).clamp(-1.0,1.0)
        return action, logp

    @torch.no_grad()
    def act(self, state, base, deterministic=True):
        return self.forward(state,base,deterministic)[0]


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = mlp(state_dim+action_dim,1)
        self.q2 = mlp(state_dim+action_dim,1)

    def forward(self,s,a):
        x=torch.cat([s,a],-1)
        return self.q1(x),self.q2(x)

    def minimum(self,s,a):
        return torch.minimum(*self(s,a))


class TaskDynamics(nn.Module):
    """Predict 32-D fixed state projection + chunk reward from (s, action)."""
    def __init__(self,state_dim,action_dim,feature_dim=32,seed=42):
        super().__init__()
        rng=np.random.RandomState(seed)
        projection=rng.randn(state_dim,feature_dim).astype(np.float32)/math.sqrt(state_dim)
        self.register_buffer("projection",torch.from_numpy(projection))
        self.net=mlp(state_dim+action_dim,feature_dim+1)
        self.feature_dim=feature_dim

    def target(self,next_state,reward):
        return torch.cat([next_state @ self.projection,reward],-1)

    def forward(self,state,action):
        return self.net(torch.cat([state,action],-1))

    def loss(self,batch):
        expected=self.target(batch["next_state"],batch["reward"]).detach()
        predicted=self(batch["state"],batch["action"])
        # Squared standardized state features and reward.
        return F.mse_loss(predicted,expected)


def task_aware_ot(state, generated, reference, expert, next_state, reward,
                  dynamics, weight=0.25, epsilon=0.05, iterations=50):
    """Transport in joint executed action and *predicted task outcome* space.

    The real outcome is the environment-observed next state/reward, not
    model-imagined expert outcomes. Reference branch is stop-gradient.
    """
    b,k,d=generated.shape
    s=state[:,None,:].expand(-1,k,-1).reshape(b*k,-1)
    gen_task=dynamics(s,generated.reshape(b*k,d)).reshape(b,k,-1)
    with torch.no_grad():
        ref_task=dynamics(s,reference.reshape(b*k,d)).reshape(b,k,-1)
        actual_task=dynamics.target(next_state,reward)
    # Convert D-dimensional task features to per-channel RMS scale.
    f=gen_task.shape[-1]
    weight=float(weight)/math.sqrt(f)
    g=torch.cat([generated,weight*gen_task],-1).unsqueeze(2)
    r=torch.cat([reference,weight*ref_task],-1).unsqueeze(2)
    y=torch.cat([expert,weight*actual_task],-1).unsqueeze(1).unsqueeze(2)
    loss_fn=ConditionalActionOTLoss(epsilon=epsilon,sinkhorn_iterations=iterations)
    return loss_fn(g,r,y)[0]


def soft_target_update(target,source,tau=0.005):
    with torch.no_grad():
        for a,b in zip(target.parameters(),source.parameters()):
            a.lerp_(b,tau)


def sac_critic_step(q,target_q,actor,batch,next_base,optimizer,alpha=0.005,
                    tau=0.005):
    s,a,r,s2,gamma,done=(batch[k] for k in
        ("state","action","reward","next_state","discount","terminal"))
    with torch.no_grad():
        next_a,next_logp=actor(s2,next_base)
        target=r+gamma*(1-done)*(target_q.minimum(s2,next_a)-alpha*next_logp)
    q1,q2=q(s,a)
    loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    soft_target_update(target_q,q,tau)
    return float(loss.detach())


def residual_actor_step(actor,q,batch,base,optimizer,alpha=0.005):
    for p in q.parameters():
        p.requires_grad_(False)
    a,logp=actor(batch["state"],base)
    loss=(alpha*logp-q.minimum(batch["state"],a)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    for p in q.parameters():
        p.requires_grad_(True)
    return float(loss.detach())
