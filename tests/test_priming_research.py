"""CPU smoke tests for sequential reward-guided Priming experiments."""
import math
import numpy as np
import pytest
import torch
from torch import nn

from diffusion_policy.model.priming.reward_refinement import (
    Replay, ResidualActor, TwinQ, TaskDynamics, chunk_return,
    sac_critic_step, residual_actor_step, task_aware_ot,
)
from diffusion_policy.policy.priming_unet_image_policy import PrimingUnetImagePolicy
from scripts.run_priming_research import Controller, episode


def test_chunk_reward_uses_actual_executed_steps():
    r, discount = chunk_return([1., 2., 3.], .9)
    assert r == pytest.approx(1 + .9*2 + .9**2*3)
    assert discount == pytest.approx(.9**3)


def test_replay_roundtrip_after_wrapping():
    r = Replay(3, 5, 4)
    for i in range(5):
        r.add(np.full(5, i), np.full(4, i), i+1,
              np.full(5, -i), .98, i%2==0)
    assert r.size == 3 and r.index == 2
    restore = Replay.from_state_dict(r.state_dict())
    assert restore.size == r.size and restore.index == r.index
    assert np.array_equal(restore.arrays["state"], r.arrays["state"])
    assert restore.sample(4, "cpu")["action"].shape == (4,4)


def test_sac_updates_and_bounded_residual():
    torch.manual_seed(7)
    actor = ResidualActor(5, 4, bound=.1)
    critic = TwinQ(5,4)
    target = TwinQ(5,4)
    target.load_state_dict(critic.state_dict())
    batch = {
        "state": torch.randn(8,5),
        "next_state": torch.randn(8,5),
        "action": torch.randn(8,4).clamp(-1,1),
        "reward": torch.randn(8,1),
        "discount": torch.full((8,1), .99**8),
        "terminal": torch.zeros(8,1),
    }
    base = torch.zeros(8,4)
    a,logp = actor(batch["state"], base)
    assert torch.isfinite(logp).all()
    assert a.abs().max() <= .1 + 1e-6
    before = next(critic.parameters()).detach().clone()
    optq = torch.optim.Adam(critic.parameters(),lr=.001)
    opta = torch.optim.Adam(actor.parameters(),lr=.001)
    qloss = sac_critic_step(critic,target,actor,batch,base,optq)
    aloss = residual_actor_step(actor,critic,batch,base,opta)
    assert math.isfinite(qloss) and math.isfinite(aloss)
    assert not torch.equal(before,next(critic.parameters()))


def test_dynamics_outcome_ot_backward():
    torch.manual_seed(13)
    model = TaskDynamics(6,4,feature_dim=3)
    state = torch.randn(2,6)
    live = torch.randn(2,4,4,requires_grad=True)
    reference = torch.randn(2,4,4,requires_grad=True)
    real = torch.randn(2,4)
    future = torch.randn(2,6)
    reward = torch.rand(2,1)
    loss = task_aware_ot(state,live,reference,real,future,reward,model)
    assert torch.isfinite(loss)
    loss.backward()
    assert live.grad is not None and live.grad.abs().sum() > 0
    assert reference.grad is None
    assert model.loss(dict(state=state,action=real,
                           next_state=future,reward=reward)).isfinite()


class TinyEncoder(nn.Module):
    def output_shape(self):
        return (5,)

    def forward(self, obs):
        return torch.cat([obs["image"].mean((-1,-2)),
                          obs["agent_pos"]],dim=-1)


def test_priming_image_policy_one_step_loss_and_actions():
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    torch.manual_seed(5)
    policy = PrimingUnetImagePolicy(
        obs_encoder=TinyEncoder(),shape_meta={"action":{"shape":[2]}},
        horizon=16,n_obs_steps=2,n_action_steps=8,
        diffusion_step_embed_dim=32,down_dims=(32,64),
        kernel_size=3,n_groups=8)
    normalizer=LinearNormalizer()
    normalizer.fit({"agent_pos":torch.randn(16,2,2),
                    "action":torch.randn(16,16,2)})
    normalizer["image"]=get_image_range_normalizer()
    policy.set_normalizer(normalizer)
    batch={"obs":{"image":torch.rand(3,2,3,12,12),
                  "agent_pos":torch.randn(3,2,2)},
           "action":torch.randn(3,16,2)}
    loss=policy.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in policy.model.parameters())
    output=policy.predict_action(batch["obs"])
    assert output["action"].shape==(3,8,2)
    assert output["action_pred"].shape==(3,16,2)


class Identity:
    def normalize(self,x): return x
    def unnormalize(self,x): return x


class TinyUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.zeros(1))
    def forward(self,z,t,global_cond=None):
        return z*0+self.bias


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=TinyUnet()
        self.normalizer={"obs":Identity(),"action":Identity()}
        self.horizon,self.action_dim=16,2
        self.n_obs_steps,self.n_action_steps=2,8
        self.obs_dim=20
    def _condition(self,x):
        return x.flatten(1)


class FakeChunkEnv:
    def seed(self,seed): self.seed_value=seed
    def reset(self):
        self.rewards=[]
        self.step_no=0
        return np.zeros((2,40),np.float32)
    def get_rewards(self): return self.rewards
    def step(self,action):
        assert action.shape==(8,2)
        self.rewards.extend([.1,.4])
        self.step_no+=1
        return np.zeros((2,40),np.float32),.4,self.step_no>=2,{}


def test_data_collector_does_not_use_wrapper_max_as_training_reward():
    c=Controller(TinyTeacher(),"lowdim",torch.device("cpu"))
    replay=Replay(10,40,16)
    score,n=episode(FakeChunkEnv(),c,seed=102,gamma=.9,replay=replay)
    assert score==pytest.approx(.4)
    assert n==2 and replay.size==2
    assert replay.arrays["reward"][0,0]==pytest.approx(.1+.9*.4)
    assert replay.arrays["discount"][0,0]==pytest.approx(.9**2)
    assert replay.arrays["terminal"][:2,0].tolist()==[0.,1.]
