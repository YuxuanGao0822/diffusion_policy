"""CPU tests for the lowdim/image online Priming research components.

Run from the repository root in robodiff:
    python -m pytest -q tests/test_priming_control_research.py
"""
from types import SimpleNamespace
import torch
import torch.nn as nn
import numpy as np

from diffusion_policy.model.priming.control_research import (
    Actor, TwinQ, Dynamics, Replay, task_ot_features, PushTChunkEnv,
)
from scripts.train_priming_control import train_update


class TinyGenerator(nn.Module):
    def __init__(self,h=16,da=2,cond=4):
        super().__init__()
        self.net=nn.Linear(h*da+cond,h*da)
        self.h,self.da=h,da

    def forward(self,z,t,global_cond):
        return self.net(torch.cat([z.flatten(1),global_cond],dim=1)).reshape(
            len(z),self.h,self.da)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=TinyGenerator()
        self.n_obs_steps=2
        self.n_action_steps=8
        self.horizon=16
        self.action_dim=2
        self.obs_dim=2

    def _generate(self,noise,cond):
        t=torch.zeros(len(noise),dtype=torch.long,device=noise.device)
        return self.model(noise,t,global_cond=cond)


def _populate(replay,count=24):
    for i in range(count):
        replay.add(torch.randn(1,4),torch.randn(1,8,2).clamp(-1,1),
                   1.25,torch.randn(1,4),i%5==0,4 if i%3==0 else 8)


def test_chunk_replay_preserves_discount_exponent_and_shapes():
    replay=Replay(100)
    _populate(replay)
    cond,action,rew,nxt,done,duration=replay.sample(6,"cpu")
    assert cond.shape==(6,4)
    assert action.shape==(6,8,2)
    assert nxt.shape==cond.shape
    assert rew.shape==done.shape==duration.shape==(6,)
    assert set(duration.tolist()).issubset({4.,8.})
    assert all(0.<=x<=1. for x in done.tolist())


def test_frozen_teacher_residual_bounded():
    teacher=TinyTeacher()
    actor=Actor(teacher,"residual",residual_scale=0.1).initialize(4)
    cond=torch.randn(7,4)
    noise=torch.randn(7,16,2)
    actual,base=actor.actions(cond,noise)
    assert actual.shape==base.shape==(7,8,2)
    assert (actual-base).abs().max().item()<=0.100001
    assert not any(p.requires_grad for p in actor.teacher.parameters())
    assert all(p.requires_grad for p in actor.trainable.parameters())


def test_q_and_task_feature_gradients():
    teacher=TinyTeacher()
    actor=Actor(teacher,"q").initialize(4)
    action,_=actor.actions(torch.randn(3,4),torch.randn(3,16,2))
    assert action.requires_grad
    dynamics=Dynamics(4)
    features=task_ot_features(dynamics,torch.randn(3,4),
                              action[:,None],1.)
    assert features.shape==(3,1,1,20)
    features.sum().backward()
    assert any(p.grad is not None for p in actor.trainable.parameters())


def test_residual_q_learning_update_is_finite():
    torch.manual_seed(10)
    teacher=TinyTeacher()
    actor=Actor(teacher,"residual").initialize(4)
    critic=TwinQ(4)
    tc=TwinQ(4)
    tc.load_state_dict(critic.state_dict())
    ta=Actor(teacher,"residual").initialize(4).trainable
    ta.load_state_dict(actor.trainable.state_dict())
    ta.requires_grad_(False)
    replay=Replay(30)
    _populate(replay)
    opt_actor=torch.optim.Adam(actor.trainable.parameters(),lr=1e-4)
    opt_critic=torch.optim.Adam(critic.parameters(),lr=1e-4)
    args=SimpleNamespace(batch=4,gamma=0.99,anchor=5.,tau=.005)
    metrics=train_update(actor,critic,tc,ta,replay,opt_actor,opt_critic,
                         None,None,None,"residual",args,torch.device("cpu"))
    assert np.isfinite(list(metrics.values())).all()


def test_chunk_reward_uses_each_step_and_correct_bootstrap_duration():
    class Dummy:
        def __init__(self):self.count=0
        def step(self,action):
            self.count+=1
            return np.ones(40)*self.count,float(self.count),self.count==3,{}
        def close(self):pass
    wrapper=object.__new__(PushTChunkEnv)
    wrapper.env=Dummy()
    wrapper.image=False
    wrapper.n_obs_steps=2
    wrapper.n_action_steps=8
    wrapper.max_steps=300
    wrapper.gamma=0.9
    wrapper.steps=0
    from collections import deque
    wrapper.history=deque([np.zeros(40),np.zeros(40)],maxlen=2)
    obs,r,done,k,best=wrapper.step(np.zeros((8,2)))
    assert obs.shape==(2,20)
    assert done and k==3 and best==3.
    assert abs(r-(1+0.9*2+0.9**2*3))<1e-6
    assert wrapper.steps==3
