#!/usr/bin/env python3
"""Train/evaluate conservative online Priming on PushT Lowdim or Image.

Requires a trusted Stage-1 EMA checkpoint. Offline losses alone are never
used to select the best actor: selections use disjoint fixed rollout seeds.
"""
import argparse
import copy
import json
import os
import pathlib
import random
import sys
import time

ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.model.priming.control_research import (
    set_seed,load_teacher,PushTChunkEnv,encode_observation,Actor,TwinQ,
    Dynamics,Replay,sample_noise,evaluate,soft_update,task_ot_features,
)
from diffusion_policy.model.priming.distribution_loss import ConditionalActionOTLoss
from diffusion_policy.common.pytorch_util import optimizer_to


def read_state(path,device):
    # Only read checkpoints produced by this trusted local research worker.
    return torch.load(path,map_location=device,weights_only=False)


def atomic_save(payload,path):
    path=pathlib.Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def optional_offline_samples(dataset,actor,batch_size,device):
    if dataset is None:
        raise ValueError("task_ot requires demonstration dataset")
    indices=np.random.randint(0,len(dataset),size=batch_size)
    records=[dataset[int(i)] for i in indices]
    first=records[0]["obs"]
    if isinstance(first,dict):
        batch={k:torch.stack([r["obs"][k] for r in records]).to(device)
               for k in first}
        nobs=actor.teacher.normalizer.normalize(batch)
    else:
        batch=torch.stack([r["obs"] for r in records]).to(device)
        nobs=actor.teacher.normalizer["obs"].normalize(batch)
    with torch.no_grad():
        cond=actor.teacher._condition(nobs)
    target=torch.stack([r["action"] for r in records]).to(device)
    real=actor.teacher.normalizer["action"].normalize(target)
    real=real[:,actor.start:actor.start+actor.steps]
    return cond,real


def train_update(actor,critic,target_critic,target_actor,replay,optim_actor,
                 optim_critic,dynamics,optim_dynamics,offline,objective,args,device):
    s,a,r,ns,done,duration=replay.sample(args.batch,device)
    with torch.no_grad():
        nz=sample_noise(actor,len(s),device)
        next_a,_=actor.actions(ns,nz,network=target_actor)
        # TD3-style target policy smoothing is avoided: the policy's own
        # Gaussian noise supplies target stochasticity.
        v1,v2=target_critic(ns,next_a)
        target=r+(args.gamma**duration)*(1-done)*torch.minimum(v1,v2)
    q1,q2=critic(s,a)
    loss_critic=F.mse_loss(q1,target)+F.mse_loss(q2,target)
    optim_critic.zero_grad(set_to_none=True)
    loss_critic.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(),10.)
    optim_critic.step()

    dyn_loss=torch.tensor(0.,device=device)
    if dynamics is not None:
        pred_next=dynamics(s,a)
        dyn_loss=F.mse_loss(pred_next,ns)
        optim_dynamics.zero_grad(set_to_none=True)
        dyn_loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics.parameters(),10.)
        optim_dynamics.step()

    for param in critic.parameters():
        param.requires_grad_(False)
    if dynamics is not None:
        # Preserve d(feature)/d(action) but never modify dynamics via actor OT.
        dynamics.requires_grad_(False)
    nz=sample_noise(actor,len(s),device)
    predicted,teacher_actions=actor.actions(s,nz)
    v1,v2=critic(s,predicted)
    actor_q=-torch.minimum(v1,v2).mean()
    # Anchor is on deployed normalized actions only, same teacher noise/obs.
    anchor=F.mse_loss(predicted,teacher_actions.detach())
    actor_loss=actor_q + args.anchor*anchor
    ot_value=torch.tensor(0.,device=device)
    if objective == "task_ot":
        with torch.no_grad():
            cond,expert=optional_offline_samples(offline,actor,args.ot_conditions,device)
        b=cond.shape[0]
        k=args.ot_samples
        cc=cond.repeat_interleave(k,dim=0)
        live_z=sample_noise(actor,b*k,device)
        live,_=actor.actions(cc,live_z)
        with torch.no_grad():
            ref_z=sample_noise(actor,b*k,device)
            reference,_=actor.actions(cc,ref_z)
        live=live.reshape(b,k,actor.steps,actor.action_dim)
        reference=reference.reshape(b,k,actor.steps,actor.action_dim)
        real=expert.unsqueeze(1)
        # Gradients flow through frozen trained dynamics to the action.
        live_features=task_ot_features(dynamics,cond,live,args.effect_scale)
        ref_features=task_ot_features(dynamics,cond,reference,args.effect_scale)
        real_features=task_ot_features(dynamics,cond,real,args.effect_scale)
        ot_value,_=ConditionalActionOTLoss(
            epsilon=args.ot_epsilon,sinkhorn_iterations=args.ot_iterations
        )(live_features,ref_features,real_features)
        actor_loss=actor_loss+args.ot_weight*ot_value
    optim_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.trainable.parameters(),1.)
    optim_actor.step()
    for param in critic.parameters():
        param.requires_grad_(True)
    if dynamics is not None:
        dynamics.requires_grad_(True)
    soft_update(critic,target_critic,args.tau)
    soft_update(actor.trainable,target_actor,args.tau)
    return {"q_loss":float(loss_critic.detach()),"actor_q":float(actor_q.detach()),
            "anchor":float(anchor.detach()),"task_ot":float(ot_value.detach()),
            "dynamics_loss":float(dyn_loss.detach())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    p=parser
    p.add_argument("--phase",choices=["baseline","residual","q","task_ot"],required=True)
    p.add_argument("--modality",choices=["lowdim","image"],required=True)
    p.add_argument("--teacher",required=True)
    p.add_argument("--previous",default=None,help="latest/best.pt from previous stage")
    p.add_argument("--out",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--steps",type=int,default=10000,help="online decision chunks, not episodes")
    p.add_argument("--max-env-steps",type=int,default=300)
    p.add_argument("--eval-every",type=int,default=500)
    p.add_argument("--eval-episodes",type=int,default=30)
    p.add_argument("--report-episodes",type=int,default=50)
    p.add_argument("--gamma",type=float,default=0.99)
    p.add_argument("--capacity",type=int,default=50000)
    p.add_argument("--batch",type=int,default=128)
    p.add_argument("--warmup",type=int,default=256)
    p.add_argument("--updates-per-chunk",type=int,default=1)
    p.add_argument("--actor-lr",type=float,default=1e-5)
    p.add_argument("--residual-lr",type=float,default=1e-4)
    p.add_argument("--critic-lr",type=float,default=3e-4)
    p.add_argument("--dynamics-lr",type=float,default=1e-4)
    p.add_argument("--tau",type=float,default=0.005)
    p.add_argument("--anchor",type=float,default=5.)
    p.add_argument("--residual-scale",type=float,default=0.1)
    p.add_argument("--exploration",type=float,default=0.05)
    p.add_argument("--ot-weight",type=float,default=0.1)
    p.add_argument("--ot-samples",type=int,default=4)
    p.add_argument("--ot-conditions",type=int,default=4)
    p.add_argument("--ot-epsilon",type=float,default=0.05)
    p.add_argument("--ot-iterations",type=int,default=50)
    p.add_argument("--effect-scale",type=float,default=1.0)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.steps<1 or args.eval_every<1 or args.batch<1:
        p.error("invalid training budget")
    if args.phase in ("q","task_ot") and not args.previous:
        p.error("q and task_ot require --previous")
    if args.phase=="residual" and args.previous:
        p.error("residual is initialized from frozen teacher, not previous")
    if args.phase=="baseline" and args.previous:
        p.error("baseline does not need previous")
    set_seed(args.seed)
    device=torch.device(args.device)
    out=pathlib.Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    latest=out/"latest.pt"
    best=out/"best.pt"
    teacher,cfg=load_teacher(args.teacher,device)
    modality_matches=("image" in str(cfg.task.name)) == (args.modality=="image")
    if not modality_matches:
        raise ValueError("teacher checkpoint modality does not match --modality")
    # Condition dim for image is obs_encoder feature dim * history length;
    # for lowdim it is n_obs_steps * obs_dim.
    cond_dim=teacher.n_obs_steps*teacher.obs_dim
    actor_phase="teacher" if args.phase=="baseline" else args.phase
    actor=Actor(teacher,actor_phase,residual_scale=args.residual_scale).initialize(cond_dim).to(device)
    eval_seeds=list(range(220000,220000+args.eval_episodes))
    report_seeds=list(range(320000,320000+args.report_episodes))

    if args.phase=="baseline":
        stats=evaluate(actor,teacher,args.modality,device,eval_seeds,
                       args.max_env_steps,args.gamma)
        report=evaluate(actor,teacher,args.modality,device,report_seeds,
                        args.max_env_steps,args.gamma)
        with (out/"selection.json").open("w") as f:
            json.dump(stats,f,indent=2)
        with (out/"heldout.json").open("w") as f:
            json.dump(report,f,indent=2)
        print("Frozen teacher selection",stats["mean_score"],
              "heldout",report["mean_score"])
        return

    critic=TwinQ(cond_dim,actor.steps,actor.action_dim).to(device)
    target_critic=copy.deepcopy(critic).requires_grad_(False)
    target_actor=copy.deepcopy(actor.trainable).requires_grad_(False)
    dynamics=Dynamics(cond_dim,actor.steps,actor.action_dim).to(device) if args.phase=="task_ot" else None
    optim_dynamics=torch.optim.Adam(dynamics.parameters(),lr=args.dynamics_lr) if dynamics is not None else None
    optim_actor=torch.optim.Adam(
        actor.trainable.parameters(),
        lr=args.residual_lr if args.phase=="residual" else args.actor_lr)
    optim_critic=torch.optim.Adam(critic.parameters(),lr=args.critic_lr)
    replay=Replay(args.capacity)
    best_score=float("-inf")
    seen=0
    episodes=0

    if args.previous:
        prev=read_state(args.previous,"cpu")
        if prev.get("modality") != args.modality:
            raise ValueError("previous checkpoint modality mismatch")
        critic.load_state_dict(prev["critic"])
        target_critic.load_state_dict(prev["target_critic"])
        replay.transitions.extend(prev["replay"])
        if args.phase=="task_ot":
            if prev.get("phase") != "q":
                raise ValueError("task_ot must follow q")
            actor.trainable.load_state_dict(prev["actor"])
            target_actor.load_state_dict(prev["target_actor"])
        elif prev.get("phase")!="residual":
            raise ValueError("q must follow residual")
        print("Loaded prior critic and",len(replay),"replay transitions",flush=True)

    if args.resume and latest.exists():
        state=read_state(latest,"cpu")
        if state["phase"]!=args.phase or state["modality"]!=args.modality:
            raise ValueError("resume checkpoint mismatch")
        actor.trainable.load_state_dict(state["actor"])
        target_actor.load_state_dict(state["target_actor"])
        critic.load_state_dict(state["critic"])
        target_critic.load_state_dict(state["target_critic"])
        replay.transitions.clear()
        replay.transitions.extend(state["replay"])
        optim_actor.load_state_dict(state["optim_actor"])
        optim_critic.load_state_dict(state["optim_critic"])
        optimizer_to(optim_actor,device)
        optimizer_to(optim_critic,device)
        if dynamics is not None:
            dynamics.load_state_dict(state["dynamics"])
            optim_dynamics.load_state_dict(state["optim_dynamics"])
            optimizer_to(optim_dynamics,device)
        seen,episodes,best_score=state["seen"],state["episodes"],state["best_score"]
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            np.random.set_state(state["numpy_rng"])
        print("Resumed step",seen,flush=True)

    offline=None
    if args.phase=="task_ot":
        import hydra
        offline=hydra.utils.instantiate(cfg.task.dataset)
        # Dynamics is warmed from transitions before task-aware OT is added.
        if not(args.resume and latest.exists()):
            for warm in range(min(1000,max(200,len(replay)//2))):
                if len(replay)<args.batch: break
                s,a,_,ns,_,_=replay.sample(args.batch,device)
                dl=F.mse_loss(dynamics(s,a),ns)
                optim_dynamics.zero_grad(set_to_none=True)
                dl.backward()
                torch.nn.utils.clip_grad_norm_(dynamics.parameters(),10.)
                optim_dynamics.step()
            print("Warm-started dynamics on collected replay",flush=True)

    env=PushTChunkEnv(args.modality,teacher.n_obs_steps,teacher.n_action_steps,
                      args.max_env_steps,args.gamma)
    log=open(out/"training.jsonl","a",buffering=1)
    def store(is_best):
        state={"phase":args.phase,"modality":args.modality,
           "actor":actor.trainable.state_dict(),
           "target_actor":target_actor.state_dict(),
           "critic":critic.state_dict(),"target_critic":target_critic.state_dict(),
           "optim_actor":optim_actor.state_dict(),
           "optim_critic":optim_critic.state_dict(),
           "dynamics":dynamics.state_dict() if dynamics is not None else None,
           "optim_dynamics":optim_dynamics.state_dict() if optim_dynamics is not None else None,
           "replay":list(replay.transitions),"seen":seen,
           "episodes":episodes,"best_score":best_score,
           "torch_rng":torch.get_rng_state(),
           "python_rng":random.getstate(),"numpy_rng":np.random.get_state()}
        atomic_save(state,latest)
        if is_best:
            atomic_save(state,best)
    try:
        obs=env.reset(1000+args.seed*100000+episodes)
        done=False
        while seen<args.steps:
            cond=encode_observation(teacher,obs,device)
            with torch.no_grad():
                action,_=actor.actions(cond,sample_noise(actor,1,device))
                if args.exploration>0:
                    action=(action + args.exploration*torch.randn_like(action)).clamp(-1,1)
                real_action=teacher.normalizer["action"].unnormalize(action)
            nobs,ret,done,duration,_=env.step(real_action[0].cpu().numpy())
            next_cond=encode_observation(teacher,nobs,device)
            replay.add(cond,action,ret,next_cond,done,duration)
            obs=nobs
            seen+=1
            if done:
                episodes+=1
                obs=env.reset(1000+args.seed*100000+episodes)
            metrics={}
            if len(replay)>=max(args.batch,args.warmup):
                for _ in range(args.updates_per_chunk):
                    metrics=train_update(actor,critic,target_critic,target_actor,replay,
                        optim_actor,optim_critic,dynamics,optim_dynamics,offline,
                        args.phase,args,device)
            if seen%args.eval_every==0 or seen==args.steps:
                # Deterministic evaluation seeds across methods.
                current=evaluate(actor,teacher,args.modality,device,eval_seeds,
                                 args.max_env_steps,args.gamma)
                improved=current["mean_score"]>best_score
                best_score=max(best_score,current["mean_score"])
                row={"step":seen,"episodes":episodes,"selection_mean":current["mean_score"],
                     "best_selection_mean":best_score,"replay_size":len(replay),**metrics}
                log.write(json.dumps(row)+"\n")
                store(improved)
                print(args.modality,args.phase,"step",seen,
                      "selection",round(current["mean_score"],4),
                      "best",round(best_score,4),flush=True)
        if not best.exists():
            raise RuntimeError("no best checkpoint saved")
    finally:
        log.close()
        env.close()

    state=read_state(best,"cpu")
    actor.trainable.load_state_dict(state["actor"])
    actor.eval()
    report=evaluate(actor,teacher,args.modality,device,report_seeds,
                    args.max_env_steps,args.gamma)
    with (out/"heldout.json").open("w") as f:
        json.dump(report,f,indent=2)
    with (out/"success.json").open("w") as f:
        json.dump({"phase":args.phase,"modality":args.modality,
                   "best_selection_mean":best_score,
                   "heldout_mean":report["mean_score"],"steps":seen},f,indent=2)
    print("Completed",args.phase,"heldout",report["mean_score"],flush=True)


if __name__=="__main__":
    main()
