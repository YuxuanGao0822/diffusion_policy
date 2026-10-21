#!/usr/bin/env python3
"""Sequential reward-guided Priming research: lowdim -> RGB modality validation.

0: held-out baseline assessment; 1: frozen residual SAC;
2: anchored critic-guided one-step U-Net; 3: task-outcome OT + Q;
4: independently pretrain PushT RGB one-step model and repeat 0-3.

The same simulator physics in stage 4 means a *modality transfer check*,
not an independent manipulation-task/generalization claim.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from diffusion_policy.model.priming.reward_refinement import (
    Replay, ResidualActor, TwinQ, TaskDynamics, chunk_return,
    sac_critic_step, residual_actor_step, soft_target_update,
    task_aware_ot,
)

OmegaConf.register_new_resolver("eval",eval,replace=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_teacher(path,device,modality):
    """Stage-1 image/lowdim EMA only. Dill files must be locally trusted."""
    payload=torch.load(path,map_location="cpu",pickle_module=dill)
    cfg=payload["cfg"]
    is_image=modality=="image"
    actual_target=str(cfg.policy.get("_target_",""))
    expected=("priming_unet_image_policy" if is_image
              else "priming_unet_lowdim_policy")
    if expected not in actual_target:
        raise ValueError("Checkpoint policy type does not match "+modality+
                         ": "+actual_target)
    teacher=hydra.utils.instantiate(cfg.policy)
    teacher.load_state_dict(payload["state_dicts"]["ema_model"],strict=True)
    teacher.to(device).eval().requires_grad_(False)
    if is_image:
        teacher.obs_encoder.eval()
    return teacher


def environment(modality,n_obs_steps,n_action_steps,max_steps):
    from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
    if modality=="image":
        from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
        base=PushTImageEnv(legacy=True,render_size=96)
    else:
        from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
        base=PushTKeypointsEnv(
            legacy=True, keypoint_visible_rate=1.0,agent_keypoints=False,
            **PushTKeypointsEnv.genenerate_keypoint_manager_params())
    return MultiStepWrapper(base,n_obs_steps=n_obs_steps,
                            n_action_steps=n_action_steps,
                            max_episode_steps=max_steps)


def tensor_obs(obs,modality,device):
    if modality=="image":
        return {key:torch.as_tensor(value,dtype=torch.float32,
                                    device=device).unsqueeze(0)
                for key,value in obs.items()}
    # Original PushTKeypointsRunner strips visibility masks, Do=obs_dim.
    n=obs.shape[-1]//2
    return torch.as_tensor(obs[:,:n],dtype=torch.float32,
                           device=device).unsqueeze(0)


@torch.no_grad()
def encode(teacher,obs,modality,device):
    o=tensor_obs(obs,modality,device)
    if modality=="image":
        return teacher.encode_condition(o).detach()
    n=teacher.normalizer["obs"].normalize(o)
    return teacher._condition(n).detach()


def generated(model,s,h,da):
    z=torch.randn(s.shape[0],h,da,device=s.device,dtype=s.dtype)
    t=torch.zeros(s.shape[0],device=s.device,dtype=torch.long)
    return model(z,t,global_cond=s)


def base_chunk(teacher,s):
    pred=generated(teacher.model,s,teacher.horizon,teacher.action_dim)
    i=teacher.n_obs_steps-1
    return pred[:,i:i+teacher.n_action_steps].clamp(-1.0,1.0).flatten(1)


class Controller:
    def __init__(self,teacher,modality,device,residual=None,generator=None):
        self.teacher=teacher
        self.modality=modality
        self.device=device
        self.residual=residual
        self.generator=generator
        self.action_dim=teacher.n_action_steps*teacher.action_dim

    @torch.no_grad()
    def feature(self,obs):
        return encode(self.teacher,obs,self.modality,self.device)

    @torch.no_grad()
    def action(self,s,deterministic=True,explore=0.0):
        if self.generator is not None:
            full=generated(self.generator,s,self.teacher.horizon,self.teacher.action_dim)
            i=self.teacher.n_obs_steps-1
            a=full[:,i:i+self.teacher.n_action_steps].flatten(1).clamp(-1,1)
        else:
            a=base_chunk(self.teacher,s)
            if self.residual is not None:
                a=self.residual.act(s,a,deterministic=deterministic)
        if explore:
            a=(a+explore*torch.randn_like(a)).clamp(-1,1)
        return a

    @torch.no_grad()
    def environment_action(self,a):
        n=a.reshape(-1,self.teacher.n_action_steps,self.teacher.action_dim)
        return self.teacher.normalizer["action"].unnormalize(n)[0].cpu().numpy()


def episode(env,controller,seed,gamma,replay=None,explore=0.0,deterministic=True):
    env.seed(int(seed))
    obs=env.reset()
    done=False
    best=0.0
    transitions=0
    while not done:
        s=controller.feature(obs)
        act=controller.action(s,deterministic=deterministic,explore=explore)
        old_count=len(env.get_rewards())
        next_obs,_,done,_=env.step(controller.environment_action(act))
        primitive=env.get_rewards()[old_count:]
        if not primitive:
            raise RuntimeError("PushT MultiStepWrapper did not expose rewards")
        best=max(best,max(map(float,primitive)))
        if replay is not None:
            ns=controller.feature(next_obs)
            rew,discount=chunk_return(primitive,gamma)
            replay.add(s[0].cpu().numpy(),act[0].cpu().numpy(),
                       rew,ns[0].cpu().numpy(),discount,done)
        obs=next_obs
        transitions+=1
    return float(best),transitions


def evaluate(controller,args,modality,tag,output,seeds=None):
    base_seed=200000 if modality=="lowdim" else 300000
    if seeds is None:
        seeds=[base_seed+i for i in range(args.eval_episodes)]
    env=environment(modality,controller.teacher.n_obs_steps,
                    controller.teacher.n_action_steps,args.max_steps)
    previous=torch.random.get_rng_state()
    scores=[]
    try:
        for seed in seeds:
            torch.manual_seed(seed) # same generator randomness per seed/phase
            score,_=episode(env,controller,seed,args.gamma)
            scores.append((int(seed),score))
    finally:
        torch.random.set_rng_state(previous)
        env.close()
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["seed","max_reward"])
        w.writerows(scores)
    mean=float(np.mean([v for _,v in scores]))
    print(f"[{modality} {tag}] N={len(scores)} mean score={mean:.6f}",
          flush=True)
    return mean


def save_state(path,phase,completed,episode_idx,best,actor,gen,q,target,dyn,
               opt_actor,opt_q,opt_dyn,replay,extra=None):
    payload={
        "phase":phase,"completed":completed,"episode":episode_idx,
        "best_score":best,"actor":actor.state_dict() if actor is not None else None,
        "generator":gen.state_dict() if gen is not None else None,
        "q":q.state_dict(),"target_q":target.state_dict(),
        "dynamics":dyn.state_dict() if dyn is not None else None,
        "opt_actor":opt_actor.state_dict(),
        "opt_q":opt_q.state_dict(),
        "opt_dyn":opt_dyn.state_dict() if opt_dyn is not None else None,
        "replay":replay.state_dict(),
        "rng_torch":torch.random.get_rng_state(),
        "rng_numpy":np.random.get_state(),
        "rng_python":random.getstate(),
        "extra":extra or {},
    }
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    torch.save(payload,tmp)
    os.replace(tmp,path)


def load_state(path,device):
    return torch.load(path,map_location=device,weights_only=False)


def restore_rng(state):
    torch.random.set_rng_state(state["rng_torch"].cpu())
    np.random.set_state(state["rng_numpy"])
    random.setstate(state["rng_python"])


def generator_action(model,teacher,s):
    full=generated(model,s,teacher.horizon,teacher.action_dim)
    i=teacher.n_obs_steps-1
    return full[:,i:i+teacher.n_action_steps].flatten(1).clamp(-1,1)


class GeneratorTarget:
    """SAC bootstrapping from implicit one-step generative actor (no log pi)."""
    def __init__(self,model,teacher):
        self.model=model
        self.teacher=teacher
    def __call__(self,s,base):
        a=generator_action(self.model,self.teacher,s)
        return a,torch.zeros(s.shape[0],1,device=s.device)


def initialize_phase(phase,teacher,device,args,predecessor=None):
    sd=teacher.n_obs_steps*(teacher.obs_feature_dim if
                           hasattr(teacher,"obs_feature_dim") else teacher.obs_dim)
    ad=teacher.n_action_steps*teacher.action_dim
    actor=ResidualActor(sd,ad,args.residual_bound).to(device) if phase==1 else None
    gen=copy.deepcopy(teacher.model).to(device) if phase in (2,3) else None
    q=TwinQ(sd,ad).to(device)
    tq=copy.deepcopy(q).eval()
    dyn=TaskDynamics(sd,ad,seed=args.seed).to(device) if phase==3 else None
    if predecessor is not None:
        q.load_state_dict(predecessor["q"])
        tq.load_state_dict(predecessor["target_q"])
        if phase==2 and predecessor["actor"] is not None:
            # Residual policy is a *frozen reference* for Stage 2 distillation.
            reference=ResidualActor(sd,ad,args.residual_bound).to(device)
            reference.load_state_dict(predecessor["actor"])
            reference.eval().requires_grad_(False)
        else:
            reference=None
        if phase==3 and predecessor["generator"] is not None:
            gen.load_state_dict(predecessor["generator"])
    else:
        reference=None
    replay=Replay(args.capacity,sd,ad) if predecessor is None else Replay.from_state_dict(predecessor["replay"])
    for net in [q,tq]:
        net.train()
    optq=torch.optim.Adam(q.parameters(),lr=args.q_lr)
    opta=torch.optim.Adam((actor if phase==1 else gen).parameters(),lr=args.actor_lr)
    optd=torch.optim.Adam(dyn.parameters(),lr=args.dynamics_lr) if dyn is not None else None
    return actor,gen,q,tq,dyn,reference,replay,opta,optq,optd


def critic_step(q,tq,teacher,actor,gen,batch,optq,args):
    with torch.no_grad():
        next_base=base_chunk(teacher,batch["next_state"])
    next_actor=(actor if gen is None else GeneratorTarget(gen,teacher))
    return sac_critic_step(q,tq,next_actor,batch,next_base,optq,
                           alpha=args.entropy_coef if gen is None else 0.0,
                           tau=args.target_tau)


def actor_step(phase,teacher,actor,gen,q,dyn,residual_ref,batch,opta,args):
    s=batch["state"]
    if phase==1:
        with torch.no_grad():
            base=base_chunk(teacher,s)
        return residual_actor_step(actor,q,batch,base,opta,args.entropy_coef)

    # Direct differentiable one-step U-Net Q-guided actor fine-tuning.
    for p in q.parameters():
        p.requires_grad_(False)
    a=generator_action(gen,teacher,s)
    with torch.no_grad():
        base=base_chunk(teacher,s)
        if residual_ref is None:
            residual_target=base
        else:
            residual_target=residual_ref.act(s,base,deterministic=True)
    qloss=-args.q_weight*q.minimum(s,a).mean()
    anchor=args.teacher_anchor*F.mse_loss(a,base)
    distill=args.residual_distill*F.mse_loss(a,residual_target) if phase==2 else 0.0
    loss=qloss+anchor+distill
    if phase==3:
        # The dynamics network remains frozen while its action Jacobian
        # supplies task-feature gradients; avoid bootstrapping imagined reward
        # as an independent success signal.
        for p in dyn.parameters():
            p.requires_grad_(False)
        b=s.shape[0]
        k=args.task_samples
        repeated=s[:,None,:].expand(-1,k,-1).reshape(b*k,-1)
        live=generator_action(gen,teacher,repeated).reshape(b,k,-1)
        with torch.no_grad():
            reference=generator_action(gen,teacher,repeated).reshape(b,k,-1)
        t_loss=task_aware_ot(
            s,live,reference,batch["action"],batch["next_state"],batch["reward"],
            dyn,weight=args.task_feature_weight,
            epsilon=args.ot_epsilon,iterations=args.ot_iterations)
        loss=loss+args.task_ot_weight*t_loss
    opta.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gen.parameters(),args.grad_clip)
    opta.step()
    if phase==3:
        for p in dyn.parameters():
            p.requires_grad_(True)
    for p in q.parameters():
        p.requires_grad_(True)
    return float(loss.detach())


def train_dynamics(dyn,opt,replay,args,device):
    # Fixed held-out subset; robustly refuse unconstrained task OT if model
    # does not improve over a constant predictor on this validation subset.
    n=replay.size
    required=max(args.batch_size*2, 4 if args.smoke else 100)
    if n<required:
        raise RuntimeError("Not enough transitions to train task dynamics: "+
                           f"{n} < {required}. Collect more episodes.")
    rng=np.random.RandomState(args.seed+303)
    ids=rng.permutation(n)
    cutoff=max(1,int(0.8*n))
    train_ids,valid_ids=ids[:cutoff],ids[cutoff:]
    def batch_ids(index):
        return {k:torch.as_tensor(v[index],device=device)
                for k,v in replay.arrays.items()}
    dyn.train()
    for step in range(args.dynamics_updates):
        selected=rng.choice(train_ids,size=min(args.batch_size,len(train_ids)),replace=True)
        batch=batch_ids(selected)
        loss=dyn.loss(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    dyn.eval()
    with torch.no_grad():
        test=batch_ids(valid_ids)
        predicted=dyn(test["state"],test["action"])
        labels=dyn.target(test["next_state"],test["reward"])
        mse=F.mse_loss(predicted,labels).item()
        constant=F.mse_loss(labels,labels.mean(dim=0,keepdim=True)).item()
    ratio=mse/max(constant,1e-12)
    print(f"[task dynamics] holdout mse={mse:.6f} / constant={constant:.6f}; ratio={ratio:.4f}")
    if not np.isfinite(ratio) or ratio>args.max_dynamics_mse_ratio:
        raise RuntimeError("Dynamics validation gate failed; refusing task-aware OT. "+
                           f"ratio={ratio:.4f} threshold={args.max_dynamics_mse_ratio}.")
    return ratio


def run_phase(phase,modality,teacher,args,parent,previous=None):
    name={1:"residual_sac",2:"anchored_q_priming",3:"task_aware_ot"}[phase]
    run=Path(parent)/name
    run.mkdir(parents=True,exist_ok=True)
    # Guard resumed experiments against silently changing the checkpoint
    # initialization, optimizer parameters, seed, or training budget.
    manifest=run/"run_config.json"
    relevant={
        "phase":phase,"modality":modality,"checkpoint":str(args.checkpoint)
            if modality=="lowdim" else str(args.image_checkpoint or "image_stage1"),
        "parameters":{k:v for k,v in vars(args).items()
            if k not in ("phase","dry_run","output","image_checkpoint","smoke")},
    }
    if manifest.exists():
        if json.loads(manifest.read_text()) != relevant:
            raise RuntimeError("Run parameters changed for existing "+str(run)+
                "; use a distinct --output directory for the new experiment.")
    else:
        manifest.write_text(json.dumps(relevant,indent=2,sort_keys=True))
    best_path=run/"best.pt"
    latest_path=run/"latest.pt"
    device=next(teacher.model.parameters()).device
    actor,gen,q,tq,dyn,residual_ref,replay,opta,optq,optd=initialize_phase(
        phase,teacher,device,args,predecessor=previous)
    ctrl=Controller(teacher,modality,device,actor,gen)
    start=0
    best=float("-inf")
    if latest_path.exists():
        ckpt=load_state(latest_path,device)
        if ckpt["phase"]!=phase:
            raise ValueError("checkpoint phase mismatch")
        actor.load_state_dict(ckpt["actor"]) if actor is not None else gen.load_state_dict(ckpt["generator"])
        q.load_state_dict(ckpt["q"])
        tq.load_state_dict(ckpt["target_q"])
        if dyn is not None and ckpt["dynamics"] is not None:
            dyn.load_state_dict(ckpt["dynamics"])
        opta.load_state_dict(ckpt["opt_actor"])
        optq.load_state_dict(ckpt["opt_q"])
        if optd is not None and ckpt["opt_dyn"] is not None:
            optd.load_state_dict(ckpt["opt_dyn"])
        replay=Replay.from_state_dict(ckpt["replay"])
        start=int(ckpt["episode"])
        best=float(ckpt["best_score"])
        restore_rng(ckpt)
        if ckpt["completed"]:
            print(f"SKIP completed {modality} {name} (episodes={start})")
            return ckpt
    else:
        # Baseline selection metric before phase updates. Same held-out seed
        # set at each phase. Will not be called a final unbiased test.
        baseline=evaluate(ctrl,args,modality,name+"_initial",run/"initial.csv")
        best=baseline
        save_state(best_path,phase,False,0,best,actor,gen,q,tq,dyn,
                   opta,optq,optd,replay,extra={"score":baseline})
    if phase==3 and dyn is not None and start==0:
        train_dynamics(dyn,optd,replay,args,device)
    train_env=environment(modality,teacher.n_obs_steps,
                          teacher.n_action_steps,args.max_steps)
    logs=run/"train.jsonl"
    try:
        for i in range(start,args.episodes):
            # Actor rollout collection. Never use held-out evaluation seeds.
            score,steps=episode(
                train_env,ctrl,args.train_seed+i,args.gamma,replay,
                explore=args.exploration_std if phase!=1 else 0.0,
                deterministic=(phase!=1))
            row={"episode":i,"train_max_score":score,
                 "transitions":steps,"replay_size":replay.size}
            if replay.size>=args.batch_size:
                last_q,last_actor=0.0,0.0
                for j in range(args.updates_per_episode):
                    b=replay.sample(args.batch_size,device)
                    last_q=critic_step(q,tq,teacher,actor,gen,b,optq,args)
                    last_actor=actor_step(
                        phase,teacher,actor,gen,q,dyn,residual_ref,b,opta,args)
                    if phase==3 and j%3==0:
                        d_batch=replay.sample(args.batch_size,device)
                        last_dyn=dyn.loss(d_batch)
                        optd.zero_grad(set_to_none=True)
                        last_dyn.backward()
                        optd.step()
                row.update({"q_loss":last_q,"actor_loss":last_actor})
            if (i+1)%args.eval_every==0 or i+1==args.episodes:
                value=evaluate(ctrl,args,modality,name+f"_ep{i+1:04d}",
                               run/f"eval_ep{i+1:04d}.csv")
                row["val_mean_score"]=value
                if value>best:
                    best=value
                    save_state(best_path,phase,False,i+1,best,actor,gen,q,tq,dyn,
                               opta,optq,optd,replay,extra={"score":best})
                save_state(latest_path,phase,False,i+1,best,actor,gen,q,tq,dyn,
                           opta,optq,optd,replay)
            with logs.open("a") as f:
                f.write(json.dumps(row)+"\n")
            print(f"[{modality} {name}] ep={i+1}/{args.episodes} "
                  f"train={score:.3f} best_eval={best:.4f}",flush=True)
    finally:
        train_env.close()
    save_state(latest_path,phase,True,args.episodes,best,actor,gen,q,tq,dyn,
               opta,optq,optd,replay,extra={"best_score":best})
    print(f"COMPLETE {modality}/{name}: best validation {best:.6f}")
    return load_state(latest_path,device)


def choose_image_checkpoint(directory):
    paths=list((Path(directory)/"checkpoints").glob("epoch=*-test_mean_score=*.ckpt"))
    if paths:
        def quality(p):
            m=re.search(r"test_mean_score=([0-9]+(?:\.[0-9]+)?)",p.name)
            return float(m.group(1)) if m else -1
        return str(max(paths,key=quality).resolve())
    latest=Path(directory)/"checkpoints"/"latest.ckpt"
    if not latest.exists():
        raise FileNotFoundError("PushT image Stage-1 training emitted no checkpoint")
    return str(latest.resolve())


def run_image_pretraining(args):
    path=Path(args.image_checkpoint).expanduser().resolve() if args.image_checkpoint else None
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return str(path)
    out=Path(args.output)/"image_stage1"
    out.mkdir(parents=True,exist_ok=True)
    complete_marker=out/"image_pretrain_complete.json"
    if complete_marker.is_file():
        meta=json.loads(complete_marker.read_text())
        if meta.get("epochs") != (2 if args.smoke else args.image_epochs):
            raise RuntimeError("Image Stage-1 pretrain budget differs; use new --output")
        return choose_image_checkpoint(out)
    if (out/"checkpoints").exists() and list((out/"checkpoints").glob("*.ckpt")):
        raise RuntimeError(
            "Interrupted image Stage-1 pretraining found in "+str(out)+
            ". Cannot silently treat a partial image model as fully trained. "+
            "Resume the original image workspace manually or start with a "+
            "different --output; --image-checkpoint explicitly selects a snapshot.")
    if not Path("data/pusht/pusht_cchi_v7_replay.zarr").exists():
        raise FileNotFoundError("Missing data/pusht/pusht_cchi_v7_replay.zarr")
    cmd=[
        sys.executable,"train.py","--config-name=train_priming_image_workspace",
        "training.device="+args.device,
        "training.seed="+str(args.seed),
        "training.num_epochs="+str(args.image_epochs),
        "training.debug="+str(bool(args.smoke)).lower(),
        "training.resume=false","logging.mode=disabled",
        "task.env_runner.n_train_vis=0","task.env_runner.n_test_vis=0",
        "dataloader.batch_size="+str(args.image_batch),
        "val_dataloader.batch_size="+str(args.image_batch),
        "hydra.run.dir="+str(out.resolve())
    ]
    print("Starting independent image Priming Stage-1:", " ".join(cmd),flush=True)
    subprocess.run(cmd,check=True)
    result=choose_image_checkpoint(out)
    complete_marker.write_text(json.dumps({
        "epochs":2 if args.smoke else args.image_epochs,
        "checkpoint":result,"timestamp":time.time(),
    },indent=2))
    return result


def research_modality(modality,checkpoint,args):
    teacher=load_teacher(checkpoint,torch.device(args.device),modality)
    root=Path(args.output)/modality
    root.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    baseline_file=root/"baseline_select.csv"
    ctrl=Controller(teacher,modality,device)
    if not baseline_file.exists():
        evaluate(ctrl,args,modality,"frozen_stage1",baseline_file)
    with baseline_file.open() as f:
        vals=[float(r["max_reward"]) for r in csv.DictReader(f)]
    print(f"BASELINE {modality} mean={np.mean(vals):.6f} n={len(vals)}")
    previous=None
    run_names={1:"residual_sac",2:"anchored_q_priming",3:"task_aware_ot"}
    for phase in (1,2,3):
        previous=run_phase(phase,modality,teacher,args,root,previous)
        # The NEXT stage gets the best predecessor POLICY parameters, and
        # the latest critic/replay state. Do not accidentally propagate a
        # late overtrained policy just because it is 'latest'.
        best=load_state(root/run_names[phase]/"best.pt",device)
        key="actor" if phase==1 else "generator"
        previous[key]=best[key]
    # Final seeds are kept out of all checkpoint selection and updates.
    base_seed=500000 if modality=="lowdim" else 600000
    seeds=[base_seed+i for i in range(args.final_eval_episodes)]
    summary=[]
    controllers=[("frozen_stage1",Controller(teacher,modality,device))]
    for phase in (1,2,3):
        name=run_names[phase]
        best=load_state(root/name/"best.pt",device)
        if phase==1:
            sd=teacher.n_obs_steps*(teacher.obs_feature_dim if
                hasattr(teacher,"obs_feature_dim") else teacher.obs_dim)
            ad=teacher.n_action_steps*teacher.action_dim
            residual=ResidualActor(sd,ad,args.residual_bound).to(device)
            residual.load_state_dict(best["actor"])
            residual.eval()
            ctrl=Controller(teacher,modality,device,residual=residual)
        else:
            generator=copy.deepcopy(teacher.model).to(device)
            generator.load_state_dict(best["generator"])
            generator.eval()
            ctrl=Controller(teacher,modality,device,generator=generator)
        controllers.append((name,ctrl))
    for name,ctrl in controllers:
        score=evaluate(ctrl,args,modality,name+"_unseen",
                       root/("final_"+name+".csv"),seeds=seeds)
        summary.append({"policy":name,"heldout_score":score,
                        "n_seeds":len(seeds)})
    with (root/"final_comparison.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("FINAL HELD-OUT SUMMARY",modality,summary,flush=True)
    return previous


def preflight(args):
    if not Path("data/pusht/pusht_cchi_v7_replay.zarr").exists():
        raise FileNotFoundError("PushT Zarr dataset missing")
    if not Path(args.checkpoint).expanduser().is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.episodes<1 or args.eval_every<1 or args.eval_episodes<1:
        raise ValueError("Invalid sampling budget")
    if args.task_samples<2:
        raise ValueError("Task OT K must be at least 2")
    if args.phase not in ("all","lowdim","image"):
        raise ValueError("phase unsupported")
    print("PREFLIGHT OK; output",Path(args.output).resolve())
    print("lowdim E350:",Path(args.checkpoint).resolve())
    print("image checkpoint:",args.image_checkpoint or
          f"independent Stage-1 training ({args.image_epochs} epochs)")
    print("Online episodes per RL phase:",args.episodes)
    print("Phases run sequentially. Failure terminates without marking completion.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True,help="trusted lowdim Stage-1 E350 ckpt")
    p.add_argument("--image-checkpoint",default=None,
                   help="optional independently trained one-step PushT Image EMA ckpt")
    p.add_argument("--phase",choices=("all","lowdim","image"),default="all")
    p.add_argument("--output",default="data/outputs/priming_research")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--train-seed",type=int,default=10000)
    p.add_argument("--episodes",type=int,default=120)
    p.add_argument("--eval-episodes",type=int,default=25)
    p.add_argument("--final-eval-episodes",type=int,default=50)
    p.add_argument("--eval-every",type=int,default=20)
    p.add_argument("--max-steps",type=int,default=300)
    p.add_argument("--capacity",type=int,default=30000)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--updates-per-episode",type=int,default=40)
    p.add_argument("--gamma",type=float,default=0.99)
    p.add_argument("--target-tau",type=float,default=0.005)
    p.add_argument("--entropy-coef",type=float,default=0.005)
    p.add_argument("--residual-bound",type=float,default=0.10)
    p.add_argument("--actor-lr",type=float,default=1e-5)
    p.add_argument("--q-lr",type=float,default=3e-4)
    p.add_argument("--dynamics-lr",type=float,default=3e-4)
    p.add_argument("--dynamics-updates",type=int,default=1500)
    p.add_argument("--max-dynamics-mse-ratio",type=float,default=0.98)
    p.add_argument("--teacher-anchor",type=float,default=2.0)
    p.add_argument("--residual-distill",type=float,default=0.5)
    p.add_argument("--q-weight",type=float,default=0.01)
    p.add_argument("--exploration-std",type=float,default=0.03)
    p.add_argument("--grad-clip",type=float,default=1.0)
    p.add_argument("--task-ot-weight",type=float,default=0.1)
    p.add_argument("--task-feature-weight",type=float,default=0.25)
    p.add_argument("--task-samples",type=int,default=4)
    p.add_argument("--ot-epsilon",type=float,default=0.05)
    p.add_argument("--ot-iterations",type=int,default=50)
    p.add_argument("--image-epochs",type=int,default=350)
    p.add_argument("--image-batch",type=int,default=64)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    if args.smoke:
        args.episodes=2
        args.eval_every=1
        args.eval_episodes=1
        args.final_eval_episodes=1
        args.updates_per_episode=1
        args.dynamics_updates=10
        args.batch_size=2
        args.max_dynamics_mse_ratio=float("inf")
    preflight(args)
    if args.dry_run:
        return
    set_seed(args.seed)
    if args.phase in ("all","lowdim"):
        research_modality("lowdim",args.checkpoint,args)
    if args.phase in ("all","image"):
        image_ckpt=run_image_pretraining(args)
        # Separate image output from lowdim so no model weights or replay
        # are accidentally reused across observation modalities.
        research_modality("image",image_ckpt,args)
    print("ALL REQUESTED RESEARCH PHASES COMPLETE",flush=True)


if __name__=="__main__":
    main()
