#!/usr/bin/env python3
"""D: Does OT self-repulsion move the action mean instead of increasing diversity?

Repulsion-only one-step normalized SGD updates from E350 at K=8/16/32.
Measures held-out input-noise Jacobian and action variance; separately
compares training-sample pair distances to a FROZEN Stage-1 reference
versus a FRESH reference recomputed after the parameter update.

All eval conditions/noise/probes are fixed across train seeds and K.
No checkpoint is changed or saved. This is local mechanistic evidence,
not a PushT reward evaluation.
"""
import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import hydra
import numpy as np
import torch

from diffusion_policy.model.priming.finite_difference import (
    unit_loss_gradient,symmetric_scores,comparison)
from diffusion_policy.model.priming.sample_drift import (
    half_pairwise_squared,sample_variance,mean_shift_rms)
from scripts.diagnose_priming_ot_gradients import load_policy,seeded_noise
from scripts.diagnose_priming_noise_mechanism import (
    pick_indices,condition_and_target,make_train_batch,loss_components,
    sensitivity_from_function,window_slice,write_csv)


def generate(policy,noise,cond,clip):
    out=policy._generate(noise,cond.expand(noise.shape[0],-1))
    return out.clamp(-1,1) if clip and policy.clip_sample else out


def evaluation_cases(policy,ds,indices,k,seed,device):
    rows=[]
    for index in indices:
        cond,expert=condition_and_target(policy,ds[int(index)],device)
        z=seeded_noise((k,policy.horizon,policy.action_dim),
                       seed+int(index),cond.dtype,device)
        zref=seeded_noise((k,policy.horizon,policy.action_dim),
                          seed+1_000_000+int(index),cond.dtype,device)
        with torch.no_grad():
            old_live=generate(policy,z,cond,clip=True).detach().clone()
            old_ref=generate(policy,zref,cond,clip=True).detach().clone()
        rows.append((int(index),cond,expert,z,zref,old_live,old_ref))
    return rows


def evaluate(policy,cases,window,probes,seed):
    sl=window_slice(policy,window)
    records=[]
    for index,cond,expert,z,zref,old_live,old_ref in cases:
        sens=sensitivity_from_function(
            lambda z:generate(policy,z,cond,clip=True),z,sl,
            probes,seed+index+33,expert[0,sl,:])
        with torch.no_grad():
            live=generate(policy,z,cond,clip=True)[:,sl,:]
            ref=generate(policy,zref,cond,clip=True)[:,sl,:]
        old_l=old_live[:,sl,:]
        old_r=old_ref[:,sl,:]
        records.append({
            "Sz":sens["jacobian_sq_per_noise_dim"],
            "noise_std":sens["noise_std"],
            "expert_mse":sens["pure_noise_expert_mse"],
            "variance":sample_variance(live),
            "mean_shift_rms":mean_shift_rms(live,old_l),
            "pair_fresh":half_pairwise_squared(live,ref),
            "pair_stale":half_pairwise_squared(live,old_r),
        })
    return {key:float(np.mean([r[key] for r in records]))
            for key in records[0]}


def train_pair_stats(policy,cond,live_z,ref_z,old_live,old_ref):
    with torch.no_grad():
        b,k,h,da=old_live.shape
        c=cond.repeat_interleave(k,dim=0)
        live=policy._generate(live_z,c).reshape(b,k,h,da)
        ref=policy._generate(ref_z,c).reshape(b,k,h,da)
    return {
        "train_pair_fresh":float(np.mean(
            [half_pairwise_squared(live[i],ref[i]) for i in range(b)])),
        "train_pair_stale":float(np.mean(
            [half_pairwise_squared(live[i],old_ref[i]) for i in range(b)])),
        "train_variance":float(np.mean(
            [sample_variance(live[i]) for i in range(b)])),
        "train_mean_shift_rms":float(np.mean(
            [mean_shift_rms(live[i],old_live[i]) for i in range(b)])),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--train-seeds",type=int,nargs="+",default=[20261021,20261022,20261023])
    p.add_argument("--samples-per-condition",type=int,nargs="+",default=[8,16,32])
    p.add_argument("--step-norm",type=float,default=0.001)
    p.add_argument("--train-conditions",type=int,default=2)
    p.add_argument("--eval-conditions",type=int,default=8)
    p.add_argument("--eval-samples",type=int,default=16)
    p.add_argument("--probes",type=int,default=8)
    p.add_argument("--eval-seed",type=int,default=314159)
    p.add_argument("--window",nargs="+",choices=("full","executed"),
                   default=["full","executed"])
    p.add_argument("--ot-window",choices=("full","executed"),default="full")
    p.add_argument("--epsilon",type=float,default=0.05)
    p.add_argument("--iterations",type=int,default=50)
    p.add_argument("--output",default="data/outputs/priming_pusht/noise_reference_drift/k_sweep.csv")
    a=p.parse_args()
    if (a.step_norm<=0 or min(a.samples_per_condition)<2 or
        min(a.train_conditions,a.eval_conditions,a.eval_samples,a.probes)<1 or
        a.eval_samples<2):
        p.error("invalid sample count or step norm")
    device=torch.device(a.device)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    ckpt=Path(a.checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    cfg,policy=load_policy(str(ckpt),device)
    policy.eval()
    ds=hydra.utils.instantiate(cfg.task.dataset)
    val=ds.get_validation_dataset()
    ids=pick_indices(val,a.eval_conditions,a.eval_seed)
    cases=evaluation_cases(policy,val,ids,a.eval_samples,a.eval_seed,device)
    policy.model.requires_grad_(False)
    baseline={window:evaluate(policy,cases,window,a.probes,a.eval_seed)
              for window in a.window}
    params=list(policy.model.parameters())
    rows=[]
    print("Fixed held-out baseline:",baseline,flush=True)
    for seed in a.train_seeds:
        train_ids=pick_indices(ds,a.train_conditions,seed)
        for k in a.samples_per_condition:
            policy.model.requires_grad_(True)
            cond,real=make_train_batch(policy,ds,train_ids,device)
            b,h,da=real.shape
            cc=cond.repeat_interleave(k,dim=0)
            train_z=seeded_noise((b*k,h,da),seed+23,real.dtype,device)
            train_ref=seeded_noise((b*k,h,da),seed+24,real.dtype,device)
            with torch.no_grad():
                old_live=policy._generate(train_z,cc).reshape(b,k,h,da).detach().clone()
                old_ref=policy._generate(train_ref,cc).reshape(b,k,h,da).detach().clone()
            losses=loss_components(policy,cond,real,k,seed+23,
                                   a.ot_window,a.epsilon,a.iterations)
            unit,gradient_norm=unit_loss_gradient(-losses["repulsion"],params)
            del losses,cc
            policy.model.requires_grad_(False)
            def score():
                metrics={window:evaluate(policy,cases,window,a.probes,a.eval_seed)
                         for window in a.window}
                metrics["training"]=train_pair_stats(
                    policy,cond,train_z,train_ref,old_live,old_ref)
                return metrics
            base_train=train_pair_stats(
                policy,cond,train_z,train_ref,old_live,old_ref)
            plus,minus=symmetric_scores(params,unit,a.step_norm,score)
            for window in a.window:
                row={"train_seed":seed,"K":k,"window":window,
                     "update_norm":a.step_norm,
                     "gradient_norm":gradient_norm,
                     "eval_conditions":len(cases),"eval_samples":a.eval_samples,
                     "probes":a.probes}
                for metric in ("Sz","noise_std","expert_mse","variance",
                               "pair_fresh","pair_stale"):
                    change=comparison(baseline[window][metric],
                                      plus[window][metric],
                                      minus[window][metric],a.step_norm)
                    row["before_"+metric]=baseline[window][metric]
                    row["after_"+metric]=minus[window][metric]
                    row["delta_"+metric]=change["delta_observed"]
                    row["pred_delta_"+metric]=change["delta_predicted"]
                row["val_mean_shift_rms"]=minus[window]["mean_shift_rms"]
                for metric in ("train_pair_fresh","train_pair_stale","train_variance"):
                    row["before_"+metric]=base_train[metric]
                    row["after_"+metric]=minus["training"][metric]
                    row["delta_"+metric]=minus["training"][metric]-base_train[metric]
                row["train_mean_shift_rms"]=minus["training"]["train_mean_shift_rms"]
                rows.append(row)
                print(f"seed={seed} K={k:2d} {window:8s} "
                      f"deltaSz={row['delta_Sz']:+.7g} "
                      f"deltaVar={row['delta_variance']:+.7g} "
                      f"meanShift={row['val_mean_shift_rms']:.7g} "
                      f"trainPairFresh={row['delta_train_pair_fresh']:+.7g} "
                      f"trainPairStale={row['delta_train_pair_stale']:+.7g}",
                      flush=True)
            del unit,cond,real,train_z,train_ref,old_live,old_ref
    write_csv(a.output,rows)
    print("Fresh ref = regenerated with updated weights; stale ref = fixed E350."
          " Pair distances are raw uniform-pair statistics, NOT entropic"
          " Sinkhorn objective values.",flush=True)


if __name__=="__main__":
    main()
