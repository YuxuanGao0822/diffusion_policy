#!/usr/bin/env python3
"""Fixed-evaluation finite differences along attraction, -repulsion, OT and MSE gradients.

Measures D_u S_z and a descent-step prediction. This is NOT the full
Jacobian-to-parameter-gradient cosine rho_J. Trusted E350 dill checkpoints only.
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
from scripts.diagnose_priming_ot_gradients import load_policy,seeded_noise
from scripts.diagnose_priming_noise_mechanism import (
    pick_indices,condition_and_target,make_train_batch,loss_components,
    sensitivity_from_function,window_slice,write_csv)


def validation_cases(policy,ds,indices,samples,seed,device):
    cases=[]
    for idx in indices:
        cond,expert=condition_and_target(policy,ds[int(idx)],device)
        z=seeded_noise((samples,policy.horizon,policy.action_dim),
                       seed+int(idx),cond.dtype,device)
        cases.append((int(idx),cond,expert,z))
    return cases


def score_policy(policy,cases,windows,probes,eval_seed):
    output={}
    for window in windows:
        sl=window_slice(policy,window)
        scores=[]
        for idx,cond,expert,z in cases:
            def generate(noise):
                y=policy._generate(noise,cond.expand(len(noise),-1))
                return y.clamp(-1.,1.) if policy.clip_sample else y
            score=sensitivity_from_function(
                generate,z,sl,probes,eval_seed+int(idx)+33,
                expert[0,sl,:])
            scores.append(score)
        output[window]={key:float(np.mean([s[key] for s in scores]))
             for key in ("jacobian_sq_per_noise_dim","noise_std",
                         "pure_noise_expert_mse")}
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--train-seeds",nargs="+",type=int,default=[20261021,20261022,20261023])
    p.add_argument("--eval-seed",type=int,default=314159)
    p.add_argument("--train-conditions",type=int,default=2)
    p.add_argument("--train-samples",type=int,default=8)
    p.add_argument("--eval-conditions",type=int,default=16)
    p.add_argument("--eval-samples",type=int,default=12)
    p.add_argument("--probes",type=int,default=8)
    p.add_argument("--windows",nargs="+",choices=("full","executed"),
                   default=["full","executed"])
    p.add_argument("--arms",nargs="+",
                   choices=("attraction","negative_repulsion","ot","mse"),
                   default=["attraction","negative_repulsion","ot","mse"])
    p.add_argument("--step-norms",nargs="+",type=float,
                   default=[0.0005,0.001,0.003])
    p.add_argument("--ot-window",choices=("full","executed"),default="full")
    p.add_argument("--epsilon",type=float,default=0.05)
    p.add_argument("--iterations",type=int,default=50)
    p.add_argument("--output",default="data/outputs/priming_pusht/noise_gradient_direction/central_fd.csv")
    args=p.parse_args()
    if (not args.step_norms or min(args.step_norms)<=0 or
        min(args.train_conditions,args.train_samples,args.eval_conditions,
            args.eval_samples,args.probes)<1 or
        args.train_samples<2 or args.eval_samples<2):
        p.error("invalid sample counts or step norms")
    device=torch.device(args.device)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg,policy=load_policy(str(checkpoint),device)
    policy.eval()
    ds=hydra.utils.instantiate(cfg.task.dataset)
    val=ds.get_validation_dataset()
    eval_ids=pick_indices(val,args.eval_conditions,args.eval_seed)
    cases=validation_cases(policy,val,eval_ids,args.eval_samples,args.eval_seed,device)
    policy.model.requires_grad_(False)
    original=score_policy(policy,cases,args.windows,args.probes,args.eval_seed)
    print("Fixed validation",len(cases),"conditions, baseline:",original,flush=True)
    params=list(policy.model.parameters())
    rows=[]
    for seed in args.train_seeds:
        train_ids=pick_indices(ds,args.train_conditions,seed)
        for arm in args.arms:
            policy.model.requires_grad_(True)
            cond,real=make_train_batch(policy,ds,train_ids,device)
            losses=loss_components(policy,cond,real,args.train_samples,
                                   seed+23,args.ot_window,args.epsilon,args.iterations)
            obj=-losses["repulsion"] if arm=="negative_repulsion" else losses[arm]
            direction,gnorm=unit_loss_gradient(obj,params)
            loss_values={k:float(v.detach()) for k,v in losses.items()}
            del losses,obj,cond,real
            policy.model.requires_grad_(False)
            for h in args.step_norms:
                plus,minus=symmetric_scores(
                    params,direction,h,
                    lambda:score_policy(policy,cases,args.windows,args.probes,args.eval_seed))
                for window in args.windows:
                    row={"train_seed":seed,"arm":arm,"window":window,
                         "update_norm":h,"loss_gradient_norm":gnorm,
                         "eval_seed":args.eval_seed,
                         "eval_conditions":len(cases),
                         "eval_samples":args.eval_samples,
                         "probes":args.probes,**loss_values}
                    for metric in original[window]:
                        details=comparison(original[window][metric],
                                           plus[window][metric],
                                           minus[window][metric],h)
                        row["baseline_"+metric]=original[window][metric]
                        for key,v in details.items():
                            row[key+"_"+metric]=v
                    rows.append(row)
                    print(f"seed={seed} {arm:20s} {window:8s} "
                          f"h={h:.4g} "
                          f"D_uSz={row['directional_derivative_jacobian_sq_per_noise_dim']:+.7g} "
                          f"pred={row['delta_predicted_jacobian_sq_per_noise_dim']:+.7g} "
                          f"actual={row['delta_observed_jacobian_sq_per_noise_dim']:+.7g}",
                          flush=True)
            del direction
    write_csv(args.output,rows)
    print("Reminder: positive D_u Sz means gradient descent DECREASES Sz;"
          " these are directional derivatives, not rho_J.",flush=True)


if __name__=="__main__":
    main()
