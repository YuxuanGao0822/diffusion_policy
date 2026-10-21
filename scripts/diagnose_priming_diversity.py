#!/usr/bin/env python3
"""Compare noise sensitivity of trusted Stage-1 / Stage-2 PushT checkpoints.

Run from the diffusion_policy repository root. Only open trusted dill ckpts.
Uses training/validation dataset observations; does not run PushT rollouts.
"""
import argparse
import csv
import gc
from pathlib import Path

import dill
import hydra
import numpy as np
import torch


@torch.no_grad()
def score_samples(pred, real, start, steps):
    """pred [K,H,Da] raw normalized outputs; real [H,Da] expert actions."""
    k = pred.shape[0]
    if k < 2:
        raise ValueError("need at least two noise samples")
    mean = pred.mean(dim=0)
    err = pred - real.unsqueeze(0)
    # Pair RMS over distinct noise samples, excluding i=j.
    pair_sq = (2*k/(k-1) * (pred-mean).square().sum((-1, -2)).mean()).item()
    expert_sq = err.square().sum((-1, -2)).mean().item()
    return {
        "noise_std": (pred-mean).square().mean().sqrt().item(),
        "pair_rms": pair_sq**0.5,
        "expert_rms": expert_sq**0.5,
        "pair_vs_expert_sq": pair_sq/max(expert_sq, 1.e-12),
        "attraction_proxy": expert_sq/max(pair_sq+expert_sq, 1.e-12),
        "sample_expert_mse": err.square().mean().item(),
        "mean_expert_mse": (mean-real).square().mean().item(),
        "clip_rate_full": (pred.abs()>1).float().mean().item(),
        "clip_rate_executed": (pred[:,start:start+steps].abs()>1).float().mean().item(),
    }


def load_checkpoint(path, device, weights):
    # dill can execute code: ONLY inspect your own trusted checkpoint files.
    payload = torch.load(path, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    model = hydra.utils.instantiate(cfg.policy)
    model.load_state_dict(payload["state_dicts"][weights], strict=True)
    model.to(device).eval()
    del payload
    return cfg, model


@torch.no_grad()
def evaluate(model, ds, indices, k, chunk, seed, split, label, device):
    results = []
    for index in indices:
        item = ds[int(index)]
        obs = item["obs"].unsqueeze(0).to(device)
        expert = model.normalizer["action"].normalize(
            item["action"].unsqueeze(0).to(device))[0].cpu()
        condition = model._condition(model.normalizer["obs"].normalize(obs))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed+int(index)+(100000 if split=="val" else 0))
        generated = []
        for offset in range(0,k,chunk):
            count = min(chunk,k-offset)
            z = torch.randn(
                count, model.horizon, model.action_dim,
                generator=generator, dtype=condition.dtype).to(device)
            generated.append(model._generate(z,condition.expand(count,-1)).cpu())
        pred = torch.cat(generated)
        start = model.n_obs_steps-1 if model.oa_step_convention else model.n_obs_steps
        stats = score_samples(pred,expert,start,model.n_action_steps)
        results.append({"checkpoint":label,"split":split,"sample_index":int(index),**stats})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",nargs="+",required=True)
    parser.add_argument("--names",nargs="+",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--weights",choices=("ema_model","model"),default="ema_model")
    parser.add_argument("--conditions",type=int,default=24)
    parser.add_argument("--samples",type=int,default=64)
    parser.add_argument("--noise-batch",type=int,default=32)
    parser.add_argument("--seed",type=int,default=20261021)
    parser.add_argument("--output",default="data/outputs/priming_pusht/diversity_audit.csv")
    args=parser.parse_args()
    if len(args.names)!=len(args.checkpoint):
        parser.error("names and checkpoints must have equal length")
    if args.conditions<1 or args.samples<2 or args.noise_batch<1:
        parser.error("invalid conditions/samples/noise-batch")
    device=torch.device(args.device)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path,label in zip(args.checkpoint,args.names):
        path=Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        print("Evaluating",label,path,flush=True)
        cfg,model=load_checkpoint(str(path),device,args.weights)
        dataset=hydra.utils.instantiate(cfg.task.dataset)
        val_dataset=dataset.get_validation_dataset()
        for split,ds in (("train",dataset),("val",val_dataset)):
            if len(ds)==0:
                print(split,"empty; skipping")
                continue
            count=min(len(ds),args.conditions)
            idx=np.random.RandomState(args.seed+(split=="val")).choice(
                len(ds),size=count,replace=False)
            current=evaluate(model,ds,idx,args.samples,args.noise_batch,
                             args.seed,split,label,device)
            rows+=current
            print(split,"mean noise_std=",np.mean([x["noise_std"] for x in current]),
                  "mean pair/expert²=",np.mean([x["pair_vs_expert_sq"] for x in current]),
                  "clip executed=",np.mean([x["clip_rate_executed"] for x in current]),
                  flush=True)
        del cfg,model,dataset,val_dataset
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("no examples evaluated")
    with output.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved",output)
    print("Compare checkpoints on same split: noise_std, pair_vs_expert_sq,"
          " sample_expert_mse, and clip_rate_executed.")


if __name__=="__main__":
    main()
