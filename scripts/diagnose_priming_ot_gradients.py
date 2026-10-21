#!/usr/bin/env python3
"""Audit OT and MSE parameter-gradient directions at trusted PushT checkpoints.

Measures gradients of the EXACT detached-scale, detached-Sinkhorn OT
surrogate used by ConditionalActionOTLoss, plus the Stage-1 noisy
reconstruction MSE, at fixed matched observations and noise seeds.

NOTE: Reports a local gradient diagnostic at EMA checkpoints, not the
actual historical online optimizer updates or closed-loop reward gradients.
Run from diffusion_policy root and load ONLY checkpoints you trust (dill).
"""
import argparse
import csv
import gc
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dill
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from diffusion_policy.model.priming.distribution_loss import (
    _half_squared_distances, uniform_sinkhorn_plan,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


def differentiable_ot_components(live, reference, expert, epsilon, iterations):
    """Mirror the production quadratic OT branch, including stop-gradient.

    live/reference: [B,K,H,Da]; expert: [B,1,H,Da].
    Returned A and R remain differentiable w.r.t. live; ref and OT plan do not.
    """
    b, k = live.shape[:2]
    assert reference.shape == live.shape
    assert expert.shape == (b, 1, *live.shape[-2:])
    x = live.reshape(b, k, -1)
    y = reference.detach().reshape(b, k, -1)
    target = expert.detach().reshape(b, 1, -1)
    with torch.no_grad():
        raw_cross = _half_squared_distances(x.detach(), target)
        raw_self = _half_squared_distances(x.detach(), y)
        squared_rms = raw_cross.mean() + raw_self.mean()
        scale = squared_rms.clamp_min(1.e-6).sqrt()
    cross = _half_squared_distances(x / scale, target / scale)
    self_cost = _half_squared_distances(x / scale, y / scale)
    attraction = cross.mean()
    plan = uniform_sinkhorn_plan(self_cost, epsilon, iterations)
    repulsion = (plan * self_cost).sum(dim=(-2, -1)).mean()
    return attraction, repulsion


def _dot(left, right):
    value = 0.0
    for a, b in zip(left, right):
        if a is not None and b is not None:
            value += float((a.detach().double() * b.detach().double()).sum())
    return value


def gradient_alignment_metrics(attraction, repulsion, mse, parameters):
    """Compare g_A, the repulsive loss gradient -g_R, and g_MSE.

    The loss is A - R. A positive attraction_vs_repulsive_cos means
    the two contributions to the *loss* gradient align.
    """
    params = list(parameters)
    ga = torch.autograd.grad(attraction, params, retain_graph=True,
                             allow_unused=True)
    gr = torch.autograd.grad(repulsion, params, allow_unused=True)
    gm = torch.autograd.grad(mse, params, allow_unused=True)
    aa, rr, ar = _dot(ga, ga), _dot(gr, gr), _dot(ga, gr)
    mm, am, rm = _dot(gm, gm), _dot(ga, gm), _dot(gr, gm)
    na, nr, nm = math.sqrt(max(aa, 0.0)), math.sqrt(max(rr, 0.0)), math.sqrt(max(mm, 0.0))
    no = math.sqrt(max(aa + rr - 2.0 * ar, 0.0))
    eps = 1e-30
    return {
        "grad_attraction_norm": na,
        "grad_repulsion_norm": nr,
        "grad_mse_norm": nm,
        "grad_ot_norm": no,
        "repulsion_to_attraction_grad_norm": nr / max(na, eps),
        "attraction_vs_repulsive_cos": -ar / max(na * nr, eps),
        "ot_vs_mse_cos": (am - rm) / max(no * nm, eps),
    }


def seeded_noise(shape, seed, dtype, device):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return torch.randn(shape, generator=rng, dtype=dtype).to(device)


def load_policy(checkpoint, device):
    # Loading dill checkpoints executes code. Restrict to files YOU created.
    payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    # Stage-1 snapshots were saved with stage='pretrain'; instantiate
    # the same network in posttrain mode to make OT comparisons symmetric.
    policy = hydra.utils.instantiate(cfg.policy, stage="posttrain")
    policy.load_state_dict(payload["state_dicts"]["ema_model"], strict=True)
    policy.to(device).eval()
    del payload
    return cfg, policy


def one_batch(policy, items, k, epsilon, iterations, window, seed, device):
    obs = torch.stack([x["obs"] for x in items]).to(device)
    action = torch.stack([x["action"] for x in items]).to(device)
    nobs = policy.normalizer["obs"].normalize(obs)
    expert = policy.normalizer["action"].normalize(action)
    condition = policy._condition(nobs)
    b, h, da = expert.shape
    cond = condition.repeat_interleave(k, dim=0)
    live_noise = seeded_noise((b*k,h,da), seed, expert.dtype, device)
    ref_noise = seeded_noise((b*k,h,da), seed + 1, expert.dtype, device)
    live = policy._generate(live_noise, cond).reshape(b,k,h,da)
    with torch.no_grad():
        reference = policy._generate(ref_noise, cond).reshape(b,k,h,da)
    target = expert.unsqueeze(1)
    if window == "executed":
        start = policy.n_obs_steps - 1 if policy.oa_step_convention else policy.n_obs_steps
        sl = slice(start, start + policy.n_action_steps)
        live = live[:,:,sl,:]
        reference = reference[:,:,sl,:]
        target = target[:,:,sl,:]

    attraction, repulsion = differentiable_ot_components(
        live, reference, target, epsilon, iterations)

    # Match the original Stage-1 reconstruction loss (NOT pure BC).
    s = policy.noise_level
    recon_noise = seeded_noise((b,h,da), seed+2, expert.dtype, device)
    noisy = (1.0-s) * expert + s * recon_noise
    recon = policy._generate(noisy, condition)
    mse = F.mse_loss(recon, expert)

    params = [p for p in policy.model.parameters() if p.requires_grad]
    geometry = gradient_alignment_metrics(attraction, repulsion, mse, params)
    del live, reference, target, recon, params
    return {
        "ot_attraction": float(attraction.detach()),
        "ot_repulsion": float(repulsion.detach()),
        "ot_total": float((attraction-repulsion).detach()),
        "reconstruction_mse": float(mse.detach()),
        **geometry,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", nargs="+", required=True,
                   help="trusted E350 and posttrain checkpoint paths")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conditions", type=int, default=16,
                   help="number of randomly selected conditions per data split")
    p.add_argument("--batch-size", type=int, default=2,
                   help="conditions per gradient evaluation; use 1-4 initially")
    p.add_argument("--samples", type=int, default=8,
                   help="K live and K stop-gradient references per condition")
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--window", choices=("full","executed"), default="full")
    p.add_argument("--seed", type=int, default=20261021)
    p.add_argument("--output", default="data/outputs/priming_pusht/ot_gradient_audit.csv")
    args = p.parse_args()
    if len(args.names) != len(args.checkpoint):
        p.error("one name is required for each checkpoint")
    if args.conditions < 1 or args.batch_size < 1 or args.samples < 2:
        p.error("conditions/batch-size must be positive; samples must be >=2")
    device = torch.device(args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ckpt, label in zip(args.checkpoint, args.names):
        ckpt = Path(ckpt).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        print("Gradient audit:", label, ckpt, flush=True)
        cfg, policy = load_policy(str(ckpt), device)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.get_validation_dataset()
        for split, ds in (("train",dataset),("val",val_dataset)):
            if len(ds) == 0:
                continue
            n = min(args.conditions, len(ds))
            # Deterministic index sets and Gaussian noise across checkpoints.
            indices = np.random.RandomState(args.seed+(split=="val")).choice(
                len(ds), size=n, replace=False)
            for i in range(0, n, args.batch_size):
                ids = indices[i:i+args.batch_size]
                seed = args.seed + (100000 if split=="val" else 0) + i
                metrics = one_batch(
                    policy, [ds[int(j)] for j in ids], args.samples,
                    args.epsilon, args.iterations, args.window, seed, device)
                rows.append({
                    "checkpoint": label, "split": split, "batch": i//args.batch_size,
                    "conditions_in_batch": len(ids), "window": args.window,
                    **metrics,
                })
                print(f"  {split} batch {i//args.batch_size}: "
                      f"||gR||/||gA||={metrics['repulsion_to_attraction_grad_norm']:.5f} "
                      f"cos(gOT,gMSE)={metrics['ot_vs_mse_cos']:.4f}", flush=True)
                policy.zero_grad(set_to_none=True)
        del policy, dataset, val_dataset, cfg
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("no conditions evaluated")
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved", out)
    for label in args.names:
        for split in ("train","val"):
            group = [r for r in rows if r["checkpoint"]==label and r["split"]==split]
            if not group:
                continue
            for key in ("repulsion_to_attraction_grad_norm",
                        "attraction_vs_repulsive_cos","ot_vs_mse_cos",
                        "grad_ot_norm", "reconstruction_mse"):
                print(f"{label:18s} {split:5s} {key:40s} "
                      f"{np.mean([r[key] for r in group]):+.6f}")


if __name__=="__main__":
    main()
