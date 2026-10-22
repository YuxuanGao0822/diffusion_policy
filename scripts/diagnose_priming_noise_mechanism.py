#!/usr/bin/env python3
"""Noise-input Jacobian and controlled one-step Stage-2 gradient interventions.

Reads trusted Stage-1/Stage-2 EMA dill checkpoints; never overwrites them.
A: Hutchinson VJP estimate of ||dG(z,o)/dz||_F^2/dim(z).
B: identical Stage-1 initialization, train batch, random seeds, and validation
conditions: no update, attraction, -repulsion, OT, or Stage-1 denoising MSE.
Unclipped actions in training, deployment clipping in reported sensitivity.
These are local diagnostics, not PushT reward or proof of useful diversity.
"""
import argparse
import csv
import gc
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F

from scripts.diagnose_priming_ot_gradients import (
    load_policy, seeded_noise, differentiable_ot_components,
)


def window_slice(policy, window):
    if window == "full":
        return slice(None)
    if window != "executed":
        raise ValueError("window must be full or executed")
    start = policy.n_obs_steps - 1 if policy.oa_step_convention else policy.n_obs_steps
    return slice(start, start + policy.n_action_steps)


def sensitivity_from_function(generate, noise, window, probes=4, seed=1,
                              target=None):
    """Hutchinson VJP; v entries are Rademacher +/-1 over *output* space.

    The denominator uses all input noise dimensions, also for executed-only
    outputs. Each sample must be independent of other examples in the batch.
    """
    if probes < 1 or noise.ndim != 3 or noise.shape[0] < 2:
        raise ValueError("need >=2 noise samples and >=1 Hutchinson probe")
    z = noise.detach().clone().requires_grad_(True)
    y = generate(z)[:, window, :]
    if y.ndim != 3:
        raise ValueError("generate must return [K,H,Da]")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    norm2 = []
    for j in range(probes):
        v = torch.randint(0, 2, y.shape, generator=rng,
                          dtype=torch.int8).to(y.device, dtype=y.dtype)
        v = v.mul_(2).sub_(1)
        jtv = torch.autograd.grad((y * v).sum(), z,
                                  retain_graph=(j+1 < probes))[0]
        norm2.append(jtv.flatten(1).square().sum(dim=1).mean())
    out = {
        "jacobian_sq_per_noise_dim":
            float((torch.stack(norm2).mean() / z[0].numel()).detach()),
        "noise_std": float((y.detach()-y.detach().mean(0))
                           .square().mean().sqrt()),
    }
    if target is not None:
        if tuple(target.shape) != tuple(y.shape[1:]):
            raise ValueError("expert target shape mismatch")
        out["pure_noise_expert_mse"] = float(
            (y.detach() - target.unsqueeze(0)).square().mean())
    return out


def pick_indices(ds, n, seed):
    if not len(ds):
        raise ValueError("cannot evaluate an empty dataset split")
    return np.random.RandomState(seed).choice(len(ds), min(len(ds), n),
                                              replace=False)


def condition_and_target(policy, item, device):
    obs = item["obs"].unsqueeze(0).to(device)
    expert = item["action"].unsqueeze(0).to(device)
    with torch.no_grad():
        condition = policy._condition(
            policy.normalizer["obs"].normalize(obs))
        expert = policy.normalizer["action"].normalize(expert)
    return condition, expert


def evaluate_item(policy, item, window, samples, probes, seed, device):
    cond, target = condition_and_target(policy, item, device)
    noise = seeded_noise((samples, policy.horizon, policy.action_dim),
                         seed, cond.dtype, device)
    sl = window_slice(policy, window)

    def generated(z):
        y = policy._generate(z, cond.expand(z.shape[0], -1))
        return y.clamp(-1., 1.) if policy.clip_sample else y

    return sensitivity_from_function(generated, noise, sl, probes,
                                     seed+33, target[0, sl, :])


def evaluate(policy, dataset, indices, window, samples, probes, seed, device):
    rows = []
    for idx in indices:
        result = evaluate_item(policy, dataset[int(idx)], window, samples,
                               probes, seed+int(idx), device)
        rows.append({"sample_index": int(idx), **result})
    keys = ("jacobian_sq_per_noise_dim", "noise_std",
            "pure_noise_expert_mse")
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return means, rows


def make_train_batch(policy, ds, indices, device):
    items = [ds[int(i)] for i in indices]
    obs = torch.stack([x["obs"] for x in items]).to(device)
    act = torch.stack([x["action"] for x in items]).to(device)
    cond = policy._condition(policy.normalizer["obs"].normalize(obs))
    act = policy.normalizer["action"].normalize(act)
    return cond, act


def loss_components(policy, cond, real, k, seed, window, epsilon, iterations):
    """Exactly mirror production quadratic OT: detached feature scale and plan."""
    b, h, da = real.shape
    cc = cond.repeat_interleave(k, dim=0)
    live_z = seeded_noise((b*k, h, da), seed, real.dtype, real.device)
    ref_z = seeded_noise((b*k, h, da), seed+1, real.dtype, real.device)
    live = policy._generate(live_z, cc).reshape(b, k, h, da)
    with torch.no_grad():
        ref = policy._generate(ref_z, cc).reshape(b, k, h, da)
    sl = window_slice(policy, window)
    attraction, repulsion = differentiable_ot_components(
        live[:, :, sl, :], ref[:, :, sl, :], real[:, None, sl, :],
        epsilon, iterations)
    corruption_noise = seeded_noise((b, h, da), seed+2,
                                    real.dtype, real.device)
    s = policy.noise_level
    recon = policy._generate((1-s)*real+s*corruption_noise, cond)
    mse = F.mse_loss(recon, real)
    return {"attraction": attraction, "repulsion": repulsion,
            "ot": attraction-repulsion, "mse": mse}


def take_sgd_step(parameters, loss, lr, matched_step_norm=None):
    """One no-momentum SGD update; does not modify source checkpoints."""
    params = list(parameters)
    for p in params:
        p.grad = None
    loss.backward()
    squared = sum(float(p.grad.detach().float().square().sum())
                  for p in params if p.grad is not None)
    gn = math.sqrt(squared)
    if not math.isfinite(gn):
        raise FloatingPointError("non-finite parameter gradient")
    if matched_step_norm is not None and gn == 0.:
        raise ValueError("cannot match step norm when gradient is zero")
    eta = matched_step_norm/gn if matched_step_norm is not None else lr
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p.add_(p.grad, alpha=-eta)
                p.grad = None
    return {"gradient_norm": gn, "learning_rate_applied": eta,
            "parameter_update_l2": gn*eta}


def write_csv(path, rows):
    if not rows:
        raise ValueError("no result rows")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path, flush=True)


def run_jacobian(args, device):
    if len(args.checkpoint) != len(args.names):
        raise ValueError("--checkpoint and --names lengths differ")
    rows = []
    for path, label in zip(args.checkpoint, args.names):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        cfg, policy = load_policy(path, device)
        policy.eval().requires_grad_(False)  # differentiate only input noise
        train = hydra.utils.instantiate(cfg.task.dataset)
        splits = {"train": train, "val": train.get_validation_dataset()}
        for split in args.split:
            ds = splits[split]
            seed = args.seed+(100000 if split=="val" else 0)
            indices = pick_indices(ds, args.conditions, seed)
            for window in args.window:
                avg, details = evaluate(policy, ds, indices, window,
                    args.samples, args.probes, seed, device)
                rows.extend({"checkpoint": label, "split": split,
                             "window": window, **r} for r in details)
                print(f"{label:22s} {split:5s} {window:8s} "
                      f"Sz={avg['jacobian_sq_per_noise_dim']:.8g} "
                      f"std={avg['noise_std']:.8g} "
                      f"expert_mse={avg['pure_noise_expert_mse']:.8g}",
                      flush=True)
        del cfg, policy, train, splits
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
    write_csv(args.output, rows)


def run_intervention(args, device):
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    cfg, initial = load_policy(args.checkpoint, device)
    train = hydra.utils.instantiate(cfg.task.dataset)
    val = train.get_validation_dataset()
    train_ids = pick_indices(train, args.train_conditions, args.seed)
    val_ids = pick_indices(val, args.conditions, args.seed+100000)
    before = {}
    for window in args.window:
        before[window], _ = evaluate(initial, val, val_ids, window,
            args.samples, args.probes, args.seed+100000, device)
    del cfg, initial
    gc.collect()
    if device.type=="cuda":
        torch.cuda.empty_cache()

    arms = ("control", "attraction", "negative_repulsion", "ot", "mse")
    rows = []
    for arm in arms:
        _, policy = load_policy(args.checkpoint, device)
        policy.eval()
        cond, real = make_train_batch(policy, train, train_ids, device)
        update = {"gradient_norm": 0., "learning_rate_applied": 0.,
                  "parameter_update_l2": 0.}
        loss_vals = {}
        if arm != "control":
            losses = loss_components(policy, cond, real, args.train_samples,
                args.seed+23, args.ot_window, args.epsilon, args.iterations)
            loss_vals = {k: float(v.detach()) for k,v in losses.items()}
            objective = -losses["repulsion"] if arm=="negative_repulsion" else losses[arm]
            update = take_sgd_step(policy.model.parameters(), objective,
                args.lr, args.matched_step_norm)
            del objective, losses
        policy.model.requires_grad_(False)
        for window in args.window:
            avg, _ = evaluate(policy, val, val_ids, window,
                args.samples, args.probes, args.seed+100000, device)
            row = {"arm": arm, "window": window,
                   "train_conditions": len(train_ids),
                   "train_samples": args.train_samples,
                   "ot_window": args.ot_window, **loss_vals, **update}
            for k, v in before[window].items():
                row["before_"+k] = v
                row["after_"+k] = avg[k]
                row["delta_"+k] = avg[k]-v
            rows.append(row)
            print(f"{arm:20s} {window:8s} "
                  f"update={update['parameter_update_l2']:.6g} "
                  f"delta_Sz={row['delta_jacobian_sq_per_noise_dim']:+.7g} "
                  f"delta_std={row['delta_noise_std']:+.7g}",
                  flush=True)
        del policy, cond, real
        gc.collect()
        if device.type=="cuda":
            torch.cuda.empty_cache()
    write_csv(args.output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    j = sub.add_parser("jacobian")
    j.add_argument("--checkpoint", nargs="+", required=True)
    j.add_argument("--names", nargs="+", required=True)
    j.add_argument("--split", nargs="+", choices=("train","val"), default=["val"])
    i = sub.add_parser("intervene")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--train-conditions", type=int, default=2)
    i.add_argument("--train-samples", type=int, default=8)
    i.add_argument("--ot-window", choices=("full","executed"), default="full")
    i.add_argument("--epsilon", type=float, default=0.05)
    i.add_argument("--iterations", type=int, default=50)
    i.add_argument("--lr", type=float, default=1e-5)
    i.add_argument("--matched-step-norm", type=float, default=None)
    for cmd in (j,i):
        cmd.add_argument("--window", nargs="+", choices=("full","executed"),
                         default=["full","executed"])
        cmd.add_argument("--conditions", type=int, default=8)
        cmd.add_argument("--samples", type=int, default=8)
        cmd.add_argument("--probes", type=int, default=4)
        cmd.add_argument("--seed", type=int, default=20261021)
        cmd.add_argument("--device", default="cuda:0")
        cmd.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.conditions<1 or args.samples<2 or args.probes<1:
        parser.error("conditions>0, samples>=2, probes>=1 required")
    if args.mode=="intervene":
        if args.train_conditions<1 or args.train_samples<2:
            parser.error("train_conditions>0, train_samples>=2 required")
        if args.lr<=0:
            parser.error("lr must be >0")
        if args.matched_step_norm is not None and args.matched_step_norm<=0:
            parser.error("matched-step-norm must be >0")
    device = torch.device(args.device)
    if args.mode=="jacobian":
        run_jacobian(args, device)
    else:
        run_intervention(args, device)


if __name__=="__main__":
    main()
