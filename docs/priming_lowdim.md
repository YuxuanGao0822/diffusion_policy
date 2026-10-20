# Priming Policy on PushT lowdim

Implementation target: YuxuanGao0822/diffusion_policy (fork of real-stanford/diffusion_policy).
This is a **new experimental policy**, not an upstream Priming Models reproduction or a
validated robotics benchmark. The original Diffusion Policy remains untouched.

## Requirements

Use the repository's existing conda environment and dataset. In particular, prepare
`data/pusht/pusht_cchi_v7_replay.zarr`, as referenced by
`diffusion_policy/config/task/pusht_lowdim.yaml`. This patch adds no third-party
library requirements beyond the original project.

## Stage 1: fixed-noise reconstruction

From the repository root:

```bash
python train.py --config-name=train_priming_lowdim_workspace \
  training.stage=pretrain training.num_epochs=500 \
  logging.mode=disabled
```

It trains the same conditional U-Net to predict normalized clean action chunks
from `(1 - s) * action + s * noise` (default `s=0.8`). It uses a single
constant time embedding and does not use a DDPM noise scheduler. Use the path
to its `checkpoints/latest.ckpt` for Stage 2.

For a two-epoch pipeline check that still runs the normal env runner:

```bash
python train.py --config-name=train_priming_lowdim_workspace \
  training.stage=pretrain training.debug=true logging.mode=disabled
```

## Stage 2: conditional OT post-training

Start **a new output directory** and initialize from the Stage-1 checkpoint:

```bash
python train.py --config-name=train_priming_lowdim_workspace \
  training.stage=posttrain \
  training.init_checkpoint=/ABS/PATH/TO/PRETRAIN/checkpoints/latest.ckpt \
  training.init_weights=ema \
  training.learning_rate=1e-5 training.num_epochs=200 \
  dataloader.batch_size=16 val_dataloader.batch_size=16 \
  training.rollout_every=10 training.checkpoint_every=10 \
  training.lr_warmup_steps=100 \
  logging.mode=disabled
```

For Stage-2 smoke checks, append `training.debug=true` and optionally
`policy.samples_per_condition=2 dataloader.batch_size=4` to reduce memory.
Use `training.resume=false` to force a new run even when the output directory
contains a checkpoint; by default a Stage-2 job resumes its own latest
checkpoint if present.

The Stage-2 loss matches independently generated batches **within each
observation group**. For one demonstration per observation (the standard
PushT sequence setup), generated-to-real OT simplifies to a uniform mean
squared feature distance. The other term is a Sinkhorn transport between
active generated and independently drawn stop-gradient generated samples.
The optional temporal difference feature loss has weight
`policy.delta_weight`, initially 0.0 (raw action features only).

Distributional loss is estimated with independent Gaussian seeds for each
observation; training uses K active and K reference action chunks per
observation (`policy.samples_per_condition`, default K=4). Consequently, the
Stage-2 working batch is far larger than the dataset batch. Reduce the
Stage-2 dataloader batch size to avoid OOM.

## Testing

```bash
python -m pytest -q tests/test_priming_lowdim.py
```

This uses a tiny test generator on CPU and verifies:
- correct Sinkhorn marginals and stop gradients;
- backward passes for Stage 1 / Stage 2;
- a **single** generator forward for `predict_action()`;
- PushT action chunk slicing (`n_obs_steps-1 : n_obs_steps-1+n_action_steps`).

## Evaluation / baselines

The existing `PushTKeypointsRunner` is reused by the inherited workspace.
Compare `test_mean_score` with the original Diffusion Policy under the
same dataset/splits and evaluation seeds. Report policy forward-pass count,
wall-clock inference latency, and success/coverage metrics. The Stage-1
checkpoint already generates one-step actions but may produce poor rollouts.

## Limitations

- The noise level `s=0.8` and OT settings are **initial hypotheses** for
  robot actions, not experimentally optimized values.
- One action demonstration for each continuous observation does **not**
  identify the full conditional action distribution; attractive matching may
  still encourage conditional mode averaging. The self-interaction is not a
  guarantee against mode collapse.
- This MVP only supports global observation conditioning for PushT lowdim;
  no inpainting, visual encoder, action history or variable-length masks.
- The normalized generator output is clipped only at inference. Stage-2
  optimization uses unclipped outputs.
- Pre-/post-training switch uses fresh optimizer and stage-specific output
  folders. Post-training copies either online or EMA pretraining weights into
  *both* the new online and EMA models (default source: EMA).
- The original inherited workspace calculates supervised MSE on a sampled
  rollout chunk for logging. For this stochastic generator the statistic is
  not a proper conditional generation metric.
- This implementation has not been benchmarked on the actual PushT Zarr
  dataset or GPU; such a run is required before claiming task performance.
