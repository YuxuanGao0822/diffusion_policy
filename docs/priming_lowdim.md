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

## Weights & Biases (W&B) experiment tracking

This policy's Hydra config enables `logging.mode=online` by default. The
`logging.project` is `priming_policy_pusht_lowdim`; all runs are grouped by
`logging.group=priming_pusht_lowdim` and labeled via `logging.job_type`
(`pretrain` or `posttrain`). Override these properties from the CLI.

For initial setup on the **training server**, authenticate interactively:

```bash
conda activate robodiff
wandb login --relogin
unset WANDB_MODE WANDB_DISABLED WANDB_OFFLINE
export CUDA_VISIBLE_DEVICES=7
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
```

Do **not** put W&B API keys in Hydra configs, training scripts, commit messages
or version control. Revoke any W&B API key previously shared publicly and log
in again with a newly generated key.

Stage 1 (A100/A800 80 GB starting configuration):

```bash
python train.py --config-name=train_priming_lowdim_workspace \
  training.stage=pretrain training.device=cuda:0 \
  training.num_epochs=1500 training.resume=false \
  training.learning_rate=1e-4 dataloader.batch_size=512 \
  training.rollout_every=50 training.checkpoint_every=50 \
  logging.mode=online \
  hydra.run.dir=data/outputs/priming_pusht/stage1_wandb
```

Stage 2 (after Stage 1 completes, with a fresh output directory):

```bash
python train.py --config-name=train_priming_lowdim_workspace \
  training.stage=posttrain training.device=cuda:0 \
  training.init_checkpoint="$(pwd)/data/outputs/priming_pusht/stage1_wandb/checkpoints/latest.ckpt" \
  training.init_weights=ema training.resume=false \
  training.num_epochs=120 training.learning_rate=1e-5 \
  dataloader.batch_size=64 val_dataloader.batch_size=32 \
  policy.samples_per_condition=16 policy.ot_iterations=50 \
  training.rollout_every=10 training.checkpoint_every=10 \
  logging.mode=online \
  hydra.run.dir=data/outputs/priming_pusht/stage2_wandb
```

With `CUDA_VISIBLE_DEVICES=7`, PyTorch sees physical GPU 7 as `cuda:0`.
These batch sizes are initial proposals, not guaranteed to fit every GPU
or task configuration. Reduce Stage-2 `dataloader.batch_size` if needed.

Additional W&B/JSON scalar fields (alongside the existing `train_loss`,
`val_loss`, `lr`, `test_mean_score`, etc.):

| Stage | Every training minibatch | Validation mean per epoch |
| --- | --- | --- |
| Pretrain | `train/recon_mse` | `val/recon_mse` |
| Posttrain | `train/ot_raw`, `train/ot_attraction`, `train/ot_repulsion`, `train/ot_total` | `val/ot_raw`, `val/ot_attraction`, `val/ot_repulsion`, `val/ot_total` |
| Posttrain, `delta_weight>0` | `train/ot_delta` | `val/ot_delta` |

The training metrics are logged per minibatch; validation component metrics
are averaged across validation minibatches at the configured `val_every`
interval. The existing `test_mean_score` is available only on epochs where
`rollout_every` triggers. As the inherited Top-K checkpoint selector expects
`test_mean_score`, keep `checkpoint_every` aligned with `rollout_every`
(or smaller rollout interval that divides the checkpoint interval).

These metrics are diagnostic and do not imply successful PushT training.

## Diagnosing near-unit post-training loss and testing Energy Distance

Stage-2 OT near 1 is **not** by itself an implementation error. The detached
pair-RMS scale uses generated-to-expert cost plus generated-to-independent-
reference cost; when generated diversity is small relative to expert error,
the attraction term approaches 1 and Sinkhorn self-interaction can be small.
The PushT condition is a continuous observation, with one paired action
sequence per dataset occurrence, so per-condition cross-OT uses a single
expert target. This is a structural identifiability limit when transferring
the conditional distribution objective from images/video to lowdim actions.

Use a held-out checkpoint diversity audit before interpreting OT loss as
mode collapse. On identical observation batches and identical Gaussian seeds,
compare Stage-1 EMA against Stage-2 EMA, measuring
generated_pair_vs_expert_sq_ratio, within_noise_coordinate_std,
sample_to_expert_mse, and output_clipped_fraction_executed_steps.
The trained checkpointer uses dill: load ONLY trusted checkpoint files.

A **paper-motivated ablation** is enabled by:

```bash
policy.distribution_objective=energy
```

This selects the energy-distance surrogate from Priming Models Appendix B.2:
mean Euclidean generated-to-expert distance **minus** mean Euclidean
generated-to-independent-reference distance (reference is stop-gradient).
The same detached pair-RMS scaling is retained. Training logs use
`train/energy_attraction`, `train/energy_repulsion`, and
`train/energy_total` (and `val/energy_*`) instead of `ot_*`.
`policy.ot_epsilon` and `policy.ot_iterations` do not affect the energy
objective. The original `distribution_objective=ot` is **unchanged**.

Start from the SAME Stage-1 checkpoint and run a new independent output
directory with the same seed, learning rate, batch size, K, and evaluation
settings to isolate the choice of objective. Use PushT rollout score as the
primary outcome; energy loss values and OT loss values are on different scales
and are not directly comparable. This option is exploratory, **not** a
validated improvement. Consider 3 independent seeds for a final comparison.

### Checkpoint diversity audit on your server

Before running another lengthy post-training experiment, inspect the SAME
training and validation observations under Stage-1 and Stage-2 EMA weights:

```bash
python scripts/diagnose_priming_diversity.py \
  --checkpoint "$STAGE1_CKPT" "$STAGE2_CKPT" \
  --names stage1_e350 stage2_latest \
  --conditions 24 --samples 64 --noise-batch 32 \
  --device cuda:0 \
  --output data/outputs/priming_pusht/diversity_audit.csv
```

Use actual absolute checkpoint paths for STAGE1_CKPT and STAGE2_CKPT.
The audit produces per-condition rows for both train and val with noise
sensitivity, generated pair distance, sample/expert error, and fraction of
normalized actions outside the inference clamp [-1,1]. This is not a
benchmark rollout and does not prove causal reasons for score changes.
It uses matched observation indices and Gaussian noise across checkpoints.


### Ablation: train distribution loss only on deployed actions

The default `policy.distribution_window=full` matches all 16 predicted
actions (indices 0..15). However, in the current PushT lowdim configuration,
receding-horizon inference executes only 8 actions (indices 1..8), because
`n_obs_steps=2`, `n_action_steps=8`, and OA convention is enabled.

For an isolated control-aware OT ablation, use:

```bash
policy.distribution_window=executed
```

This computes distributional alignment and self-interaction **only on the
actions actually deployed** by `predict_action`; the U-Net still predicts the
full sequence so the architecture and rollout interface are unchanged.
This may improve task relevance, but also discards supervision on the
unexecuted context/tail and is not established to improve scores.

Compare from the same Stage-1 EMA checkpoint, same noise samples per
condition, learning rate, seed and rollout schedule. Save into a **new**
Hydra run directory. This switch is independent of
`policy.distribution_objective=ot|energy`; keep `ot` fixed for this test.
Use the updated checkpoint diversity audit's `*_executed` metrics to
compare matched condition errors on the deployed action window.
