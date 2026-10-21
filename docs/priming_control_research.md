# Priming Policy: four-phase automatic control research

**Research prototype, not a validated benchmark.** Run only with checkpoints
you trust: checkpoint loading uses Python/dill deserialization.
All stages stop immediately on errors and preserve intermediate runs.

## Study design

0. Calibrate the frozen Lowdim E350 EMA baseline with held-out PushT seeds.
1. **Frozen Residual RL:** freeze the one-NFE Priming policy; optimize a small
   bounded residual actor and twin-Q critic on real (simulator) transitions.
2. **Anchored Q-Guided Priming:** initialize a trainable U-Net from the
   identical E350 teacher; continue with critic from phase 1 and online
   replay. Actor loss = -min(Q1,Q2) + lambda * MSE(actor, frozen teacher).
3. **Task-aware OT:** initialize from phase-2 U-Net and critic; train a
   frozen-observation-feature forward dynamics predictor on collected chunk
   transitions; attach predicted next-feature differences to **executed
   action chunks** before computing Sinkhorn self-interaction OT. Actor still
   optimizes conservative Q and frozen-teacher anchor.
4. **Independent PushT Image validation:** train a separate image-conditioned
   Priming Stage-1 teacher from scratch (unless supplied), then repeat frozen
   baseline, residual, anchored Q and task-aware OT with the SAME underlying
   methods. Lowdim checkpoints are never loaded into Image networks.

The 8-action chunk represents the actual 8 individual PushT steps, and
the collector sums gamma^i reward[i] with bootstrap factor gamma^chunk_length.
A termination or 300-step timeout shortens the final chunk correctly.
No reward is taken from the evaluation runner's max-reward summary.
All checkpoint state, replay and metrics are stored per method.

PushT Image is a new **observation-modality benchmark** on the same PushT
physical task. This is cross-modality validation, not a second independent
robotics task, and claims must reflect that distinction.

## Hardware/data

- Existing robodiff conda environment with PushT dataset:
  \`data/pusht/pusht_cchi_v7_replay.zarr\`; Image requires the \`img\` key.
- One visible A100 80GB, with physical GPU ID 7 remapped to \`cuda:0\`.
- Evaluation: selection environment seeds 220000+, final held-out seeds
  320000+. Stage-1 checkpoint from older benchmark seeds (100000+).
- Image Stage-1 train/eval uses the original image dataset and runner,
  not the Lowdim observation labels.

## Smoke verification (run before full experiment)

\`\`\`bash
cd ~/diffusion_policy
conda activate robodiff
git pull origin main
python -m pytest -q tests/test_priming_control_research.py
CKPT="$(realpath data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt)"
python scripts/run_priming_control_research.py \\
  --stage1-checkpoint "$CKPT" --gpu 7 \\
  --out data/outputs/priming_pusht/control_research_smoke \\
  --smoke
\`\`\`

Smoke mode does NOT establish learning convergence. It checks the workflow:
2 image pretrain epochs, 80 online chunks per RL stage, 2 selection and
2 held-out episodes. It still requires a GPU, dataset, and both simulators.

## Full serial research: one command

\`\`\`bash
python scripts/run_priming_control_research.py \\
  --stage1-checkpoint "$CKPT" \\
  --gpu 7 \\
  --out data/outputs/priming_pusht/control_research_full \\
  --online-steps 10000 --eval-every 500 \\
  --eval-episodes 30 --heldout-episodes 50 \\
  --image-pretrain-epochs 400 \\
  --batch 128 --warmup 256 \\
  --anchor 5.0 --exploration 0.05 \\
  --ot-weight 0.1 --effect-scale 1.0 \\
  --resume
\`\`\`

On disconnect, use tmux:
\`tmux new -s priming_auto\`, run command, detach with Ctrl-b d.
Resume by running same command with --resume. Finished stages are skipped.
Incomplete online stages load their latest.pt checkpoint (latest evaluation
boundary), not the previous phase. Image Stage-1 does not support mid-epoch
resume: completed scored checkpoints can be reused, otherwise its stage
may need retraining.

If you have already separately trained an Image Stage-1 EMA checkpoint,
add \`--image-checkpoint /path/to/your/image/epoch=...ckpt\` (quote path
when shell spaces or equals interfere with Hydra). Image Stage-1 results
cannot be compared numerically to Lowdim without acknowledging modality.

For intermediate Lowdim-only verification: append \`--skip-image\`.

## Outputs / interpretation

\`\`\`text
control_research_full/
  pipeline_manifest.json
  lowdim/0_baseline/heldout.json
  lowdim/1_residual/{best.pt,latest.pt,training.jsonl,heldout.json,success.json}
  lowdim/2_anchored_q/...
  lowdim/3_task_aware_ot/...
  image_stage1_pretrain/{checkpoints/,logs.json.txt}
  image/0_baseline/...
  image/1_residual/...
  image/2_anchored_q/...
  image/3_task_aware_ot/...
\`\`\`

- Stage selection uses repeated fixed **selection** seeds; a method's
  best.pt and success.json are based on this validation split.
- The \`heldout.json\` files are for final reporting and MUST NOT select
  new checkpoints or tune algorithm parameters retrospectively.
- This prototype freezes pretrained image encoder features in online
  RL. The Image U-Net is trainable, but its ResNet is not.
- Online actor gradients use a clipped normalized action chunk and
  frozen-teacher action anchor. This is a deterministic TD3-style
  baseline with stochastic exploration, not full SAC or Policy Decorator.
- The task-aware representation is a learned **next encoder-feature**
  predictor, not a validated simulator/dynamics model. Ablate or
  validate predictive accuracy before making mechanistic claims.
- Critic bootstrapping, exploration and task-aware OT are fragile. Record
  critic TD errors, dynamics MSE, actor anchor and final PushT reward.
- No safety/optimality guarantee. A failed smoke test blocks the full
  run until fixed. Any new architectural method requires multiple
  independently trained random seeds to support claims.

## Relevant implementations

- \`diffusion_policy/model/priming/control_research.py\`:
  environment chunks, replay, actor/Q/dynamics.
- \`scripts/train_priming_control.py\`: isolated online stages.
- \`scripts/run_priming_control_research.py\`: fail-fast serial orchestration.
- \`diffusion_policy/policy/priming_unet_image_policy.py\` and
  \`diffusion_policy/config/train_priming_image_workspace.yaml\`:
  independent one-NFE Image Stage-1 pretraining.
