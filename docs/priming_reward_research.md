# Reward-guided Priming sequential research

**Experimental implementation.** CPU tests and GPU smoke tests must run in the actual robodiff/PushT environment before a multi-day experiment. This is neither a reproduced FQL nor a reproduced Policy Decorator implementation. Performance improvements are hypotheses, not guarantees.

## Stages and what is implemented

0. Held-out baseline evaluation of the frozen E350 one-step PushT lowdim policy on new seeds.
1. Frozen one-step policy + bounded tanh-Gaussian residual actor, twin-Q online SAC, replay with semi-Markov 8-step chunk reward.
2. One-step U-Net actor trained directly with Q guidance, E350 teacher anchor, and residual actor distillation; preserve previous critic and online replay.
3. Dynamics model predicts projected next-state features and chunk return; validate on heldout replay transitions, then optimize action-and-task-effect OT jointly with Q and teacher anchor.
4. Independently pretrain a PushT Image one-step policy with the repository's RGB ResNet encoder; rerun the same 0–3 online phases and evaluate on a fresh set of image PushT seeds.

Stage 4 validates **observation modality transfer**, not generalization to new physical dynamics: PushT lowdim and PushT Image use the same underlying manipulation task. No lowdim network weights are loaded into the image policy.

## One-command automated serial research

From the repository root and your robodiff conda environment:

~~~bash
cd ~/diffusion_policy
conda activate robodiff
git pull origin main
bash scripts/run_priming_research.sh
~~~

By default the shell script selects physical GPU 7 (in-process cuda:0), turns off W&B and rendering, verifies the E350 checkpoint, executes CPU tests, and runs all phases serially. It uses 120 online episodes per phase, 25 evaluation seeds for checkpoint selection, 50 different seeds for final comparison, and 350 epochs for image Stage-1 pretraining.

To use another output directory or training seed:

~~~bash
bash scripts/run_priming_research.sh --output data/outputs/priming_research_seed43 --seed 43
~~~

To run a short end-to-end smoke experiment before the full job:

~~~bash
CKPT="$(realpath data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt)"
python scripts/run_priming_research.py --checkpoint "$CKPT" --smoke --output data/outputs/priming_research_smoke
~~~

Smoke reduces the episode count, update count and image epochs (via the existing image workspace debug setting). It disables the dynamics **accuracy** gate solely for smoke, and its scores are not scientific evidence.

To view the plan and prerequisites without running:

~~~bash
python scripts/run_priming_research.py --checkpoint "$CKPT" --dry-run
~~~

To supply an already trained one-step Priming Image checkpoint instead of running image pretraining:

~~~bash
bash scripts/run_priming_research.sh --image-checkpoint /ABS/PATH/IMAGE/checkpoints/epoch=XXXX-test_mean_score=YYY.ckpt
~~~

The supplied Image checkpoint must have a policy target containing priming_unet_image_policy. A lowdim checkpoint or a DDPM image checkpoint is incompatible.

## Data and algorithm details

The implementation reads the native data/pusht/pusht_cchi_v7_replay.zarr and the native PushT simulator. Stage 4 trains a separate image policy with diffusion_policy/config/train_priming_image_workspace.yaml.

Training data comes from **online environment collection**, not the original expert Zarr dataset, except during image Stage-1 pretraining. Each replay transition stores the frozen encoder/observation condition, executed normalized 8-step action chunk, the discounted reward across *actually executed* primitive steps, a successor condition, the semi-Markov discount gamma^actual_steps and a terminal flag. The code takes raw reward increments from MultiStepWrapper.get_rewards() and never mistakes its time-accumulated aggregate for a chunk reward.

In phase 1, residual actions are clipped to valid normalized bounds. Q target uses a tanh-Gaussian residual stochastic policy and an entropy coefficient. In phases 2–3, the implicit one-step generator has no analytic log density: critic backups omit the entropy term and the model is updated using differentiable critic Q, strong E350 anchoring and, in phase 3, task-aware OT. Teacher and image encoder are frozen during online refinement. This is a controlled algorithm prototype, not an established off-policy RL reference implementation.

The task-outcome OT target concatenates observed executed actions with observed projected next state and chunk return; generated candidates use predicted task effects from a differentiable dynamics model. Validation must outperform a constant predictor (default validation MSE ratio <= 0.98) or the pipeline stops. This prevents automatically treating a poor model as a useful task metric; passing the gate alone does not prove learned dynamics are accurate.

An episode's evaluation score is the **maximum PushT step reward**, matching the existing PushT runner convention. Checkpoint-selection seeds are 200000... for lowdim / 300000... for image; entirely withheld final seeds are 500000... / 600000.... Final results for all four policies per modality are written to:

- data/outputs/priming_research/lowdim/final_comparison.csv
- data/outputs/priming_research/image/final_comparison.csv

Per-phase subdirectories contain best.pt, latest.pt, train.jsonl, an immutable run_config.json and individual evaluation CSVs. The next phase reuses **best policy weights**, latest previous critic and complete replay. A partial online RL phase resumes automatically from latest.pt; incompatible parameter changes require a new --output directory. Interrupted Image Stage-1 pretraining is detected and deliberately refuses silent reuse of a partial checkpoint: resume that workspace deliberately or start with a new --output, or explicitly supply --image-checkpoint.

## Limitations and rigorous evaluation

This implementation has not been benchmarked on the user's A100 and must pass the supplied CPU suite plus simulator smoke before full training. Image Stage-1 pretraining from scratch may be expensive; ensure available storage and a long-lived tmux session.

Image and lowdim share PushT physics, so a different manipulation task is still needed for general task-generalization claims. A final 50-seed environment evaluation does not quantify independent training-seed variance: repeat major methods with at least three training seeds. Reward-based RL is for simulation research, not unprotected real-robot execution.

Original Priming's pure action OT remains untouched; the task-aware alternative uses an explicitly different target distribution and must be presented as a new research hypothesis. Passing the dynamics gate or reducing actor loss does not guarantee closed-loop reward improvements.
