# Diagnosing noise blindness in Stage-1 / Stage-2 Priming

This audit is read-only with respect to Stage-1/Stage-2 checkpoint files.
It diagnoses *local noise-input sensitivity*, not policy success or the
paper's feature-to-parameter Jacobian. Use only trusted dill checkpoints.

## A. Noise-input Jacobian

Fixed observation, resampled noise z: measure mean sample standard deviation,
MSE to expert action, and the Hutchinson estimate

    S_z = E_z[ ||dG(z,o)/dz||_F^2 / (H*Da) ].

For the executed window only the output time steps [1:9] are selected;
the denominator remains all **32 input noise dimensions** (16*2).
Jacobians use deployed normalized-action clipping, if enabled. Eight samples
and four Rademacher probes are defaults for a low-cost diagnostic; increase
both for stronger estimates.

## B. One-step intervention

Load the exact same Stage-1 E350 EMA snapshot five times. Use identical
train observation indices, minibatch, live/reference/reconstruction Gaussian
seeds and untouched validation examples:

- control (zero update);
- attraction only;
- negative repulsion only;
- original attraction minus repulsion (quadratic OT);
- Stage-1 noisy reconstruction MSE.

The OT branch uses detached scale, detached Sinkhorn plan and the same
stop-gradient reference as the production code. Train gradients use unclipped
actions; Jacobian evaluation uses policy-deployed clipping. Updates use
one **momentum-free SGD** step on the U-Net weights, not Adam or EMA.
The two modes are common LR and common global parameter-update L2.
This isolates instantaneous gradient effects; it does not establish
long-run causal outcomes or closed-loop PushT reward.

## Reproduce

From the repository root in your robodiff environment:

    conda activate robodiff
    git pull origin main
    bash scripts/run_priming_noise_mechanism.sh 2>&1 | tee priming_noise_mechanism.log

Set PRIMING_GPU=7 if necessary; default GPU 7 is remapped to cuda:0.
Checkpoint E350 is required. Stage-2 latest checkpoints are optional and
skipped if missing. Runs should complete without training or simulator use.

Output under data/outputs/priming_pusht/noise_mechanism/:

    jacobian.csv
    single_step_fixed_lr.csv
    single_step_matched_norm.csv

Interpretation:
- Repulsion-only fails to increase S_z -> weak noise-opening action locally.
- Repulsion-only increases S_z but OT does not -> attraction offsets the effect.
- Both increase S_z but completed Stage-2 models lose sensitivity -> sustained
  optimization/schedule, not just the single update, is worth examining.

S_z estimates can be noisy and strongly dependent on the chosen samples,
parameter step and observation split. Repeat with several training seeds,
evaluation seeds and step sizes before drawing a mechanistic conclusion.
One checkpoint's action std cannot establish whether expert actions at
identical observations are multimodal.

## Test

    python -m pytest -q tests/test_priming_noise_mechanism.py

This CPU test validates the Hutchinson normalization on an identity map,
a zero-noise-response map, and matched-parameter-step norm handling.
