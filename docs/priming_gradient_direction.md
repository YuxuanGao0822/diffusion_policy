# Priming: directional Jacobian experiment

This is the next diagnostic following the 3-seed step-size sweep. It runs
at the trusted Stage-1 E350 EMA checkpoint and never overwrites it.

For each objective L in attraction, negative repulsion, OT, MSE,
let u = grad_theta L / ||grad_theta L||_2. For a fixed held-out
observation/noise/probe set, compute:

    D_u S_z = [S_z(theta+h*u) - S_z(theta-h*u)] / (2*h)
    predicted descent delta S_z = -h * D_u S_z
    observed descent delta S_z = S_z(theta-h*u) - S_z(theta)

S_z is the normalized Frobenius norm squared of the input noise Jacobian
(the same estimator as diagnose_priming_noise_mechanism.py).
The division uses all 32 input noise dimensions in both full and
executed windows. Positive D_u S_z means descending L tends to
DECREASE noise sensitivity. This is not the full gradient cosine rho_J.

## Run

    cd ~/diffusion_policy
    conda activate robodiff
    git pull origin main
    bash scripts/run_priming_gradient_direction.sh 2>&1 | tee gradient_direction.log

Output is data/outputs/priming_pusht/noise_gradient_direction/central_fd.csv.
Source trusted E350 checkpoint must exist in the standard Stage-1 folder.

Quick plumbing check before the longer run:

    CUDA_VISIBLE_DEVICES=7 python scripts/diagnose_priming_gradient_direction.py \
      --checkpoint data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt \
      --device cuda:0 --train-seeds 20261021 \
      --arms negative_repulsion --eval-conditions 2 --eval-samples 4 \
      --probes 2 --step-norms 0.001 \
      --output data/outputs/priming_pusht/noise_gradient_direction/smoke.csv

The full study uses 3 training minibatch/noise seeds, a single
held-out evaluation set, 12 evaluation Gaussian samples per observation,
8 Hutchinson probes, and update magnitudes 0.0005/0.001/0.003.
Raise PRIMING_EVAL_CONDITIONS, PRIMING_EVAL_SAMPLES, and PRIMING_PROBES
if observed deltas are too close to numerical resolution.

## Interpretation

Compare directional_derivative_jacobian_sq_per_noise_dim and
delta_predicted_jacobian_sq_per_noise_dim with
delta_observed_jacobian_sq_per_noise_dim, across seeds and h.
Direction consistency across several h supports a local first-order
explanation. Large nonlinear remainders indicate a breakdown in the
linear approximation, Monte Carlo estimation noise, or floating-point
resolution issues. Disable TF32 as the script does by default.

The three training seeds resample minibatches and noise; they are NOT
three independently trained Stage-1 models.

Next optional work after this result: compare the original detached
batch-scale OT with fixed-scale OT (matched effective Sinkhorn
temperature), then study EMA scale in multi-step controlled runs.
Do NOT interpret input-noise variance alone as useful multimodal behavior,
or claim an improvement in PushT rollout reward from this audit.
