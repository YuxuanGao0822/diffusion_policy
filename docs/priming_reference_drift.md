# Finite-K OT repulsion: frozen-reference drift versus true noise diversity

**Question:** Does maximizing the detached-reference repulsion move the
mean generated action away from the *frozen* Stage-1 reference, without
producing a wider distribution after both live/reference are regenerated
from the updated model?

This is a follow-up to the fixed-evaluation central finite-difference
gradient audit, and deliberately uses the same frozen Stage-1 E350
initialization, 3 training minibatch/noise seeds, and a matched global
parameter update norm of 0.001. The only training-side variable is
K=8,16,32 (production Stage-2 used K=32).

For each trained update compare, on exactly fixed held-out conditions/noise:

- delta_Sz: change in input-noise Jacobian sensitivity
- delta_variance: change in conditional action sample variance
- val_mean_shift_rms: movement of the *sample mean* action
- delta_pair_fresh: raw uniform-pair distance when both live and
  independent references come from the updated network
- delta_pair_stale: same raw uniform-pair distance with the *reference*
  frozen at the E350 network
- delta_train_pair_fresh / delta_train_pair_stale: analogous raw distances
  on the exact minibatch/noise used for the loss gradient

All reference comparisons are made in normalized action units. Validation
respects deployed clipping; training-side pair distances are unclipped.
Pair distances here are intentionally uniform-pair means and **not**
the production Sinkhorn-plan-weighted objective. Use them to diagnose
translation versus spread, not to claim a measured OT objective change.

## Execute

    cd ~/diffusion_policy
    conda activate robodiff
    git pull origin main
    bash scripts/run_priming_reference_drift.sh 2>&1 | tee reference_drift.log

Output: data/outputs/priming_pusht/noise_reference_drift/k_sweep.csv

## Interpret

If the frozen-reference distance increases while updated/live distance
and variance do not, the self-repulsion update may principally act as
a sample-dependent mean displacement. Increasing K should reduce
finite-sample reference-mean noise if that mechanism matters.

If K=32 reverses the sign of delta_Sz (compared to K=8), then results
from prior K=8 local diagnostics should not be generalized directly
to the actual K=32 post-training setup.

If K=32 remains negative but model mean shift dominates variance
changes, test a zero-mean, centered repulsion control next. A different
OT coupling or moving-reference method could also be relevant.

This is a local, fixed-checkpoint experiment, not closed-loop control.
Three resampling seeds do not imply generalization across trained
checkpoints or expert modes.
