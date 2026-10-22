#!/usr/bin/env bash
set -euo pipefail
cd "${PRIMING_REPO:-$HOME/diffusion_policy}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PRIMING_GPU:-7}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
CKPT="$(realpath data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt)"
OUT=data/outputs/priming_pusht/noise_gradient_direction
mkdir -p "$OUT"
python -m pytest -q tests/test_priming_gradient_direction.py
python scripts/diagnose_priming_gradient_direction.py \
  --checkpoint "$CKPT" --device cuda:0 \
  --train-seeds 20261021 20261022 20261023 \
  --eval-seed 314159 \
  --train-conditions 2 --train-samples 8 \
  --eval-conditions "${PRIMING_EVAL_CONDITIONS:-16}" \
  --eval-samples "${PRIMING_EVAL_SAMPLES:-12}" \
  --probes "${PRIMING_PROBES:-8}" \
  --windows full executed \
  --arms attraction negative_repulsion ot mse \
  --step-norms 0.0005 0.001 0.003 \
  --ot-window full --epsilon 0.05 --iterations 50 \
  --output "$OUT/central_fd.csv"
python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path("data/outputs/priming_pusht/noise_gradient_direction/central_fd.csv")
df=pd.read_csv(p)
cols=["train_seed","arm","window","update_norm",
      "directional_derivative_jacobian_sq_per_noise_dim",
      "delta_predicted_jacobian_sq_per_noise_dim",
      "delta_observed_jacobian_sq_per_noise_dim",
      "nonlinear_remainder_jacobian_sq_per_noise_dim"]
print(df[cols].to_string(index=False,float_format=lambda v:f"{v:+.6e}"))
print("Positive D_u S_z => descending the loss locally decreases S_z.")
PY
