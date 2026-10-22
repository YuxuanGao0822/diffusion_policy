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
OUT=data/outputs/priming_pusht/noise_reference_drift
mkdir -p "$OUT"
python -m pytest -q tests/test_priming_sample_drift.py
python scripts/diagnose_priming_reference_drift.py \
  --checkpoint "$CKPT" --device cuda:0 \
  --train-seeds 20261021 20261022 20261023 \
  --samples-per-condition 8 16 32 \
  --train-conditions 2 \
  --eval-conditions "${PRIMING_EVAL_CONDITIONS:-8}" \
  --eval-samples "${PRIMING_EVAL_SAMPLES:-16}" \
  --probes "${PRIMING_PROBES:-8}" \
  --step-norm 0.001 --window full executed \
  --output "$OUT/k_sweep.csv"
python - <<'PY'
import pandas as pd
from pathlib import Path
p=Path("data/outputs/priming_pusht/noise_reference_drift/k_sweep.csv")
df=pd.read_csv(p)
columns=["train_seed","K","window","delta_Sz","delta_variance",
         "val_mean_shift_rms","delta_train_pair_fresh","delta_train_pair_stale"]
print(df[columns].to_string(index=False,float_format=lambda v:f"{v:+.6e}"))
print("\nK-sweep means (Full only):")
print(df[df.window=="full"].groupby("K")[columns[3:]].mean().to_string(
     float_format=lambda v:f"{v:+.6e}"))
PY
