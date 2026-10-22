#!/usr/bin/env bash
set -euo pipefail
cd "${PRIMING_REPO:-$HOME/diffusion_policy}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PRIMING_GPU:-7}"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export WANDB_MODE=disabled HYDRA_FULL_ERROR=1 OMP_NUM_THREADS=8
OUT=data/outputs/priming_pusht/noise_mechanism
mkdir -p "$OUT"
python -m py_compile scripts/diagnose_priming_noise_mechanism.py
STAGE1="$(realpath data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt)"
CKPTS=("$STAGE1")
NAMES=(stage1_e350)
for entry in "stage2_full_ot:data/outputs/priming_pusht/stage2_from_e350_b128_k32/checkpoints/latest.ckpt" "stage2_executed_ot:data/outputs/priming_pusht/stage2_e350_executed_ot/checkpoints/latest.ckpt"; do
  name="${entry%%:*}"
  path="${entry#*:}"
  if [[ -f "$path" ]]; then
    CKPTS+=("$(realpath "$path")")
    NAMES+=("$name")
  else
    echo "Optional checkpoint unavailable, skipping: $path"
  fi
done
echo "=== A: input-noise Jacobian ==="
python scripts/diagnose_priming_noise_mechanism.py jacobian \
  --checkpoint "${CKPTS[@]}" --names "${NAMES[@]}" \
  --split val --window full executed \
  --conditions 8 --samples 8 --probes 4 --device cuda:0 \
  --output "$OUT/jacobian.csv"
echo "=== B1: one SGD step, common learning rate ==="
python scripts/diagnose_priming_noise_mechanism.py intervene \
  --checkpoint "$STAGE1" --ot-window full --window full executed \
  --train-conditions 2 --train-samples 8 \
  --conditions 8 --samples 8 --probes 4 \
  --epsilon 0.05 --iterations 50 --lr 1e-5 --device cuda:0 \
  --output "$OUT/single_step_fixed_lr.csv"
echo "=== B2: one SGD step, matched update norm ==="
python scripts/diagnose_priming_noise_mechanism.py intervene \
  --checkpoint "$STAGE1" --ot-window full --window full executed \
  --train-conditions 2 --train-samples 8 \
  --conditions 8 --samples 8 --probes 4 \
  --epsilon 0.05 --iterations 50 --matched-step-norm 0.01 --device cuda:0 \
  --output "$OUT/single_step_matched_norm.csv"
python - <<'PY'
import pandas as pd
from pathlib import Path
root = Path("data/outputs/priming_pusht/noise_mechanism")
j = pd.read_csv(root/"jacobian.csv")
print("\n[A] Jacobian")
print(j.groupby(["checkpoint","window"])[["jacobian_sq_per_noise_dim","noise_std","pure_noise_expert_mse"]].mean().round(8).to_string())
for name in ("single_step_fixed_lr.csv","single_step_matched_norm.csv"):
    d = pd.read_csv(root/name)
    print("\n[B] "+name)
    columns = ["arm","window","gradient_norm","parameter_update_l2","delta_jacobian_sq_per_noise_dim","delta_noise_std","delta_pure_noise_expert_mse"]
    print(d[columns].round(9).to_string(index=False))
PY
