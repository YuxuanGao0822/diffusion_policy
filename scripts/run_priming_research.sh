#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HYDRA_FULL_ERROR=1
CKPT="${PRIMING_STAGE1_CKPT:-data/outputs/priming_pusht/stage1_local/checkpoints/epoch=0350-test_mean_score=0.791.ckpt}"
if [[ ! -s "$CKPT" ]]; then echo "Missing Stage-1 checkpoint: $CKPT" >&2; exit 1; fi
CKPT="$(realpath "$CKPT")"
python -m pytest -q tests/test_priming_research.py
python scripts/run_priming_research.py --checkpoint "$CKPT" --phase all --device cuda:0 "$@"
