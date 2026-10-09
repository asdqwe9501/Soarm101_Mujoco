#!/usr/bin/env bash
# 사용: bash train/sft.sh <job_name> <steps> <batch_size> [추가 lerobot-train 인자...]
# DATASET 환경변수로 데이터셋 이름 변경 가능 (기본 so101_ballsort_sim)
set -euo pipefail
JOB=$1; STEPS=$2; BS=$3; shift 3
DATASET=${DATASET:-so101_ballsort_sim}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Windows: lerobot가 policy.path를 Path로 바꿔 "lerobot\smolvla_base"가 되어 HF repo id 검증에 실패한다. 로컬 스냅샷 경로를 넘긴다.
BASE=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))" | tail -1)
lerobot-train \
  --policy.path="$BASE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --dataset.repo_id=local/$DATASET \
  --dataset.root="$ROOT/data/$DATASET" \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.front": "observation.images.camera2"}' \
  --batch_size="$BS" \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=100 \
  --num_workers=4 \
  --output_dir="$ROOT/outputs/train/$JOB" \
  --job_name="$JOB" \
  --wandb.enable=false \
  "$@"
