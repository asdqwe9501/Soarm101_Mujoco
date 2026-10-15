#!/usr/bin/env bash
# 사용: bash train/sft_aux.sh <job_name> <steps> <batch_size> [추가 lerobot-train 인자...]
# train/sft.sh와 같고, lerobot-train 대신 train/sft_aux.py(보조 목표 위치 손실)를 부른다. Plan 6 Task 2.
# 환경변수: INIT(시작 체크포인트 pretrained_model 경로), DATASET(data/ 아래 이름),
#           DATA_ROOT(데이터셋 경로 직접 지정, 기본 data/$DATASET), AUX_LAMBDA(기본 0.1, 학습된 체크포인트에서 시작하면 0.01 권장), AUX_LOG_FREQ(기본 100),
#           AUX_MAX_FRAME(보조 손실에 쓸 초기 프레임 frame_index 상한, 기본 10), AUX_BS(초기 프레임 배치 크기, 기본 16)
set -euo pipefail
JOB=$1; STEPS=$2; BS=$3; shift 3
DATASET=${DATASET:-so101_ballsort_sim}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-$ROOT/data/$DATASET}
if [ -n "${INIT:-}" ]; then
  BASE="$INIT"
else
  BASE=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('lerobot/smolvla_base'))" | tail -1)
fi
cd "$ROOT"
python -m train.sft_aux \
  --policy.path="$BASE" \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.empty_cameras=1 \
  --dataset.repo_id=local/$DATASET \
  --dataset.root="$DATA_ROOT" \
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
