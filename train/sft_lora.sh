#!/usr/bin/env bash
# LoRA 미세조정: VLM 텍스트 모델 self-attn(q,k,v,o, 16층) + SmolVLA 기본 대상(expert q/v, 상태/행동 투영).
# 비전 인코더와 모든 base 가중치는 동결. r=16, lora_alpha=8(PEFT 기본값; lerobot CLI로 노출 안 됨).
# 사용: bash train/sft_lora.sh <job_name> <steps> <batch_size> [추가 lerobot-train 인자...]
# 기본 INIT=sft_v3 040000(절대 경로: adapter_config의 base_model_name_or_path로 저장됨), DATASET=so101_ballsort_sim_v5
# 실제 실행: bash train/sft_lora.sh sft_lora1 15000 16 --policy.scheduler_decay_steps=15000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && (pwd -W 2>/dev/null || pwd))"  # Windows 경로(C:/...)로 adapter_config에 저장되게
export INIT=${INIT:-$ROOT/outputs/train/sft_v3/checkpoints/040000/pretrained_model}
export DATASET=${DATASET:-so101_ballsort_sim_v5}
TARGETS='(model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj|model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'
exec bash "$ROOT/train/sft.sh" "$@" \
  --peft.method_type=LORA \
  --peft.r=16 \
  "--peft.target_modules=$TARGETS" \
  '--peft.full_training_modules=[]'
