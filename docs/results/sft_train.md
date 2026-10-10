# SmolVLA SFT 학습 기록 (sft_v1)

- 명령: `bash train/sft.sh sft_v1 30000 16`. 시작점은 lerobot/smolvla_base, `empty_cameras=1`, rename top→camera1, front→camera2이다.
- 데이터: local/so101_ballsort_sim. 에피소드 600개(색별 200개), 프레임 249,920개
- 최종 스텝: 30,000. 샘플 48만 개로 약 1.92 epoch다.
- 학습 시간: 약 6시간 35분(08:56 → 15:29). 스텝당 0.77초이고, 최대 GPU 메모리는 약 6.4GB다.
- 체크포인트: `outputs/train/sft_v1/checkpoints/{005000,...,030000,last}/pretrained_model`
  - Plan 3(DSRL)에서 쓸 체크포인트는 `outputs/train/sft_v1/checkpoints/last/pretrained_model`이다.

| step | loss |
|---|---|
| 1K | 0.039 |
| 2K | 0.025 |
| 5K | 0.020 |
| 10K | 0.012 |
| 15K | 0.010 |
| 20K | 0.008 |
| 25K | 0.007 |
| 30K | 0.007 |
