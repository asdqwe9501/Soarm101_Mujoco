# SFT 스모크 테스트 (12-에피소드 파일럿, 50 스텝)

환경: RTX 5070 Laptop 8 GB, lerobot 0.4.4, 기준 GPU 사용량 약 136 MiB.

| batch | 최대 GPU 메모리 | updt_s/step | data_s | 비고 |
|---|---|---|---|---|
| 8  | 약 4.1 GB (10초 간격 샘플, 하한값) | 0.42 s | 0.007 | num_workers=4 정상 |
| 16 | 6.5 GB (1초 간격 샘플) | 0.80 s | 0.012 | OOM 없음 |

**선택: batch_size 16.** 사용량이 목표(7 GB 이하) 안이다. 샘플당 시간은 bs8과 같아(약 0.05 s/샘플) 스텝당 샘플 수만 두 배다.
30k 스텝 예상: 0.8 s x 30000 = 약 6.7 시간 (bs8이면 약 3.5 시간, 샘플 수는 절반).

## 필요했던 추가 인자 / 조치
- `--policy.empty_cameras=1`: camera3가 조용히 버려지지 않도록 한다. 체크포인트 config.json에 `empty_cameras=1`이 저장되고 input_features에 camera1/2/3 + empty_camera_0이 있다. 평가 어댑터가 이 체크포인트를 오류 없이 로드한다.
- `--policy.path`: Windows에서 lerobot가 `lerobot/smolvla_base`를 Path로 바꿔 `lerobot\smolvla_base`가 되어 HFValidationError가 난다. 스크립트에서 `snapshot_download`로 얻은 로컬 경로를 넘긴다.
- `--rename_map`: Git Bash에서 JSON이 dict(두 키)로 정상 파싱됨을 로그에서 확인했다. 체크포인트의 policy_preprocessor.json에 rename_observations_processor와 우리 데이터셋 통계의 normalizer가 있다.
- `num_workers=4`는 Windows에서 동작했다.
- `DATASET` 환경변수로 데이터셋 이름을 바꿀 수 있다(스모크는 `so101_ballsort_sim_pilot`).

## 평가기 확인
`python -m eval.evaluate --policy smolvla --checkpoint outputs/train/smoke_bs16/checkpoints/last/pretrained_model --n-per-color 1` 이 끝까지 실행되어 JSON을 출력했다(성공률 0.0, 50 스텝 학습이므로 예상대로).
