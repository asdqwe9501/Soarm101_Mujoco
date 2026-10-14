# Plan 6: 보조 과제(목표 공 위치 예측)로 SmolVLA 색 구분 학습

**배경:**
- 학습만으로 색 구분을 높이려는 시도 다섯 가지 모두에서 오선택률이 약 23%에 머물렀다(`docs/results/color_grounding_attempts.md`).
- 사용자 결정(2026-10-14):
  - SmolVLA 자체가 색 구분을 하도록 학습 규모를 키운다(A).
  - 학습은 30k 스텝으로 한다. 스텝을 늘려도 좋아지지 않았다는 사용자 경험을 따른다.

**가설:**
- 행동 모방 손실은 대부분 쉬운 동작 구간에서 나온다. 그래서 "어느 공으로 갈지"에 대한 학습 신호가 약하다.
- 화면과 명령만 보고 목표 공 위치를 맞히는 보조 과제를 더하면, 색과 위치를 연결하는 표현을 직접 학습하게 된다.

## Global Constraints

- 보조 정답(목표 공 위치)은 **학습 손실에만** 쓴다. 정책 입력이나 추론에는 쓰지 않는다. 실제 로봇에서도 같은 정책이 그대로 동작해야 한다.
- 학습: sft_v3에서 시작해 30k 스텝, 배치 16(메모리가 허용하는 한)으로 한다.
- 데이터에서 네거티브 시연(wrong_approach)을 뺀다. 망설임 문제가 있었기 때문이다.

## Task 1: 데이터 (1,500개)

- `data/collect.py`에 `--aux-labels` 옵션을 추가한다. 켜면 프레임마다 `target_xy`(float32, (2,))를 기록한다.
  - 값은 명령한 색 공의 현재 (x, y) 위치(m, 로봇 좌표)다.
  - 키 이름이 `observation.`으로 시작하지 않으므로 lerobot이 정책 입력으로 쓰지 않는다.
- 수집 명령:

  `python -m data.collect --same-layout --randomize --mode-mix --modes default,correction --scene-mix --aux-labels --repo-id local/so101_ballsort_sim_v8 --root data/so101_ballsort_sim_v8 --n-per-color 500 --seed 500000`

## Task 2: 보조 손실 학습 (`train/sft_aux.py`)

- **보조 출력 층**: lerobot-train을 감싸는 스크립트로 SmolVLA 정책에 작은 MLP를 단다.
  - 입력: expert가 행동 토큰에 대해 낸 은닉 상태(`action_out_proj`의 입력)의 평균
  - 출력: 목표 공 위치 (x, y). 작업 공간 범위로 정규화한다.
- **행동 정보 누출 방지**:
  - 흐름 매칭 시간 t가 작으면 노이즈가 섞인 입력 행동에 궤적 정보, 곧 목표 위치가 그대로 들어 있다.
  - 그래서 보조 손실은 **t = 1(순수 노이즈)로 따로 한 번 더 forward**해서 계산한다.
  - 이렇게 하면 화면, 명령, 관절 상태만으로 목표 위치를 맞혀야 한다.
- **전체 손실**: `loss = flow_loss + λ · MSE(xy_pred, target_xy)`. λ는 스모크 테스트에서 두 손실의 크기를 보고 정한다(시작값 0.05).
- **저장과 로드**: 보조 출력 층은 정책과 함께 저장한다. 추론에는 쓰지 않으므로, 평가 어댑터가 이 층이 없어도(무시해도) 읽을 수 있어야 한다.
- **기록**: 보조 손실과 보조 예측 오차(cm)를 로그에 남긴다.

## Task 3: 학습과 평가

- 학습: `train/sft_aux.sh sft_aux1 30000 16` (INIT=sft_v3, DATASET=so101_ballsort_sim_v8)
- 평가:
  - 반사실 시험(배치 30개): 5k / 15k / 30k 체크포인트
  - 보조 예측 오차: 학습에 쓰지 않은 seed에서 측정
