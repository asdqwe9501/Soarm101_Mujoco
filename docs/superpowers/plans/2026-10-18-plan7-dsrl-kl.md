# Plan 7: KL 정규화 DSRL (기반 분포에 묶어 둔 노이즈 조종)

**배경 (DSRL v3, `outputs/dsrl/v3/metrics.csv`):**
- 기반 분포에서 노이즈를 뽑은 warmup 50 에피소드는 성공 94%였다.
- SAC 갱신이 시작되자 성공률이 떨어져 500 에피소드에 16%까지 내려갔다.
  - alpha는 1.0에서 0.003으로 줄었다. 엔트로피 목표(−12)가 정책을 좁은 분포로 몰았다.
  - critic loss는 70~160이었다(보상 합 최대 7). actor가 잘못 맞춘 Q를 따라 기반 분포 밖으로 나갔다.
- 결론: 이미 강한 기반 정책(sft_aux2)에서는 RL이 "기반 분포에서 벗어나지 않는 것"부터 보장해야 한다.

**사용자 결정 (2026-10-18):** 3번(DSRL을 사전분포/KL 정규화로 재설계)으로 진행한다.

## 설계

- **actor**: z 공간의 가우시안 N(μ(s), σ(s))를 쓰고 tanh는 쓰지 않는다. 마지막 층을 0으로 초기화해 학습 시작 시 정확히 N(0, I), 곧 기반 정책이다. 실행할 때는 기존과 같이 ±LATENT_SCALE로 자른다.
- **목표 함수**: 엔트로피 보너스 대신 사전분포 KL 벌점을 쓴다. β는 고정이며 자동 조정하지 않는다.
  - critic 목표: `y = r + γ(1−d)(min Q'(s', a') − β·KL(π(·|s') ‖ N(0, I)))`
  - actor 손실: `β·KL(π(·|s) ‖ N(0, I)) − Q(s, a)`, a ~ π
  - 최적 정책은 `π ∝ N(0, I)·exp(Q/β)`다. Q 차이가 β보다 충분히 크지 않으면 기반 분포에 머문다.
- **critic 먼저 학습**: actor는 critic이 `--critic-warmup` 회 갱신된 뒤부터 갱신한다. `--init-buffer`가 있으면 루프 시작 전에 그 횟수만큼 critic만 오프라인으로 학습한다.
- **v3 경험 재사용**: v3 버퍼(같은 sft_aux2, train 명령문, 무작위화, 약 24k 전이)를 `--init-buffer`로 불러온다. Q는 현재 actor 기준으로 계산하므로 오프폴리시 데이터로 써도 된다.
- actor 학습률은 1e-4, critic 학습률은 3e-4다.
- **평가**: KL 정책은 학습 때와 같이 분포에서 샘플링한다. 기존 SAC 체크포인트는 그대로 결정적으로 평가한다.
- 보상(`rl/staged_reward.py`), 조종 공간(12차원 DCT), EXEC_STEPS = 50은 바꾸지 않는다.

## 작업

1. `rl/kl_ac.py` `KLRegularizedAC` + `tests/test_kl_ac.py`
   - 초기 정책이 N(0, I)이고 KL이 0이다.
   - critic warmup 동안 actor가 바뀌지 않는다.
   - 문맥 밴딧을 푼다.
   - 저장 후 불러오기가 된다.
2. `train/dsrl.py`: `--algo sac|kl`, `--beta`, `--critic-warmup`, `--init-buffer`. 기본값에서 기존 동작은 그대로다.
3. `policies/dsrl.py` `DSRLPolicyAdapter`: 체크포인트 종류(kind)를 보고 알맞은 agent를 불러온다.
4. 스모크 실행 후 본 학습(2000 에피소드, β = 1, detached)을 한다. 평가는 v3와 같은 체인으로 한다(반사실 basic/heldout + 전체 작업).

**성공 기준:** 학습 중 이동 평균 성공률이 warmup 수준(약 94%) 아래로 지속적으로 떨어지지 않는다. 반사실 시험 맞는 공 첫 시도가 sft_aux2(92.2%)보다 높다.
