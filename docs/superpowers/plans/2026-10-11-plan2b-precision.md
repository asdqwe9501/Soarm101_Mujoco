# Plan 2b: 집기 정밀도 개선

**배경:**
- sft_v2(80k 스텝)는 명령한 공으로는 가지만, 집는 위치 오차가 2.5~5cm다.
- 시뮬레이션에서 잡기가 성공하려면 오차가 ±0.5cm 이내여야 한다. 시연 스크립트로 측정하면 1cm에서 3/12, 2cm에서 0/12만 성공한다.
- 진단 과정은 `docs/results/sft_eval.md`에 있다.

**Goal:** 두 가지로 정책 오차와 허용 오차 사이의 간격을 줄인다. 그다음 SFT를 이어서 학습하고, 다시 평가한다(통과 기준 80%).
1. 시뮬레이션 잡기 허용 오차를 실제 스펀지 공에 가깝게 넓힌다.
2. 시연 데이터에 "빗나간 위치에서 바로잡는" 장면을 넣는다.

## Global Constraints

- 데이터셋 형식, 키, 단위, 명령문, 성공 판정은 Plan 1과 같다.
- 시뮬레이션 제자리 자세는 `calib.SIM_HOME_NORM`이다.
- 기존 테스트는 모두 통과해야 한다. 시연 스크립트의 기본 동작 성공률은 150/150을 유지해야 한다(`python -m eval.evaluate --policy expert --n-per-color 50`).
- 정답 행동(라벨)은 언제나 "현재 상태에서 올바른 목표로 가는 행동"이어야 한다. 일부러 빗나가는 실행 경로를 라벨로 쓰면 안 된다.

## Task A: 잡기 허용 오차 완화 (물리 설정)

**Files:** `sim/scene.xml`(공 접촉 파라미터), `tools/grasp_tolerance.py`(신규, 측정 도구), `docs/results/grasp_tolerance.md`

**측정 도구:** `outputs/debug/grasp_tolerance.py`를 정식 도구로 옮긴다. 시연 스크립트가 계획할 때만 공 위치를 반경 r만큼 12방향으로 어긋나게 해서 성공 수를 센다.

**목표:**
- 0.5cm: 12/12
- 1.5cm: 10/12 이상
- 2.0cm: 가능한 한 높게

**조정 범위:**
- 공의 `solref`/`solimp`(부드럽게), `friction`, `condim`, `mass`
- 필요하면 그리퍼 쪽 접촉 우선순위(`priority`)

**조건:**
- 공이 책상 위에서 가만히 있어야 한다(`test_balls_stay_still_at_home` 통과).
- 통 안에 들어간 공이 튀어나오지 않아야 한다.
- `GRIP_OPEN`과 `GRIP_CLOSED`는 바꾸지 않는다. 바꾸면 기존 600개 시연의 라벨과 어긋난다.

**완료 기준:**
- 목표 수치를 달성하거나, 물리 설정만으로 도달할 수 있는 최선의 값과 그 한계를 근거와 함께 기록한다.
- 전체 테스트와 시연 스크립트 150/150을 유지한다.

## Task B: 보정 시연 모드 + 600개 추가 수집

**Files:** `expert/scripted.py`(보정 모드 옵션), `data/collect.py`(옵션 전달), `tests/test_expert.py`(보정 모드 테스트)

**보정 모드 (`ScriptedExpert(..., correction=True)`):**
1. 공 위로 갈 때 실행 경로는 실제 공에서 2~4cm(무작위 방향) 빗나간 지점으로 간다.
2. 그 구간의 라벨은 "현재 상태에서 올바른 공 위 자세로 향하는 행동"이다. 현재 관절값에서 올바른 자세 쪽으로 최대 속도 제한을 둔 한 스텝이다.
3. 빗나간 지점에 도착하면 0.3~0.6초 멈춘다. 이 동안 실행과 라벨 모두 올바른 자세를 향한 보정 행동이다.
4. 이어서 현재 위치에서 올바른 공 위 자세로 보정 이동한다(실행과 라벨이 같다). 그 뒤는 기존 단계(하강, 닫기 등)와 같다.

**테스트:**
- 보정 모드에서 라벨이 항상 올바른 목표 쪽을 향하는지 확인한다(라벨과 올바른 자세 사이 거리가 실행 자세보다 짧거나 같다).
- 보정 모드 성공률이 20개 중 19개 이상이다.

**수집:**
- `python -m data.collect --correction --repo-id local/so101_ballsort_sim_corr --root data/so101_ballsort_sim_corr --n-per-color 200 --seed 100000`
- seed는 기존 데이터와 겹치지 않게 한다.

## Task C: 데이터 병합 + 이어서 학습 + 평가

1. **병합:** 기존 600개와 보정 600개를 하나의 데이터셋 `local/so101_ballsort_sim_all`로 합친다. lerobot 0.4.4의 데이터셋 병합 기능(`lerobot-edit-dataset`, `src/lerobot/scripts/lerobot_edit_dataset.py`)을 먼저 확인한다.
2. **학습:** sft_v2/last에서 시작해 40k 스텝을 새 학습률 일정으로 학습한다.
   - `INIT=... DATASET=so101_ballsort_sim_all bash train/sft.sh sft_v3 40000 16 --policy.scheduler_decay_steps=40000`
3. **평가:** 정밀도 지표(`outputs/debug/reach_check.py`)와 소규모 평가(색별 5개)를 먼저 본다. 가능성이 보이면 정식 평가(색별 50개 + 연속 명령 20회)를 한다. 통과 기준은 80%다.
