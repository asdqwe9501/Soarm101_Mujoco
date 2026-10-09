# Plan 2: SmolVLA SFT 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 시뮬레이션 시연 600개로 `lerobot/smolvla_base`를 파인튜닝(SFT)한다. 그 결과를 시뮬레이션에서 평가해 단일 명령 성공률 80% 이상(통과 기준)을 확인한다.

**Architecture:**
- **학습**: 설치된 lerobot 0.4.4의 `lerobot-train`을 그대로 쓴다. 실행 인자는 `train/sft.sh` 하나에 모은다.
- **평가**: 체크포인트를 Plan 1의 정책 프로토콜(`reset(env)`, `act(obs)`)로 감싸는 어댑터 `policies/smolvla_policy.py`를 만든다. 평가는 기존 `eval/evaluate.py`로 한다.

**Tech Stack:** lerobot 0.4.4 (`C:\Users\asdqw\lerobot`, editable 설치), torch 2.7.1+cu128, RTX 5070 Laptop 8GB, Windows 11

## Global Constraints

- 설계 문서: `docs/superpowers/specs/2026-10-10-so101-ballsort-design.md` §6
- 학습 시작점: `lerobot/smolvla_base`(HF 캐시에 있음)
- 학습 데이터: `local/so101_ballsort_sim`, root는 `C:\Users\asdqw\so101_ballsort\data\so101_ballsort_sim`(600개 에피소드)
- 데이터셋 키 매핑:
  - `observation.images.top` = 손목 카메라
  - `observation.images.front` = 위쪽 카메라
  - smolvla_base의 입력 이름은 `observation.images.camera1/2/3`이다. `rename_map`으로 연결하며, 학습과 추론에 같은 매핑을 써야 한다.
- 8GB GPU 설정:
  - 비전 인코더는 고정한다(`freeze_vision_encoder=True`, 기본값).
  - `train_expert_only=True`(기본값)로 둔다.
  - 배치 크기는 Task 2의 스모크 테스트로 정한다(시작값 8).
- 학습 길이: 20k~30k 스텝
- 통과 기준: 색별 50개, 총 150개 에피소드에서 성공률 0.80 이상. 연속 명령 시험 20회도 함께 기록한다(이 단계의 통과 기준은 아님).
- 시뮬레이션 제자리 자세는 `calib.SIM_HOME_NORM`이다. 평가는 env의 성공 판정을 그대로 쓴다.
- 실패 영상은 `observation.images.front`(위쪽 카메라)로 저장한다.
- 큰 산출물(체크포인트, 로그)은 `outputs/`에 둔다(git 제외). 결과 수치만 `docs/results/`에 커밋한다.

## File Structure

```
policies/
├─ __init__.py
└─ smolvla_policy.py     SmolVLAPolicyAdapter: 체크포인트 → reset(env)/act(obs)
train/
└─ sft.sh                lerobot-train 실행 인자 모음
eval/evaluate.py         (수정) --policy smolvla --checkpoint 지원
tests/test_smolvla_policy.py
docs/results/sft_smoke.md, sft_eval.json
```

---

### Task 1: SmolVLA 정책 어댑터

**Files:**
- Create: `policies/__init__.py`, `policies/smolvla_policy.py`, `tests/test_smolvla_policy.py`
- Modify: `eval/evaluate.py` (CLI의 `--policy` 선택지와 `--checkpoint` 인자)

**Interfaces:**
- Consumes:
  - `BallSortEnv`의 `.task`(명령문)와 obs 키
  - `eval.evaluate.evaluate`, `evaluate_sequential`
- Produces:
  - `RENAME_MAP = {"observation.images.top": "observation.images.camera1", "observation.images.front": "observation.images.camera2"}`
  - `class SmolVLAPolicyAdapter(checkpoint: str, device: str = "cuda", n_action_steps: int | None = None)`
    - `.reset(env)`: 정책의 행동 큐를 비우고 `env.task`를 기억한다.
    - `.act(obs) -> np.ndarray (6,) float32`: 정규화 단위의 관절 목표값
  - CLI: `python -m eval.evaluate --policy smolvla --checkpoint <path> --n-per-color 50 --sequential 20 --video-dir outputs/eval_sft`

참고할 lerobot 코드:
- `src/lerobot/scripts/lerobot_eval.py` 520~560행: 정책과 전처리기를 만드는 순서, `rename_map` 덮어쓰기
- `src/lerobot/policies/factory.py`의 `make_pre_post_processors`
- `src/lerobot/policies/smolvla/modeling_smolvla.py`의 `select_action`, `reset`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_smolvla_policy.py`

GPU와 smolvla_base(HF 캐시)가 필요한 느린 테스트다. 학습 전이므로 smolvla_base로 형식만 검증한다.

```python
import numpy as np
import pytest
import torch

from sim.env import BallSortEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_adapter_returns_finite_joint_targets():
    from policies.smolvla_policy import SmolVLAPolicyAdapter

    env = BallSortEnv(render=True)
    obs, _ = env.reset(seed=0)
    policy = SmolVLAPolicyAdapter("lerobot/smolvla_base")
    policy.reset(env)
    a1 = policy.act(obs)
    obs, *_ = env.step(a1)
    a2 = policy.act(obs)
    for a in (a1, a2):
        assert a.shape == (6,) and a.dtype == np.float32 and np.all(np.isfinite(a))
    env.close()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_smolvla_policy.py -v`
Expected: FAIL (`ModuleNotFoundError: policies.smolvla_policy`)

- [ ] **Step 3: `policies/smolvla_policy.py` 구현**

lerobot-eval과 같은 방식으로 정책과 전처리·후처리 파이프라인을 만든다. 아래는 기준 구현이다. lerobot 0.4.4의 실제 API와 다르면 위 참고 코드에 맞춰 고치고, 바꾼 내용을 보고서에 적는다.

smolvla_base에는 카메라가 3개(`camera1~3`) 정의되어 있고 우리는 2개만 준다. 이때 필요한 처리를 확인한다. 정책 설정의 `empty_cameras`를 쓰거나, 없는 카메라를 처리하는 기존 동작을 따른다. 학습(Task 2)에서 쓰는 설정과 반드시 같아야 한다.

```python
"""SmolVLA 체크포인트를 평가기 정책 프로토콜(reset(env), act(obs))로 감싼다."""

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.front": "observation.images.camera2",
}


class SmolVLAPolicyAdapter:
    def __init__(self, checkpoint, device="cuda", n_action_steps=None):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": RENAME_MAP},
            },
        )
        self.task = None

    def reset(self, env):
        self.policy.reset()
        self.task = env.task

    def _batch(self, obs):
        batch = {"observation.state": torch.from_numpy(obs["observation.state"]).unsqueeze(0), "task": [self.task]}
        for key in ("observation.images.top", "observation.images.front"):
            img = torch.from_numpy(obs[key]).permute(2, 0, 1).float().div(255.0)
            batch[key] = img.unsqueeze(0)
        return batch

    @torch.inference_mode()
    def act(self, obs):
        action = self.policy.select_action(self.pre(self._batch(obs)))
        action = self.post(action)
        return action.squeeze(0).float().cpu().numpy().astype(np.float32)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_smolvla_policy.py -v`
Expected: 1 passed

- [ ] **Step 5: `eval/evaluate.py` CLI 확장**

`main()`만 수정한다.

```python
    ap.add_argument("--policy", choices=["expert", "smolvla"], default="expert")
    ap.add_argument("--checkpoint", default=None, help="--policy smolvla일 때 pretrained_model 경로")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ...
    if args.policy == "smolvla":
        from policies.smolvla_policy import SmolVLAPolicyAdapter

        policy = SmolVLAPolicyAdapter(args.checkpoint, n_action_steps=args.n_action_steps)
    else:
        policy = ExpertPolicy()
```

SmolVLA는 카메라 영상이 필요하다. 그래서 `--policy smolvla`일 때는 `evaluate`와 `evaluate_sequential`에 `render=True`를 넘긴다.

- [ ] **Step 6: 짧은 수동 확인**

Run: `python -m eval.evaluate --policy smolvla --checkpoint lerobot/smolvla_base --n-per-color 1`
Expected: 학습 전 모델이라 성공률은 0일 것이다. 오류 없이 JSON이 출력되면 통과다. 에피소드당 걸린 시간을 보고서에 적는다.

- [ ] **Step 7: 전체 테스트 후 커밋**

```bash
python -m pytest -q
git add policies tests/test_smolvla_policy.py eval/evaluate.py
git commit -m "feat: add SmolVLA policy adapter and smolvla option in evaluator"
```

---

### Task 2: SFT 스크립트 + 스모크 테스트

**Files:**
- Create: `train/sft.sh`, `docs/results/sft_smoke.md`

**Interfaces:**
- Consumes: Task 1의 `RENAME_MAP`(같은 매핑을 써야 한다)
- Produces:
  - `train/sft.sh <job_name> <steps> <batch_size> [추가 lerobot-train 인자...]`
  - 체크포인트는 `outputs/train/<job_name>/checkpoints/<step>/pretrained_model`과 `.../last/pretrained_model`에 생긴다.

- [ ] **Step 1: `train/sft.sh` 작성**

```bash
#!/usr/bin/env bash
# 사용: bash train/sft.sh <job_name> <steps> <batch_size> [추가 lerobot-train 인자...]
set -euo pipefail
JOB=$1; STEPS=$2; BS=$3; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=local/so101_ballsort_sim \
  --dataset.root="$ROOT/data/so101_ballsort_sim" \
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
```

이 스크립트에서 확인할 것:
- Windows Git Bash에서 `--rename_map` JSON이 제대로 전달되는가. `lerobot_train.py` 235행의 PowerShell 보정 코드를 참고한다.
- 3번째 카메라를 다루려면 `--policy.empty_cameras=1` 같은 인자가 필요한가. Task 1에서 정한 방식과 같아야 한다.
- 데이터 로더 `num_workers=4`가 Windows에서 동작하는가. 안 되면 0으로 바꾼다.

- [ ] **Step 2: 스모크 테스트 (배치 크기 결정)**

Run: `bash train/sft.sh smoke_bs8 50 8`
Expected: 50스텝을 마치고 `outputs/train/smoke_bs8/checkpoints/last/pretrained_model`이 생긴다.

실행 중에 `nvidia-smi --query-gpu=memory.used --format=csv`로 최대 메모리를 기록하고, 로그에서 스텝당 시간을 기록한다.
- 메모리 부족(OOM)이 나면 배치 크기를 4로 낮춰 다시 실행한다.
- 메모리가 5GB 미만이면 배치 크기 16도 시험한다.
- 메모리에 여유가 있는 가장 큰 배치 크기를 고른다(목표: 사용량 7GB 이하).

- [ ] **Step 3: 스모크 체크포인트로 어댑터 확인**

Run: `python -m eval.evaluate --policy smolvla --checkpoint outputs/train/smoke_bs8/checkpoints/last/pretrained_model --n-per-color 1`
Expected: 오류 없이 JSON이 출력된다. 학습과 추론의 키 매핑이 맞는지를 이 단계에서 확인한다.

- [ ] **Step 4: 결과 기록과 커밋**

`docs/results/sft_smoke.md`에 다음을 적는다.
- 고른 배치 크기와 그 이유
- 최대 GPU 메모리
- 스텝당 시간
- 예상 학습 시간(30k 스텝 기준)
- 필요했던 추가 인자

```bash
git add train/sft.sh docs/results/sft_smoke.md
git commit -m "feat: add SmolVLA SFT launcher; smoke test picks batch size"
```

---

### Task 3: 본 학습 (SFT)

**Files:**
- Create: `docs/results/sft_train.md`

학습은 백그라운드로 수 시간 돌린다. 코드는 바꾸지 않는다.

- [ ] **Step 1: 학습 시작**

Run(백그라운드): `bash train/sft.sh sft_v1 30000 <Task 2에서 고른 배치> > outputs/sft_v1.log 2>&1`

예상 시간이 10시간을 넘으면 스텝 수를 20000으로 줄인다. 줄인 이유는 기록한다.

- [ ] **Step 2: 진행 확인**

`outputs/sft_v1.log`에서 loss가 줄어드는지 확인한다. 5000스텝마다 체크포인트가 저장되는지도 확인한다. 학습이 중단되면 `--resume=true --config_path=outputs/train/sft_v1/checkpoints/last/pretrained_model/train_config.json`으로 재개한다.

- [ ] **Step 3: 결과 기록과 커밋**

`docs/results/sft_train.md`에 다음을 적는다.
- 최종 스텝
- 총 학습 시간
- loss 추이(1k 간격 표)
- 체크포인트 경로

```bash
git add docs/results/sft_train.md
git commit -m "docs: record SmolVLA SFT training run"
```

---

### Task 4: SFT 모델 평가 (통과 기준 80%)

**Files:**
- Create: `docs/results/sft_eval.json`, `docs/results/sft_eval.md`

- [ ] **Step 1: 정식 평가**

Run: `python -m eval.evaluate --policy smolvla --checkpoint outputs/train/sft_v1/checkpoints/last/pretrained_model --n-per-color 50 --sequential 20 --video-dir outputs/eval_sft > docs/results/sft_eval.json`

`print` 이외의 로그가 섞이면 JSON 부분만 남긴다.

- [ ] **Step 2: 판정과 분석**
  - **`single.success_rate >= 0.80`이면** 통과다. `docs/results/sft_eval.md`에 다음을 적는다.
    - 색별 성공률
    - 실패 유형별 횟수
    - 연속 명령 결과
    - 실패 영상 2~3개에 대한 관찰
  - **0.80 미만이면** 실패 영상을 유형별로 보고 원인 가설을 세운다. 가설의 예:
    - 잘못된 공을 고른다 → 언어 조건이 약하다
    - 잡기에 실패한다 → 정밀도가 부족하다
    - 행동 묶음 길이가 너무 길다
  - 0.80 미만일 때 한 번 시도할 수 있는 조정은 `--n-action-steps 10`으로 다시 평가하는 것이다. 이는 추론 설정만 바꾸며, 행동 묶음 중 앞부분만 실행하고 다시 계획하게 한다.
  - 그래도 미달이면 결과를 사용자에게 보고하고 다음 조치를 상의한다. 다음 조치의 예는 데이터 증가, 학습 길이 연장, 학습 설정 변경이다.

- [ ] **Step 3: 커밋**

```bash
git add docs/results/sft_eval.json docs/results/sft_eval.md
git commit -m "docs: evaluate SFT SmolVLA in sim (gate 80%)"
```

## 완료 기준 (Plan 2)

- `python -m pytest -q` 전부 통과
- `docs/results/sft_eval.json`에서 `single.success_rate >= 0.80`이다. 미달이면 사용자와 상의한 결과를 기록한다.
- Plan 3(DSRL)에서 쓸 SFT 체크포인트 경로가 `docs/results/sft_train.md`에 기록되어 있다.
