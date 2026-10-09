# Plan 1: 시뮬레이션 환경 + 시연 데이터 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 실제 작업 공간을 재현한 MuJoCo 환경을 만든다. 그 위에서 시연 스크립트로 성공률 95% 이상을 달성하고, 실제 데이터셋과 같은 형식의 LeRobot v3 시연 데이터셋(600개 에피소드)을 만든다.

**Architecture:** 구성 요소별 역할은 다음과 같다.
- `sim/env.py`: Gymnasium 환경. MuJoCo를 직접 다루는 유일한 모듈이다.
- `calib.py`: 실제 로봇의 정규화 단위와 MuJoCo 라디안을 변환한다.
- `expert/`: 정답 좌표와 역기구학으로 관절 궤적을 만든다.
- `data/collect.py`: 시연을 LeRobot 데이터셋으로 저장한다.
- `eval/evaluate.py`: 정책 인터페이스(`reset(env)`, `act(obs) -> action`)로 성공률을 측정한다. 같은 인터페이스를 Plan 2의 SmolVLA와 Plan 3의 DSRL에서도 쓴다.

**Tech Stack:** Python 3.10, MuJoCo 3.14, gymnasium 1.3, numpy, lerobot 0.4.4 (`C:\Users\asdqw\lerobot`), imageio(+ffmpeg), pytest

**후속 계획:** Plan 2(SmolVLA SFT)와 Plan 3(DSRL + 반복 실행 루프)은 이 계획의 결과(시연 성공률, 카메라 보정 결과)를 보고 작성한다.

## Global Constraints

- 설계 문서: `docs/superpowers/specs/2026-10-10-so101-ballsort-design.md`
- 관측·행동 형식은 실제 데이터셋과 똑같아야 한다.
  - `observation.state`, `action`: float32 (6,)
  - 이름 순서: `shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos, wrist_flex.pos, wrist_roll.pos, gripper.pos`
  - 이미지: `observation.images.top`, `observation.images.front`, 둘 다 480×640×3 uint8
- 단위: 관절 5개는 LeRobot `RANGE_M100_100`, 그리퍼는 `RANGE_0_100`. 제어 주기는 30Hz다.
- 명령문: `grab the {red|black|sky blue} ball and put it in the box`
- 제자리 자세(정규화 단위): `[-2.8, 0.1, 93.7, 71.7, -0.7, 1.5]`
- 성공 기준:
  - 명령된 공이 통 안에 있다(중심 수평거리 < 0.045m, 높이 < 0.07m).
  - 다른 공은 통 안에 없다.
  - 모든 관절이 제자리 자세와 5 단위 이내다.
  - 900스텝 안에 끝난다.
- 좌표: 원점은 로봇 받침이고, z=0이 책상 윗면, +x는 로봇의 정면, +y는 로봇 기준 왼쪽이다. 통 중심은 (0.12, -0.19)이다.
- 실제 데이터셋: `C:\Users\asdqw\.cache\huggingface\lerobot\taeyoungss\SmolVla_grab_blackball` (repo_id `taeyoungss/SmolVla_grab_blackball`)
- 캘리브레이션 파일: `C:\Users\asdqw\.cache\huggingface\lerobot\calibration\robots\so_follower\ty_follower_arm.json`
- 모든 명령은 저장소 루트 `C:\Users\asdqw\so101_ballsort`에서 실행한다.

## 사전 조사로 확인한 사실 (계획의 근거)

1. **캘리브레이션 0점**: 실제 캘리브레이션에서 2047 tick은 제자리(접힌) 자세다. 제자리 자세 값을 변환하면 모든 관절이 약 2047 tick이다. 반면 MuJoCo의 0rad는 팔을 앞으로 편 자세다. 그래서 관절별로 "2047 tick일 때의 MuJoCo 각도" `MID_RAD`가 필요하다.
   - 첫 추정값은 `[0, -1.70, 1.54, 1.20, 0, 0]`이다. 이 값으로 그리퍼 끝은 (0.13, 0, 0)에서 아래를 향한다.
   - Task 4에서 실제 영상과 겹쳐 보며 보정한다.
2. **회전 방향**: MuJoCo `shoulder_pan` +방향은 로봇 기준 오른쪽(-y), 즉 통이 있는 쪽이다. 실제 데이터의 pan 최댓값 +52는 통 방향이므로 부호는 +이다.
3. **그리퍼 방향**: `gripperframe` 사이트의 x축이 그리퍼가 향하는 방향이다.
4. **닿는 범위**:
   - 그리퍼를 아래로 향한 채 닿는 반경은 약 0.28m다(z=0.02 기준). 높이 0.10m 이상에서는 그리퍼를 수직으로 유지하기 어렵다.
   - 따라서 공 배치 범위는 x∈[0.12,0.28], y∈[-0.10,0.20], 반경 0.28 이하로 한다.
   - 집기 직전 높이는 z=0.07, 통 위에서 놓는 높이는 z=0.09로 한다.
   - 그리퍼를 기울이는 것을 허용한다. 역기구학에서 방향 조건의 가중치를 낮게 둔다.
5. **기존 손목 카메라**: menagerie 모델에 `wrist_cam`(fovy 61)이 이미 있다. 이 카메라를 `front`로 쓰고, Task 4에서 실제 영상과 비교해 위치를 고친다.
6. **물리 계산 간격**: 모델의 기본 timestep은 0.005다. 환경에서 `1/(30*17)`로 바꿔 30Hz 제어 한 번에 정확히 17번 계산하게 한다.

## File Structure

```
so101_ballsort/
├─ pytest.ini                 pythonpath 설정
├─ calib.py                   Calib: norm↔rad 변환, 상수
├─ sim/
│  ├─ __init__.py
│  ├─ so101.xml, assets/      menagerie에서 복사 (수정하지 않음)
│  ├─ scene.xml               책상·통·공·카메라
│  └─ env.py                  BallSortEnv
├─ expert/
│  ├─ __init__.py
│  ├─ ik.py                   IK.solve()
│  └─ scripted.py             min_jerk(), ScriptedExpert
├─ data/
│  ├─ __init__.py
│  └─ collect.py              시연 → LeRobotDataset
├─ eval/
│  ├─ __init__.py
│  └─ evaluate.py             evaluate(), evaluate_sequential(), ExpertPolicy
├─ tools/
│  └─ align_overlay.py        실제 영상과 시뮬레이션 겹쳐 보기
└─ tests/
   ├─ test_calib.py
   ├─ test_env.py
   ├─ test_ik.py
   ├─ test_expert.py
   ├─ test_collect.py
   └─ test_evaluate.py
```

---

### Task 1: 프로젝트 골격 + 씬 XML

**Files:**
- Create: `pytest.ini`, `sim/__init__.py`, `sim/scene.xml`, `tests/test_scene.py`
- Copy: `C:\Users\asdqw\mujoco_menagerie\robotstudio_so101\so101.xml` → `sim/so101.xml`, `...\assets\` → `sim/assets/`

**Interfaces:**
- Produces: `sim/scene.xml`
  - body: `bin`, `ball_red`, `ball_black`, `ball_sky_blue`
  - freejoint: `ball_red`, `ball_black`, `ball_sky_blue`
  - camera: `top`, `wrist_cam`
  - geom: `table`

- [ ] **Step 1: 의존성 설치와 모델 복사**

```bash
pip install pytest
mkdir -p sim tests
cp /c/Users/asdqw/mujoco_menagerie/robotstudio_so101/so101.xml sim/
cp -r /c/Users/asdqw/mujoco_menagerie/robotstudio_so101/assets sim/
touch sim/__init__.py
printf "[pytest]\npythonpath = .\ntestpaths = tests\n" > pytest.ini
```

- [ ] **Step 2: 실패하는 테스트 작성** `tests/test_scene.py`

```python
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "sim" / "scene.xml"


def test_scene_has_named_objects():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    for body in ["bin", "ball_red", "ball_black", "ball_sky_blue"]:
        assert m.body(body).id >= 0
    for cam in ["top", "wrist_cam"]:
        assert m.camera(cam).id >= 0
    assert m.nu == 6


def test_balls_rest_on_table():
    m = mujoco.MjModel.from_xml_path(str(SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(500):  # 2.5s
        mujoco.mj_step(m, d)
    for name in ["ball_red", "ball_black", "ball_sky_blue"]:
        z = d.body(name).xpos[2]
        assert abs(z - 0.02) < 0.003, (name, z)
```

- [ ] **Step 3: 테스트 실패 확인**

Run: `python -m pytest tests/test_scene.py -v`
Expected: FAIL (`scene.xml` 없음)

- [ ] **Step 4: `sim/scene.xml` 작성**

공의 초기 위치는 겹치지 않는 임시 값이다. 실제 배치는 env가 리셋할 때 정한다.

```xml
<mujoco model="so101_ballsort">
  <include file="so101.xml"/>

  <statistic center="0.2 0 0.1" extent="0.6"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="180" elevation="-35" offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0.2 0 1.2" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.05" pos="0 0 -0.75" rgba="0.55 0.55 0.55 1"/>
    <!-- 책상 60(y) x 40(x) cm, 윗면 z=0, 로봇은 뒤쪽 가장자리(x=0) 중앙 -->
    <geom name="table" type="box" size="0.20 0.30 0.01" pos="0.20 0 -0.01" rgba="0.82 0.68 0.50 1"
      friction="0.8 0.005 0.0001"/>

    <body name="cam_target" pos="0.20 0 0"/>
    <!-- 책상 위 0.85m. 앞쪽 가장자리 위에서 로봇 쪽을 내려다봄. Task 4에서 보정 -->
    <camera name="top" pos="0.62 0 0.85" mode="targetbody" target="cam_target" fovy="50"/>

    <!-- 투명 원통형 통: 지름 9cm, 높이 7cm. 바닥판 + 벽 조각 16개 -->
    <body name="bin" pos="0.12 -0.19 0">
      <geom name="bin_bottom" type="cylinder" size="0.046 0.0015" pos="0 0 0.0015" rgba="0.85 0.9 0.95 0.35"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0450 0.0000 0.035" euler="0 0 0.0000"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0416 0.0172 0.035" euler="0 0 0.3927"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0318 0.0318 0.035" euler="0 0 0.7854"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0172 0.0416 0.035" euler="0 0 1.1781"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0000 0.0450 0.035" euler="0 0 1.5708"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0172 0.0416 0.035" euler="0 0 1.9635"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0318 0.0318 0.035" euler="0 0 2.3562"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0416 0.0172 0.035" euler="0 0 2.7489"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0450 0.0000 0.035" euler="0 0 3.1416"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0416 -0.0172 0.035" euler="0 0 3.5343"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0318 -0.0318 0.035" euler="0 0 3.9270"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0172 -0.0416 0.035" euler="0 0 4.3197"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="-0.0000 -0.0450 0.035" euler="0 0 4.7124"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0172 -0.0416 0.035" euler="0 0 5.1051"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0318 -0.0318 0.035" euler="0 0 5.4978"/>
      <geom type="box" size="0.0015 0.0102 0.035" rgba="0.85 0.9 0.95 0.35" pos="0.0416 -0.0172 0.035" euler="0 0 5.8905"/>
    </body>

    <!-- 스펀지 공 지름 4cm, 10g -->
    <body name="ball_red" pos="0.20 0.10 0.02">
      <freejoint name="ball_red"/>
      <geom name="ball_red" type="sphere" size="0.02" mass="0.01" rgba="0.85 0.05 0.05 1"
        friction="1.5 0.01 0.001" solref="0.02 1" condim="4"/>
    </body>
    <body name="ball_black" pos="0.20 0.00 0.02">
      <freejoint name="ball_black"/>
      <geom name="ball_black" type="sphere" size="0.02" mass="0.01" rgba="0.05 0.05 0.05 1"
        friction="1.5 0.01 0.001" solref="0.02 1" condim="4"/>
    </body>
    <body name="ball_sky_blue" pos="0.25 0.15 0.02">
      <freejoint name="ball_sky_blue"/>
      <geom name="ball_sky_blue" type="sphere" size="0.02" mass="0.01" rgba="0.45 0.75 0.95 1"
        friction="1.5 0.01 0.001" solref="0.02 1" condim="4"/>
    </body>
  </worldbody>
</mujoco>
```

- [ ] **Step 5: 테스트 통과 확인**

Run: `python -m pytest tests/test_scene.py -v`
Expected: 2 passed

- [ ] **Step 6: 커밋**

```bash
git add pytest.ini sim tests/test_scene.py
git commit -m "feat: add ball sorting scene with table, bin and three balls"
```

---

### Task 2: 단위 변환 (`calib.py`)

**Files:**
- Create: `calib.py`, `tests/test_calib.py`

**Interfaces:**
- Produces:
  - `JOINTS: list[str]` (6개)
  - `FEATURE_NAMES: list[str]` (`"<joint>.pos"`)
  - `HOME_NORM: np.ndarray (6,)`
  - `MID_RAD: np.ndarray (6,)`
  - `class Calib(path=DEFAULT_CALIB, mid_rad=MID_RAD)`
    - `.norm_to_rad(v: array[...,6]) -> array[...,6]`
    - `.rad_to_norm(q) -> array[...,6]`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_calib.py`

```python
import numpy as np

from calib import HOME_NORM, MID_RAD, Calib


def test_roundtrip():
    c = Calib()
    v = np.array([[-38.2, 0.0, -55.2, 30.7, -13.3, 0.3], [52.4, 85.3, 96.5, 88.6, 3.2, 69.9], HOME_NORM])
    np.testing.assert_allclose(c.rad_to_norm(c.norm_to_rad(v)), v, atol=1e-9)


def test_home_maps_near_mid_rad():
    # 실제 캘리브레이션은 제자리 자세에서 약 2047 tick이다
    q = Calib().norm_to_rad(HOME_NORM)
    np.testing.assert_allclose(q, MID_RAD, atol=0.1)


def test_real_data_range_inside_sim_joint_limits():
    import mujoco

    m = mujoco.MjModel.from_xml_path("sim/scene.xml")
    c = Calib()
    # 실제 데이터셋 observation.state 최솟값/최댓값
    for v in [[-38.2, 0.0, -55.2, 30.7, -13.3, 0.3], [52.4, 85.3, 96.5, 88.6, 3.2, 69.9]]:
        q = c.norm_to_rad(np.array(v))
        lo, hi = m.jnt_range[:6, 0], m.jnt_range[:6, 1]
        assert np.all(q >= lo - 0.05) and np.all(q <= hi + 0.05), q
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_calib.py -v`
Expected: FAIL (`ModuleNotFoundError: calib`)

- [ ] **Step 3: `calib.py` 구현**

```python
"""실제 SO101(LeRobot 정규화 단위)과 MuJoCo 관절각(rad) 사이의 변환."""

import json
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FEATURE_NAMES = [f"{j}.pos" for j in JOINTS]
DEFAULT_CALIB = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/ty_follower_arm.json"
)

# 실제 데이터셋 에피소드 시작/끝 평균 (정규화 단위)
HOME_NORM = np.array([-2.8, 0.1, 93.7, 71.7, -0.7, 1.5])

TICKS_PER_REV = 4096
MID_TICK = 2047
# 2047 tick(= 실제 캘리브레이션 중앙 = 제자리 자세)에 해당하는 MuJoCo 관절각.
# tools/align_overlay.py로 실제 영상과 겹쳐 보며 보정한다.
MID_RAD = np.array([0.0, -1.70, 1.54, 1.20, 0.0, 0.0])


class Calib:
    def __init__(self, path=DEFAULT_CALIB, mid_rad=MID_RAD):
        c = json.loads(Path(path).read_text())
        self.lo = np.array([c[j]["range_min"] for j in JOINTS], dtype=float)
        self.hi = np.array([c[j]["range_max"] for j in JOINTS], dtype=float)
        self.mid_rad = np.asarray(mid_rad, dtype=float)

    def _norm_to_ticks(self, v):
        v = np.asarray(v, dtype=float)
        span = self.hi - self.lo
        t = np.empty_like(v)
        t[..., :5] = (v[..., :5] + 100) / 200 * span[:5] + self.lo[:5]
        t[..., 5] = v[..., 5] / 100 * span[5] + self.lo[5]
        return t

    def _ticks_to_norm(self, t):
        span = self.hi - self.lo
        v = np.empty_like(t)
        v[..., :5] = (t[..., :5] - self.lo[:5]) / span[:5] * 200 - 100
        v[..., 5] = (t[..., 5] - self.lo[5]) / span[5] * 100
        return v

    def norm_to_rad(self, v):
        return self.mid_rad + (self._norm_to_ticks(v) - MID_TICK) * 2 * np.pi / TICKS_PER_REV

    def rad_to_norm(self, q):
        t = (np.asarray(q, dtype=float) - self.mid_rad) * TICKS_PER_REV / (2 * np.pi) + MID_TICK
        return self._ticks_to_norm(t)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_calib.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add calib.py tests/test_calib.py
git commit -m "feat: add LeRobot-normalized <-> MuJoCo radian calibration"
```

---

### Task 3: Gymnasium 환경 (`sim/env.py`)

**Files:**
- Create: `sim/env.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: `Calib`, `HOME_NORM`, `FEATURE_NAMES` (Task 2), `sim/scene.xml` (Task 1)
- Produces:
  - `COLORS = ("red", "black", "sky blue")`
  - `task_text(color) -> str`
  - `class BallSortEnv(render: bool = True, calib: Calib | None = None)`
    - `.reset(seed=None, options={"keep_scene": bool, "target": color})` → `(obs, info)`
    - `.step(action_norm (6,))` → `(obs, reward, terminated, truncated, info)`
    - obs: `{"observation.state": float32(6,), "observation.images.top": uint8(480,640,3), "observation.images.front": uint8(480,640,3)}`. 이미지는 `render=True`일 때만 들어간다.
    - info: `{"task": str, "target": color, "success": bool, "target_in_bin": bool, "wrong_in_bin": bool}`
  - 특권 정보 접근:
    - `.ball_pos(color) -> (3,)`
    - `.bin_pos() -> (3,)`
    - `.ball_in_bin(color) -> bool`
    - `.at_home() -> bool`
    - `.joint_norm() -> (6,)`
    - `.model`, `.data`, `.calib`, `.target`, `.task`
  - 상수:
    - `MAX_STEPS = 900`, `N_SUBSTEPS = 17`, `HOME_TOL = 5.0`
    - `BIN_CENTER = (0.12, -0.19)`, `BIN_R_IN = 0.045`, `BIN_H = 0.07`
    - `BALL_X = (0.12, 0.28)`, `BALL_Y = (-0.10, 0.20)`, `BALL_R_MAX = 0.28`
    - `FRONT_CAM = "wrist_cam"`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_env.py`

```python
import numpy as np
import pytest

from calib import HOME_NORM
from sim.env import BALL_R_MAX, COLORS, MAX_STEPS, BallSortEnv, task_text


@pytest.fixture
def env():
    e = BallSortEnv(render=False)
    yield e
    e.close()


def test_task_text():
    assert task_text("sky blue") == "grab the sky blue ball and put it in the box"


def test_reset_obs_and_info(env):
    obs, info = env.reset(seed=0)
    assert obs["observation.state"].shape == (6,)
    assert obs["observation.state"].dtype == np.float32
    np.testing.assert_allclose(obs["observation.state"], HOME_NORM, atol=2.0)
    assert info["target"] in COLORS
    assert info["task"] == task_text(info["target"])


def test_render_images_shape():
    e = BallSortEnv(render=True)
    obs, _ = e.reset(seed=0)
    for k in ["observation.images.top", "observation.images.front"]:
        assert obs[k].shape == (480, 640, 3) and obs[k].dtype == np.uint8
    e.close()


def test_placement_constraints(env):
    for s in range(50):
        env.reset(seed=s)
        ps = [env.ball_pos(c) for c in COLORS]
        b = env.bin_pos()
        for i, p in enumerate(ps):
            assert np.hypot(p[0], p[1]) <= BALL_R_MAX + 1e-6
            assert np.hypot(*(p[:2] - b[:2])) >= 0.12
            for q in ps[i + 1 :]:
                assert np.hypot(*(p[:2] - q[:2])) >= 0.07


def test_target_option(env):
    _, info = env.reset(seed=1, options={"target": "black"})
    assert info["target"] == "black"


def test_balls_stay_still_at_home(env):
    env.reset(seed=3)
    before = {c: env.ball_pos(c).copy() for c in COLORS}
    for _ in range(30):
        env.step(HOME_NORM)
    for c in COLORS:
        assert np.linalg.norm(env.ball_pos(c) - before[c]) < 0.005


def _teleport(env, color, xyz):
    adr = env.model.jnt_qposadr[env.model.joint(env._ball_joint[color]).id]
    env.data.qpos[adr : adr + 3] = xyz
    env.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]


def test_success_when_target_in_bin_and_home(env):
    _, info = env.reset(seed=4)
    b = env.bin_pos()
    _teleport(env, info["target"], [b[0], b[1], 0.025])
    _, reward, terminated, truncated, info = env.step(HOME_NORM)
    assert info["target_in_bin"] and info["success"] and terminated and not truncated
    assert reward == pytest.approx(2.0)  # 통에 들어감 +1, 최종 성공 +1


def test_wrong_ball_blocks_success(env):
    _, info = env.reset(seed=5)
    other = [c for c in COLORS if c != info["target"]][0]
    b = env.bin_pos()
    _teleport(env, info["target"], [b[0] + 0.021, b[1], 0.025])
    _teleport(env, other, [b[0] - 0.021, b[1], 0.025])
    _, reward, terminated, _, info = env.step(HOME_NORM)
    assert info["wrong_in_bin"] and not info["success"] and not terminated
    assert reward == pytest.approx(0.0)  # +1 - 1


def test_reward_events_fire_once(env):
    _, info = env.reset(seed=6)
    b = env.bin_pos()
    _teleport(env, info["target"], [b[0], b[1], 0.025])
    away = np.array(HOME_NORM)
    away[1] += 20  # 제자리에서 벗어난 자세라 아직 성공이 아니다
    _, r1, *_ = env.step(away)
    _, r2, *_ = env.step(away)
    assert r1 == pytest.approx(1.0) and r2 == pytest.approx(0.0)


def test_truncation(env):
    env.reset(seed=7)
    away = np.array(HOME_NORM)
    away[1] += 20
    for i in range(MAX_STEPS):
        _, _, terminated, truncated, _ = env.step(away)
    assert truncated and not terminated


def test_keep_scene(env):
    _, info = env.reset(seed=8, options={"target": "red"})
    b = env.bin_pos()
    _teleport(env, "red", [b[0], b[1], 0.025])
    env.step(HOME_NORM)
    pos = {c: env.ball_pos(c).copy() for c in COLORS}
    for _ in range(10):
        _, info = env.reset(options={"keep_scene": True})
        assert info["target"] != "red"
        for c in COLORS:
            np.testing.assert_allclose(env.ball_pos(c), pos[c], atol=0.003)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_env.py -v`
Expected: FAIL (`ModuleNotFoundError: sim.env`)

- [ ] **Step 3: `sim/env.py` 구현**

```python
"""SO101 색깔 공 분류 Gymnasium 환경. MuJoCo를 직접 다루는 유일한 모듈."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from calib import HOME_NORM, Calib

SCENE_XML = Path(__file__).resolve().parent / "scene.xml"
COLORS = ("red", "black", "sky blue")
FPS = 30
N_SUBSTEPS = 17
MAX_STEPS = 900
HOME_TOL = 5.0
HOME_NOISE = 1.0

BIN_CENTER = (0.12, -0.19)
BIN_JITTER = 0.02
BIN_R_IN = 0.045
BIN_H = 0.07

BALL_RADIUS = 0.02
BALL_X = (0.12, 0.28)
BALL_Y = (-0.10, 0.20)
BALL_R_MAX = 0.28
BALL_MIN_GAP = 0.07
BALL_BIN_GAP = 0.12
BALL_REST_TIP_GAP = 0.06  # 제자리 자세 그리퍼 끝과의 최소 거리

TOP_CAM = "top"
FRONT_CAM = "wrist_cam"
IMG_H, IMG_W = 480, 640


def task_text(color: str) -> str:
    return f"grab the {color} ball and put it in the box"


class BallSortEnv(gym.Env):
    metadata = {"render_fps": FPS}

    def __init__(self, render: bool = True, calib: Calib | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.model.opt.timestep = 1.0 / (FPS * N_SUBSTEPS)
        self.data = mujoco.MjData(self.model)
        self.calib = calib or Calib()
        self.render_images = render
        self._renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None

        self._ball_joint = {c: "ball_" + c.replace(" ", "_") for c in COLORS}
        self._ball_adr = {
            c: self.model.jnt_qposadr[self.model.joint(j).id] for c, j in self._ball_joint.items()
        }
        self._bin_id = self.model.body("bin").id
        self._site_id = self.model.site("gripperframe").id
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        self.action_space = gym.spaces.Box(-100.0, 100.0, (6,), np.float32)
        spaces = {"observation.state": gym.spaces.Box(-100.0, 100.0, (6,), np.float32)}
        if render:
            img = gym.spaces.Box(0, 255, (IMG_H, IMG_W, 3), np.uint8)
            spaces["observation.images.top"] = img
            spaces["observation.images.front"] = img
        self.observation_space = gym.spaces.Dict(spaces)

        self.target = None
        self.task = None
        self._steps = 0
        self._target_rewarded = False
        self._wrong_penalized = set()

    # ---------- 특권 정보 (시연 스크립트, 보상, 평가 전용) ----------
    def ball_pos(self, color: str) -> np.ndarray:
        a = self._ball_adr[color]
        return self.data.qpos[a : a + 3].copy()

    def bin_pos(self) -> np.ndarray:
        return self.model.body_pos[self._bin_id].copy()

    def ball_in_bin(self, color: str) -> bool:
        p, b = self.ball_pos(color), self.bin_pos()
        return bool(np.hypot(p[0] - b[0], p[1] - b[1]) < BIN_R_IN and p[2] < BIN_H)

    def joint_norm(self) -> np.ndarray:
        return self.calib.rad_to_norm(self.data.qpos[:6])

    def at_home(self) -> bool:
        return bool(np.all(np.abs(self.joint_norm() - HOME_NORM) < HOME_TOL))

    # ---------- 내부 ----------
    def _set_ball(self, color, xy):
        a = self._ball_adr[color]
        self.data.qpos[a : a + 3] = [xy[0], xy[1], BALL_RADIUS]
        self.data.qpos[a + 3 : a + 7] = [1, 0, 0, 0]

    def _sample_layout(self, rest_tip_xy):
        rng = self.np_random
        b = np.array(BIN_CENTER) + rng.uniform(-BIN_JITTER, BIN_JITTER, 2)
        self.model.body_pos[self._bin_id][:2] = b
        while True:
            pts = []
            for _ in range(200):
                p = np.array([rng.uniform(*BALL_X), rng.uniform(*BALL_Y)])
                if (
                    np.hypot(*p) <= BALL_R_MAX
                    and np.hypot(*(p - b)) >= BALL_BIN_GAP
                    and np.hypot(*(p - rest_tip_xy)) >= BALL_REST_TIP_GAP
                    and all(np.hypot(*(p - q)) >= BALL_MIN_GAP for q in pts)
                ):
                    pts.append(p)
                    if len(pts) == len(COLORS):
                        return pts

    def _obs(self):
        obs = {"observation.state": self.joint_norm().astype(np.float32)}
        if self.render_images:
            for key, cam in (("observation.images.top", TOP_CAM), ("observation.images.front", FRONT_CAM)):
                self._renderer.update_scene(self.data, camera=cam)
                obs[key] = self._renderer.render().copy()
        return obs

    def _info(self, success=False):
        return {
            "task": self.task,
            "target": self.target,
            "success": success,
            "target_in_bin": self.ball_in_bin(self.target),
            "wrong_in_bin": any(self.ball_in_bin(c) for c in COLORS if c != self.target),
        }

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if not options.get("keep_scene"):
            mujoco.mj_resetData(self.model, self.data)
            home = HOME_NORM + self.np_random.uniform(-HOME_NOISE, HOME_NOISE, 6)
            q = self.calib.norm_to_rad(home)
            self.data.qpos[:6] = q
            self.data.ctrl[:6] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
            mujoco.mj_kinematics(self.model, self.data)
            rest_tip = self.data.site_xpos[self._site_id][:2].copy()
            for c, p in zip(COLORS, self._sample_layout(rest_tip)):
                self._set_ball(c, p)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(N_SUBSTEPS * 6):  # 0.2초 정착
                mujoco.mj_step(self.model, self.data)

        remaining = [c for c in COLORS if not self.ball_in_bin(c)]
        if not remaining:
            raise RuntimeError("all balls are already in the bin")
        target = options.get("target")
        if target is None:
            target = remaining[self.np_random.integers(len(remaining))]
        if target not in remaining:
            raise ValueError(f"target {target!r} is not on the table")
        self.target, self.task = target, task_text(target)
        self._steps = 0
        self._target_rewarded = False
        self._wrong_penalized = {c for c in COLORS if c != target and self.ball_in_bin(c)}
        return self._obs(), self._info()

    def step(self, action):
        q = self.calib.norm_to_rad(np.asarray(action, dtype=float))
        self.data.ctrl[:6] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)
        self._steps += 1

        reward = 0.0
        if self.ball_in_bin(self.target) and not self._target_rewarded:
            self._target_rewarded = True
            reward += 1.0
        for c in COLORS:
            if c != self.target and c not in self._wrong_penalized and self.ball_in_bin(c):
                self._wrong_penalized.add(c)
                reward -= 1.0

        info = self._info()
        success = info["target_in_bin"] and not info["wrong_in_bin"] and self.at_home()
        info["success"] = success
        if success:
            reward += 1.0
        truncated = self._steps >= MAX_STEPS and not success
        return self._obs(), reward, success, truncated, info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_env.py -v`
Expected: 11 passed

`test_balls_stay_still_at_home`가 실패하면 제자리 자세에서 그리퍼가 공과 부딪히는 것이다. `BALL_REST_TIP_GAP`을 0.08로 늘린다.

- [ ] **Step 5: 커밋**

```bash
git add sim/env.py tests/test_env.py
git commit -m "feat: add BallSortEnv with randomized layout, rewards and success check"
```

---

### Task 4: 실제 영상과 겹쳐 보며 보정 (카메라·관절 0점·물체 위치)

**Files:**
- Create: `tools/align_overlay.py`
- Modify: `calib.py` (`MID_RAD`), `sim/scene.xml` (`top` 카메라 pos/fovy, `bin` pos), `sim/so101.xml` (`wrist_cam` pos/quat, 필요할 때만), `sim/env.py` (`BIN_CENTER`, 필요할 때만)

**Interfaces:**
- Consumes: `BallSortEnv`, `Calib`
- Produces: 보정된 상수. 다른 모듈의 인터페이스는 바뀌지 않는다.

이 작업은 사람이 영상을 보고 판단하는 단계다. 자동 테스트 대신 겹친 이미지를 기준으로 판정한다.

- [ ] **Step 1: `tools/align_overlay.py` 작성**

이 스크립트는 실제 데이터셋 프레임의 관절값을 시뮬레이션 로봇에 그대로 적용하고 렌더링한다. 그다음 실제 영상과 50%씩 섞은 이미지를 저장한다.

```python
"""실제 데이터셋 프레임과 시뮬레이션 렌더링을 50% 겹쳐 PNG로 저장한다.

사용: python tools/align_overlay.py --episode 0 --frames 0 150 300
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim.env import BallSortEnv

REPO_ID = "taeyoungss/SmolVla_grab_blackball"
ROOT = Path.home() / ".cache/huggingface/lerobot/taeyoungss/SmolVla_grab_blackball"
OUT = Path("outputs/align")


def to_hwc_uint8(img):
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 150, 300])
    args = ap.parse_args()

    ds = LeRobotDataset(REPO_ID, root=ROOT, episodes=[args.episode], video_backend="pyav")
    env = BallSortEnv(render=True)
    env.reset(seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.frames:
        item = ds[f]
        q = env.calib.norm_to_rad(item["observation.state"].numpy())
        env.data.qpos[:6] = q
        mujoco.mj_forward(env.model, env.data)
        obs = env._obs()
        for key in ["observation.images.top", "observation.images.front"]:
            real = to_hwc_uint8(item[key])
            blend = (0.5 * real + 0.5 * obs[key]).astype(np.uint8)
            name = key.split(".")[-1]
            iio.imwrite(OUT / f"ep{args.episode}_f{f}_{name}.png", np.concatenate([real, obs[key], blend], axis=1))
        print(f, np.round(item["observation.state"].numpy(), 1))
    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 실행해서 첫 겹친 이미지 만들기**

Run: `python tools/align_overlay.py --episode 0 --frames 0 150 300`
Expected: `outputs/align/`에 PNG 6장이 생긴다. 각 이미지는 [실제 | 시뮬레이션 | 겹침] 순서다.

`pyav`로 AV1을 디코딩하지 못하면 `video_backend="torchcodec"`으로 바꿔 다시 실행한다.

- [ ] **Step 3: 관절 0점 보정 (`calib.MID_RAD`)**
  - 기준은 `top` 겹침 이미지의 frame 0, 즉 제자리 자세다. 이 프레임에서 시뮬레이션 로봇의 실루엣이 실제 로봇과 겹치도록 `MID_RAD[1:4]`(lift, elbow, wrist_flex)를 0.05rad 단위로 조정한다.
  - 다음으로 팔을 뻗은 프레임(150, 300)에서도 겹치는지 확인한다. 반대 방향으로 움직이는 관절이 있으면 그 관절의 부호가 반대라는 뜻이다. 그때는 `Calib`에 `sign` 배열을 추가해 해당 관절만 -1로 만들고, `test_calib.py`의 왕복 테스트를 다시 돌린다.
  - 그리퍼: 실제로 열린 프레임에서 열린 정도가 맞는지 확인한다.

- [ ] **Step 4: `top` 카메라 보정**
  - 책상 테두리와 로봇 받침의 위치가 겹치도록 `scene.xml`의 `top` 카메라 `pos`와 `fovy`, 그리고 `cam_target` 위치를 조정한다.
  - 높이는 책상 위 0.85m로 고정한다(사용자 제공 값).
- [ ] **Step 5: 통 위치 보정**
  - 실제 영상의 통과 겹치도록 `scene.xml`의 `bin` pos와 `env.py`의 `BIN_CENTER`를 함께 고친다.
  - 통 크기가 다르게 보이면 지름과 높이도 고친다. 이때 `BIN_R_IN`과 `BIN_H` 상수도 맞춰야 한다.
- [ ] **Step 6: `front`(손목) 카메라 보정**
  - 그리퍼 몸체에 대한 카메라의 상대 위치는 로봇 자세와 무관하다. 그래서 `front` 겹침 이미지에서 집게 끝이 보이는 위치가 같아지도록 `so101.xml`의 `wrist_cam` `pos`, `quat`, `fovy`를 조정한다.
  - 로봇을 정면에서 봤을 때 오른쪽 옆면(로봇 기준 +y 쪽)에 부착되어 있다.
- [ ] **Step 7: 회귀 테스트 + 결과 기록**

Run: `python -m pytest -v`
Expected: 모두 통과

보정 전후 겹침 이미지(frame 0 `top`, `front`)를 `docs/alignment/`에 복사한다.

- [ ] **Step 8: 커밋**

```bash
git add calib.py sim tools docs/alignment
git commit -m "feat: calibrate joint offsets, cameras and bin pose against real recordings"
```

---

### Task 5: 역기구학 (`expert/ik.py`)

**Files:**
- Create: `expert/__init__.py`, `expert/ik.py`, `tests/test_ik.py`

**Interfaces:**
- Consumes: `BallSortEnv.model` (Task 3)
- Produces: `class IK(model)`
  - `.solve(target_xyz, q_init (6,), ori_weight=0.1) -> (q (6,), pos_err_m: float, tilt_deg: float)`
  - 관절 0~3만 푼다. `wrist_roll`과 `gripper`는 `q_init` 값을 그대로 둔다.
  - `tilt_deg`는 그리퍼 방향과 연직 아래 방향 사이의 각도다.

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_ik.py`

```python
import numpy as np
import pytest

from calib import HOME_NORM, Calib
from expert.ik import IK
from sim.env import BALL_R_MAX, BALL_X, BALL_Y, BIN_CENTER, BIN_JITTER, BallSortEnv


@pytest.fixture(scope="module")
def ik_and_home():
    env = BallSortEnv(render=False)
    return IK(env.model), Calib().norm_to_rad(HOME_NORM)


def _grid():
    for x in np.linspace(*BALL_X, 5):
        for y in np.linspace(*BALL_Y, 6):
            if np.hypot(x, y) <= BALL_R_MAX:
                yield x, y


@pytest.mark.parametrize("z", [0.012, 0.07])
def test_reach_ball_region(ik_and_home, z):
    ik, home = ik_and_home
    for x, y in _grid():
        q, err, tilt = ik.solve(np.array([x, y, z]), home)
        assert err < 0.003, (x, y, z, err)
        assert tilt < 35, (x, y, z, tilt)


def test_reach_bin_release(ik_and_home):
    ik, home = ik_and_home
    for dx in (-BIN_JITTER, 0, BIN_JITTER):
        for dy in (-BIN_JITTER, 0, BIN_JITTER):
            p = np.array([BIN_CENTER[0] + dx, BIN_CENTER[1] + dy, 0.09])
            q, err, tilt = ik.solve(p, home)
            assert err < 0.005 and tilt < 45, (p, err, tilt)


def test_keeps_roll_and_gripper(ik_and_home):
    ik, home = ik_and_home
    q, _, _ = ik.solve(np.array([0.2, 0.0, 0.05]), home)
    assert q[4] == home[4] and q[5] == home[5]
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_ik.py -v`
Expected: FAIL (`ModuleNotFoundError: expert.ik`)

- [ ] **Step 3: `expert/ik.py` 구현**

```python
"""gripperframe 사이트 위치를 목표로 하는 감쇠 최소제곱 IK. 그리퍼는 아래를 향하도록 약하게 유도한다."""

import mujoco
import numpy as np

DOWN = np.array([0.0, 0.0, -1.0])


class IK:
    def __init__(self, model, n_seeds=8, iters=300, damping=1e-5, max_dq=0.1, seed=0):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site("gripperframe").id
        self.lo = model.jnt_range[:6, 0]
        self.hi = model.jnt_range[:6, 1]
        self.n_seeds, self.iters, self.damping, self.max_dq = n_seeds, iters, damping, max_dq
        self.rng = np.random.default_rng(seed)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def _fk(self, q):
        self.data.qpos[:6] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        return self.data.site_xpos[self.site], self.data.site_xmat[self.site].reshape(3, 3)

    def _descend(self, target, q, w):
        for _ in range(self.iters):
            p, R = self._fk(q)
            ep = target - p
            eo = np.cross(R[:, 0], DOWN)
            mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
            J = np.vstack([self._jp, w * self._jr])[:, :4]
            e = np.concatenate([ep, w * eo])
            dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), e)
            q[:4] = np.clip(q[:4] + np.clip(dq, -self.max_dq, self.max_dq), self.lo[:4], self.hi[:4])
        p, R = self._fk(q)
        tilt = np.degrees(np.arccos(np.clip(R[:, 0] @ DOWN, -1.0, 1.0)))
        return q, float(np.linalg.norm(target - p)), float(tilt)

    def solve(self, target_xyz, q_init, ori_weight=0.1):
        target = np.asarray(target_xyz, dtype=float)
        starts = [np.array(q_init, dtype=float)]
        for _ in range(self.n_seeds):
            q = np.array(q_init, dtype=float)
            q[0] = np.arctan2(-target[1], target[0])  # pan +방향 = -y
            q[1:4] = self.rng.uniform(self.lo[1:4], self.hi[1:4])
            starts.append(q)
        results = [self._descend(target, s.copy(), ori_weight) for s in starts]
        ok = [r for r in results if r[1] < 0.003]
        return min(ok, key=lambda r: r[2]) if ok else min(results, key=lambda r: r[1])
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_ik.py -v`
Expected: 4 passed

`test_reach_ball_region`이 특정 지점에서만 실패하면 로봇이 닿지 않는 영역이다. `sim/env.py`의 `BALL_R_MAX`를 0.01씩 줄여 다시 실행한다. 통과한 값은 커밋 메시지에 기록한다.

- [ ] **Step 5: 커밋**

```bash
git add expert tests/test_ik.py sim/env.py
git commit -m "feat: add multi-start damped least-squares IK for gripper site"
```

---

### Task 6: 시연 스크립트 (`expert/scripted.py`)

**Files:**
- Create: `expert/scripted.py`, `tests/test_expert.py`

**Interfaces:**
- Consumes: `IK` (Task 5), `BallSortEnv` 특권 정보 (Task 3), `HOME_NORM` (Task 2)
- Produces:
  - `min_jerk(a, b, s) -> array`
  - `class ScriptedExpert(env, seed=0, dart_sigma=0.5)`
    - `.reset()`: 현재 env 상태로 궤적을 계획한다. env.reset 직후에 호출한다.
    - `.act() -> (executed (6,), clean (6,))`: 정규화 단위. `executed`는 노이즈가 섞인 실행용 행동이고, `clean`은 데이터셋 라벨용 정답 행동이다.
  - 상수: `GRIP_OPEN = 60.0`, `GRIP_CLOSED = 0.0`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_expert.py`

```python
import numpy as np

from calib import HOME_NORM
from expert.scripted import ScriptedExpert, min_jerk
from sim.env import BallSortEnv


def test_min_jerk_endpoints():
    a, b = np.zeros(6), np.ones(6)
    np.testing.assert_allclose(min_jerk(a, b, 0.0), a)
    np.testing.assert_allclose(min_jerk(a, b, 1.0), b)
    np.testing.assert_allclose(min_jerk(a, b, 0.5), 0.5 * b)


def test_trajectory_starts_and_ends_at_home():
    env = BallSortEnv(render=False)
    env.reset(seed=0)
    ex = ScriptedExpert(env, seed=0, dart_sigma=0.0)
    ex.reset()
    assert ex.n_steps < 900
    first = ex.act()[1]
    np.testing.assert_allclose(first, env.joint_norm(), atol=1.5)
    np.testing.assert_allclose(ex.traj[-1], HOME_NORM, atol=1e-6)


def run_episode(env, ex, seed):
    _, info = env.reset(seed=seed)
    ex.reset()
    for _ in range(900):
        executed, _ = ex.act()
        _, _, terminated, truncated, info = env.step(executed)
        if terminated or truncated:
            break
    return info["success"]


def test_expert_success_rate():
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=0)
    wins = sum(run_episode(env, ex, s) for s in range(20))
    assert wins >= 19, wins  # 95% 이상
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_expert.py -v`
Expected: FAIL (`ModuleNotFoundError: expert.scripted`)

- [ ] **Step 3: `expert/scripted.py` 구현**

```python
"""정답 좌표 + IK로 관절 궤적을 만드는 시연 스크립트.

제자리 → 공 위 → 하강 → 닫기 → 들기 → 통 위 → 열기 → 제자리
"""

import numpy as np

from calib import HOME_NORM
from expert.ik import IK
from sim.env import FPS

GRIP_OPEN = 60.0
GRIP_CLOSED = 0.0
PRE_Z = 0.07        # 공 위 대기 높이 (gripperframe 기준)
GRASP_Z = 0.012     # 집는 높이 (공 중심 0.02보다 약간 아래)
LIFT_Z = 0.08
RELEASE_Z = 0.09  # 공 아래면 ≈0.08 > 통 높이 0.07
# 각 구간 길이 [s]: 공 위, 하강, 닫기, 들기, 통 위, 열기, 제자리, 대기
DURATIONS = [3.0, 1.5, 1.0, 1.5, 3.0, 1.0, 3.5, 1.0]
XY_JITTER = 0.005
DART_RHO = 0.9


def min_jerk(a, b, s):
    s = np.clip(s, 0.0, 1.0)
    w = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a + (b - a) * w


class ScriptedExpert:
    def __init__(self, env, seed=0, dart_sigma=0.5):
        self.env = env
        self.ik = IK(env.model)
        self.rng = np.random.default_rng(seed)
        self.dart_sigma = dart_sigma
        self.traj = None
        self.t = 0

    @property
    def n_steps(self):
        return len(self.traj)

    def _pose(self, xyz, q_prev_rad, grip):
        q, _, _ = self.ik.solve(xyz, q_prev_rad)
        v = self.env.calib.rad_to_norm(q)
        v[5] = grip
        return v, q

    def reset(self):
        env, rng = self.env, self.rng
        ball = env.ball_pos(env.target)
        bin_ = env.bin_pos()
        jit = lambda: rng.uniform(-XY_JITTER, XY_JITTER, 2)
        start = env.joint_norm()
        q0 = env.calib.norm_to_rad(HOME_NORM)

        pre, q = self._pose([*(ball[:2] + jit()), PRE_Z], q0, GRIP_OPEN)
        grasp, q = self._pose([ball[0], ball[1], GRASP_Z], q, GRIP_OPEN)
        closed = grasp.copy()
        closed[5] = GRIP_CLOSED
        lift, q = self._pose([ball[0], ball[1], LIFT_Z], q, GRIP_CLOSED)
        above_bin, q = self._pose([*(bin_[:2] + jit()), RELEASE_Z], q, GRIP_CLOSED)
        release = above_bin.copy()
        release[5] = GRIP_OPEN
        home = HOME_NORM.copy()

        keys = [start, pre, grasp, closed, lift, above_bin, release, home, home]
        scale = rng.uniform(0.85, 1.15, len(DURATIONS))
        traj = []
        for a, b, d, k in zip(keys[:-1], keys[1:], DURATIONS, scale):
            n = max(2, int(round(d * k * FPS)))
            traj += [min_jerk(a, b, i / (n - 1)) for i in range(n)]
        self.traj = np.array(traj)
        self.t = 0
        self._noise = np.zeros(6)

    def act(self):
        clean = self.traj[min(self.t, len(self.traj) - 1)].copy()
        self.t += 1
        eps = self.rng.normal(0.0, self.dart_sigma * np.sqrt(1 - DART_RHO**2), 6)
        self._noise = DART_RHO * self._noise + eps
        self._noise[5] = 0.0  # 그리퍼에는 노이즈를 넣지 않음
        return clean + self._noise, clean
```

- [ ] **Step 4: 테스트 실행**

Run: `python -m pytest tests/test_expert.py -v`
Expected: 3 passed

`test_expert_success_rate`가 실패하면 원인 구간을 진단한다. 실패한 seed에 대해 뷰어로 재생하는 스크립트를 실행한다(`python -m mujoco.viewer`는 쓰지 않는다).

```python
# scratch: 실패 seed 재생
import mujoco.viewer, time
from sim.env import BallSortEnv
from expert.scripted import ScriptedExpert
env = BallSortEnv(render=False); ex = ScriptedExpert(env, seed=0)
env.reset(seed=SEED); ex.reset()
with mujoco.viewer.launch_passive(env.model, env.data) as v:
    for _ in range(900):
        env.step(ex.act()[0]); v.sync(); time.sleep(1/30)
```

증상별로 조정할 상수는 다음과 같다.
- 공이 미끄러진다: `GRASP_Z`, 공 `friction`/`solref`
- 놓은 공이 통 밖으로 튄다: `RELEASE_Z`, 공 `solref`
- 통 벽에 부딪힌다: `LIFT_Z`, 구간 길이

95% 이상이 될 때까지 조정한다.

- [ ] **Step 5: 커밋**

```bash
git add expert/scripted.py tests/test_expert.py
git commit -m "feat: add scripted pick-and-place expert with min-jerk and DART noise"
```

---

### Task 7: 평가기 (`eval/evaluate.py`) + 시연 스크립트 성공률 측정

**Files:**
- Create: `eval/__init__.py`, `eval/evaluate.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `BallSortEnv`, `COLORS`, `ScriptedExpert`
- Produces:
  - 정책 프로토콜: 객체에 `.reset(env)`와 `.act(obs) -> action (6,)`이 있으면 된다.
  - `class ExpertPolicy(seed=0, dart_sigma=0.0)`
  - `evaluate(policy, n_per_color, seed=0, render=False, video_dir=None, max_videos=10) -> dict`
    - 결과 키: `success_rate`, `per_color: {color: rate}`, `failures: {kind: count}`, `n`
  - `evaluate_sequential(policy, n_trials, seed=0, render=False) -> dict`
    - 결과 키: `all_success_rate`, `per_command_rate`, `n`
  - 실패 유형: `wrong_ball`, `grasp_fail`, `release_fail`, `return_fail`
  - CLI: `python -m eval.evaluate --policy expert --n-per-color 50`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_evaluate.py`

```python
from calib import HOME_NORM
from eval.evaluate import ExpertPolicy, classify_failure, evaluate, evaluate_sequential


class StayHome:
    def reset(self, env):
        pass

    def act(self, obs):
        return HOME_NORM


def test_classify_failure():
    assert classify_failure({"wrong_in_bin": True, "target_in_bin": False}, max_target_z=0.1) == "wrong_ball"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": False}, max_target_z=0.02) == "grasp_fail"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": False}, max_target_z=0.08) == "release_fail"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": True}, max_target_z=0.08) == "return_fail"


def test_evaluate_stay_home_fails_everything():
    r = evaluate(StayHome(), n_per_color=1, seed=0)
    assert r["n"] == 3 and r["success_rate"] == 0.0 and r["failures"]["grasp_fail"] == 3


def test_evaluate_expert():
    r = evaluate(ExpertPolicy(), n_per_color=2, seed=100)
    assert r["success_rate"] >= 5 / 6


def test_sequential_expert():
    r = evaluate_sequential(ExpertPolicy(), n_trials=2, seed=200)
    assert r["n"] == 2 and 0.0 <= r["all_success_rate"] <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL (`ModuleNotFoundError: eval.evaluate`)

- [ ] **Step 3: `eval/evaluate.py` 구현**

```python
"""정책 성공률 평가. 정책 프로토콜: reset(env), act(obs) -> action(6,)"""

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio
import numpy as np

from expert.scripted import ScriptedExpert
from sim.env import COLORS, FPS, MAX_STEPS, BallSortEnv

LIFTED_Z = 0.05


class ExpertPolicy:
    """특권 정보를 쓰는 시연 스크립트를 정책 프로토콜로 감싼 것 (평가기 검증용)."""

    def __init__(self, seed=0, dart_sigma=0.0):
        self.seed, self.dart_sigma = seed, dart_sigma
        self.expert = None

    def reset(self, env):
        if self.expert is None or self.expert.env is not env:
            self.expert = ScriptedExpert(env, seed=self.seed, dart_sigma=self.dart_sigma)
        self.expert.reset()

    def act(self, obs):
        return self.expert.act()[0]


def classify_failure(info, max_target_z):
    if info["wrong_in_bin"]:
        return "wrong_ball"
    if info["target_in_bin"]:
        return "return_fail"
    return "release_fail" if max_target_z > LIFTED_Z else "grasp_fail"


def run_episode(env, policy, obs, frames=None):
    policy.reset(env)
    max_z, info = 0.0, None
    for _ in range(MAX_STEPS):
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        max_z = max(max_z, env.ball_pos(env.target)[2])
        if frames is not None:
            frames.append(obs["observation.images.top"])
        if terminated or truncated:
            break
    return obs, info, max_z


def evaluate(policy, n_per_color, seed=0, render=False, video_dir=None, max_videos=10):
    env = BallSortEnv(render=render or video_dir is not None)
    wins, per_color, failures, n_videos = 0, Counter(), Counter(), 0
    ep = 0
    for color in COLORS:
        for _ in range(n_per_color):
            obs, _ = env.reset(seed=seed + ep, options={"target": color})
            frames = [] if video_dir is not None else None
            _, info, max_z = run_episode(env, policy, obs, frames)
            if info["success"]:
                wins += 1
                per_color[color] += 1
            else:
                kind = classify_failure(info, max_z)
                failures[kind] += 1
                if frames and n_videos < max_videos:
                    Path(video_dir).mkdir(parents=True, exist_ok=True)
                    name = f"ep{ep:04d}_{color.replace(' ', '_')}_{kind}.mp4"
                    imageio.mimsave(Path(video_dir) / name, frames, fps=FPS)
                    n_videos += 1
            ep += 1
    env.close()
    n = n_per_color * len(COLORS)
    return {
        "n": n,
        "success_rate": wins / n,
        "per_color": {c: per_color[c] / n_per_color for c in COLORS},
        "failures": {k: failures[k] for k in ["wrong_ball", "grasp_fail", "release_fail", "return_fail"]},
    }


def evaluate_sequential(policy, n_trials, seed=0, render=False):
    """공 3개를 무작위 순서로 연속 명령한다. 한 시험 안에서는 장면을 유지한다."""
    env = BallSortEnv(render=render)
    rng = np.random.default_rng(seed)
    all_ok, cmd_ok = 0, 0
    for t in range(n_trials):
        order = list(rng.permutation(COLORS))
        ok = True
        for i, color in enumerate(order):
            if i == 0:
                obs, _ = env.reset(seed=seed + t, options={"target": color})
            else:
                obs, _ = env.reset(options={"keep_scene": True, "target": color})
            _, info, _ = run_episode(env, policy, obs)
            cmd_ok += info["success"]
            if not info["success"]:
                ok = False
                break
        all_ok += ok
    env.close()
    return {"n": n_trials, "all_success_rate": all_ok / n_trials, "per_command_rate": cmd_ok / (3 * n_trials)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["expert"], default="expert")
    ap.add_argument("--n-per-color", type=int, default=50)
    ap.add_argument("--sequential", type=int, default=0, help="연속 명령 시험 횟수 (0이면 생략)")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--video-dir", default=None)
    args = ap.parse_args()
    policy = ExpertPolicy()
    result = {"single": evaluate(policy, args.n_per_color, args.seed, video_dir=args.video_dir)}
    if args.sequential:
        result["sequential"] = evaluate_sequential(policy, args.sequential, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

참고: `run_episode`의 실패 정보는 마지막 스텝 기준이다. 시간 초과(truncated) 때 info로 유형을 분류한다.

연속 명령 모드에서 `reset(keep_scene=True)`를 하면 `HOME_NOISE`가 적용되지 않고 로봇은 직전 자세에서 시작한다. 직전 명령이 성공했다면 로봇은 제자리에 있다. 실패했다면 그 시험은 그 자리에서 중단된다.

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: 4 passed

- [ ] **Step 5: 시연 스크립트 정식 측정 (데이터 수집 전 통과 기준)**

Run: `python -m eval.evaluate --policy expert --n-per-color 50 --sequential 20 --video-dir outputs/eval_expert`
Expected: `single.success_rate >= 0.95`. 결과 JSON을 `docs/results/expert_eval.json`에 저장한다.

95% 미만이면 Task 6 Step 4의 진단 절차로 돌아간다.

- [ ] **Step 6: 커밋**

```bash
git add eval tests/test_evaluate.py docs/results/expert_eval.json
git commit -m "feat: add policy evaluator with failure classification; expert passes 95% gate"
```

---

### Task 8: 시연 데이터 수집 (`data/collect.py`)

**Files:**
- Create: `data/__init__.py`, `data/collect.py`, `tests/test_collect.py`

**Interfaces:**
- Consumes: `BallSortEnv`, `ScriptedExpert`, `FEATURE_NAMES`, `COLORS`
- Produces:
  - `FEATURES: dict`: 실제 데이터셋과 같은 이름, 모양, dtype
  - `collect(repo_id, root, n_per_color, seed=0) -> dict`
    - 결과 키: `saved`, `discarded`, `per_color`
  - CLI: `python -m data.collect --repo-id local/so101_ballsort_sim --root data/so101_ballsort_sim --n-per-color 200`

- [ ] **Step 1: 실패하는 테스트 작성** `tests/test_collect.py`

```python
import json

from data.collect import collect


def test_collect_small_dataset(tmp_path):
    root = tmp_path / "ds"
    stats = collect("local/test_ballsort", root, n_per_color=1, seed=0)
    assert stats["saved"] == 3

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test_ballsort", root=root, video_backend="pyav")
    assert ds.num_episodes == 3
    assert set(ds.meta.tasks.index) == {
        "grab the red ball and put it in the box",
        "grab the black ball and put it in the box",
        "grab the sky blue ball and put it in the box",
    }
    item = ds[0]
    assert tuple(item["observation.state"].shape) == (6,)
    assert tuple(item["action"].shape) == (6,)
    assert tuple(item["observation.images.top"].shape) == (3, 480, 640)

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["fps"] == 30 and info["robot_type"] == "so_follower"
    assert info["features"]["action"]["names"][0] == "shoulder_pan.pos"
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `python -m pytest tests/test_collect.py -v`
Expected: FAIL (`ModuleNotFoundError: data.collect`)

- [ ] **Step 3: `data/collect.py` 구현**

```python
"""시연 스크립트를 실행해 LeRobot v3 데이터셋으로 저장한다. 성공한 에피소드만 남긴다."""

import argparse
import json
from collections import Counter

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from calib import FEATURE_NAMES
from expert.scripted import ScriptedExpert
from sim.env import COLORS, FPS, IMG_H, IMG_W, MAX_STEPS, BallSortEnv

_IMG = {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "action": {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES},
    "observation.images.front": _IMG,
    "observation.images.top": _IMG,
}


def collect(repo_id, root, n_per_color, seed=0):
    ds = LeRobotDataset.create(repo_id, FPS, FEATURES, root=root, robot_type="so_follower", use_videos=True)
    env = BallSortEnv(render=True)
    expert = ScriptedExpert(env, seed=seed)
    saved, discarded, per_color = 0, 0, Counter()
    ep_seed = seed
    try:
        for color in COLORS:
            while per_color[color] < n_per_color:
                obs, info = env.reset(seed=ep_seed, options={"target": color})
                ep_seed += 1
                expert.reset()
                for _ in range(MAX_STEPS):
                    executed, clean = expert.act()
                    ds.add_frame(
                        {
                            "observation.state": obs["observation.state"],
                            "action": clean.astype(np.float32),
                            "observation.images.top": obs["observation.images.top"],
                            "observation.images.front": obs["observation.images.front"],
                            "task": info["task"],
                        }
                    )
                    obs, _, terminated, truncated, info = env.step(executed)
                    if terminated or truncated:
                        break
                if info["success"]:
                    ds.save_episode()
                    saved += 1
                    per_color[color] += 1
                    print(f"[{saved}] {color} saved (seed {ep_seed - 1})", flush=True)
                else:
                    ds.clear_episode_buffer()
                    discarded += 1
    finally:
        ds.finalize()
        env.close()
    return {"saved": saved, "discarded": discarded, "per_color": dict(per_color)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/so101_ballsort_sim")
    ap.add_argument("--root", default="data/so101_ballsort_sim")
    ap.add_argument("--n-per-color", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(collect(args.repo_id, args.root, args.n_per_color, args.seed), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `python -m pytest tests/test_collect.py -v`
Expected: 1 passed. 영상 인코딩 때문에 수 분이 걸릴 수 있다.

- [ ] **Step 5: `.gitignore` 추가 후 커밋**

```bash
printf "outputs/\ndata/so101_ballsort_sim*/\n__pycache__/\n" > .gitignore
git add .gitignore data/__init__.py data/collect.py tests/test_collect.py
git commit -m "feat: collect successful scripted demos into LeRobot v3 dataset"
```

- [ ] **Step 6: 시범 수집 10개 + 영상 확인**

Run: `python -m data.collect --root data/so101_ballsort_sim_pilot --repo-id local/so101_ballsort_sim_pilot --n-per-color 4`
Expected: `saved: 12`

`data/so101_ballsort_sim_pilot/videos/` 아래 mp4를 열어 세 가지를 확인한다.
- 두 카메라 영상이 실제 데이터셋과 비슷한 구도인지
- 공 색이 구분되는지
- 동작이 끊기지 않는지

확인 결과를 사용자에게 보고하고, 문제가 없다는 확인을 받는다.

- [ ] **Step 7: 전체 수집 (600개)**

Run: `python -m data.collect --n-per-color 200` (백그라운드 실행 권장, 수 시간)
Expected: `saved: 600`. `discarded / (saved + discarded)`가 5% 이하여야 한다(시연 성공률과 일치).

결과 JSON을 `docs/results/collect.json`에 저장하고 커밋한다.

```bash
git add docs/results/collect.json
git commit -m "data: collect 600 scripted demos (200 per color)"
```

---

## 완료 기준 (Plan 1)

- `python -m pytest -v` 전부 통과
- `docs/alignment/`에 보정된 겹침 이미지가 있다.
- `docs/results/expert_eval.json`에서 단일 명령 성공률이 0.95 이상이다.
- `data/so101_ballsort_sim`에 600개 에피소드(색별 200개)가 있다.
