"""색 가드 시험. 프레임 고정 자료(tests/data/guard_frames)는 tools/extract_preclose_frames.py 출력의 일부다."""

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from expert.scripted import ScriptedExpert
from policies.color_guard import ColorGuard, classify_wrist
from rl.staged_reward import StagedReward
from sim.env import MAX_STEPS, BallSortEnv

FRAMES = Path(__file__).parent / "data" / "guard_frames"


def _frames(kind):
    rows = list(csv.DictReader(open(FRAMES / "labels.csv")))
    return [(np.asarray(Image.open(FRAMES / r["file"]).convert("RGB")), r["color"]) for r in rows if r["kind"] == kind]


def test_classify_preclose_frames():
    data = _frames("preclose")
    assert len(data) == 30
    acc = np.mean([classify_wrist(img) == c for img, c in data])
    assert acc >= 0.95, acc


def test_classify_approach_frames_none_or_correct():
    data = _frames("approach")
    ok = np.mean([classify_wrist(img) in (None, c) for img, c in data])
    assert ok >= 0.90, ok


def test_classify_empty_table_is_none():
    img = np.full((480, 640, 3), (250, 245, 200), np.uint8)  # 테이블 색만
    assert classify_wrist(img) is None


class AimedExpert:
    """리셋할 때마다 aims의 다음 색 공을 노리는 시연 스크립트 (마지막 색을 계속 쓴다)."""

    def __init__(self, aims):
        self.aims, self.n_resets = list(aims), 0

    def reset(self, env):
        aim = self.aims[min(self.n_resets, len(self.aims) - 1)]
        self.n_resets += 1
        self.ex = ScriptedExpert(env, seed=0, dart_sigma=0.0)
        real = env.target
        env.target = aim
        self.ex.reset()
        env.target = real

    def act(self, obs):
        return self.ex.act()[0]


@pytest.fixture(scope="module")
def env():
    e = BallSortEnv(render=True)
    yield e
    e.close()


def run(env, guard, target, seed=0, stop_at_stage=1):
    obs, _ = env.reset(seed=seed, options={"target": target})
    guard.reset(env)
    sr = StagedReward(env)
    sr.reset()
    info = {}
    for _ in range(MAX_STEPS):
        a = guard.act(obs)
        obs, _, term, trunc, _ = env.step(a)
        _, done, info = sr.step(a)
        if done or term or trunc or info["stage"] >= stop_at_stage:
            break
    return info


def test_guard_passes_correct_close(env):
    guard = ColorGuard(AimedExpert(["red"]))
    info = run(env, guard, "red")
    assert info["first_attempt"] == "target" and info["stage"] >= 1
    assert guard.stats["rejections"] == 0
    assert guard.stats["checks"][-1]["seen"] == "red"


def test_guard_full_episode_succeeds(env):
    """맞는 공을 확인하고 잡은 뒤에는 통에 놓고 제자리로 돌아가며 닫는 것을 막지 않는다."""
    guard = ColorGuard(AimedExpert(["sky blue"]))
    info = run(env, guard, "sky blue", seed=1, stop_at_stage=4)
    assert info["success"] and guard.stats["rejections"] == 0 and len(guard.stats["checks"]) == 1


def test_guard_blocks_wrong_close_then_retries(env):
    policy = AimedExpert(["black", "red"])  # 처음엔 틀린 공, 리셋 후 맞는 공
    guard = ColorGuard(policy)
    info = run(env, guard, "red")
    assert guard.stats["rejections"] == 1 and policy.n_resets == 2
    assert guard.stats["checks"][0]["seen"] == "black"
    assert info["first_attempt"] == "target" and info["attempt_color"] == "red" and info["stage"] >= 1


def _img(rgb):
    img = np.full((480, 640, 3), 128, np.uint8)
    img[300:480, 300:450] = rgb
    return img


class Scripted:
    def __init__(self, acts):
        self.acts, self.i = acts, 0

    def reset(self, env):
        pass

    def act(self, obs):
        a = self.acts[min(self.i, len(self.acts) - 1)]
        self.i += 1
        return np.array([0, 0, 0, 0, 0, a], np.float32)


def test_guard_never_lets_wrong_ball_through():
    """틀린 색만 노리는 정책: 횟수 제한 없이 매번 거부해 틀린 공을 한 번도 잡지 않는다."""
    from types import SimpleNamespace

    class Loop(Scripted):
        def reset(self, env):
            self.i = 0

    guard = ColorGuard(Loop([60] * 3 + [5] * 5))
    guard.reset(SimpleNamespace(task="grab the red ball"))
    obs = {"observation.images.top": _img((90, 180, 240)), "observation.state": np.full(6, 60.0, np.float32)}
    grips = [guard.act(obs)[5] for _ in range(2000)]
    assert guard.stats["rejections"] >= 6
    assert min(grips) >= 20  # 한 번도 잡기 시도(20 미만)로 닫지 않았다


@pytest.mark.parametrize("held_grip, rejected", [(5.0, True), (14.5, False)])
def test_guard_rechecks_after_missed_grasp(held_grip, rejected):
    """명령한 색을 확인하고 닫았는데 집게가 끝까지 닫혔으면(빈손) 놓친 것이다 → 다음 닫기의 색을 다시 확인한다.
    공을 쥔 채 열었으면(놓기) 그 뒤의 닫기는 보지 않는다."""
    from types import SimpleNamespace

    red, blue = _img((220, 20, 20)), _img((90, 180, 240))
    # 열기 → 빨강을 보며 닫기 → 닫힌 채 유지 → 다시 열기 → 파랑을 보며 닫기
    grips = [60] * 3 + [5] * 10 + [60] * 3 + [5]
    guard = ColorGuard(Scripted(grips))
    guard.reset(SimpleNamespace(task="grab the red ball"))
    out = []
    for k, g in enumerate(grips):
        img = red if k < 13 else blue
        state = np.zeros(6, np.float32)
        state[5] = held_grip if 3 < k <= 13 else 60.0  # 관절값은 직전 명령의 결과
        out.append(guard.act({"observation.images.top": img, "observation.state": state}))
    assert guard.stats["checks"][0]["seen"] == "red"
    assert (out[-1][5] > 40) == rejected  # 거부하면 닫지 않고 연다
    assert guard.stats["rejections"] == int(rejected)


def test_command_color_understands_paraphrases():
    from policies.color_guard import command_color

    assert command_color("grab the sky blue ball and put it in the box") == "sky blue"
    assert command_color("Grab the blue ball.") == "sky blue"
    assert command_color("Bring the blue-colored one to the box") == "sky blue"
    assert command_color("Take the black-colored ball.") == "black"
    assert command_color("pick up the red one") == "red"
    assert command_color("move it") is None


@pytest.mark.parametrize("retreat", ["lift", "half"])
def test_short_retreat_retries_faster_and_rechecks(env, retreat):
    """짧은 후퇴: 그리퍼를 연 채 넘겨주므로 다음 닫기도 다시 확인하고, 제자리 후퇴보다 빨리 맞는 공에 시도한다."""
    steps = {}
    for mode in ("home", retreat):
        policy = AimedExpert(["black", "red"])
        guard = ColorGuard(policy, retreat=mode)
        info = run(env, guard, "red")
        assert guard.stats["rejections"] == 1 and policy.n_resets == 2
        assert info["first_attempt"] == "target" and info["attempt_color"] == "red"
        assert len(guard.stats["checks"]) == 2 and guard.stats["checks"][1]["seen"] == "red"
        steps[mode] = guard.t
    assert steps[retreat] < steps["home"] - {"lift": 30, "half": 15}[retreat]  # half는 후퇴 45스텝의 절반을 줄인다


def test_short_retreat_keeps_gripper_open_at_handoff():
    from policies.color_guard import GRIP_OPEN

    g = ColorGuard(AimedExpert(["red"]), retreat="lift")
    g.history = list(np.full((10, 6), 30.0, np.float32))
    plan = g._retreat_plan(np.full(6, 30.0, np.float32))
    assert all(a[5] == GRIP_OPEN for a in plan)
    with pytest.raises(ValueError):
        ColorGuard(AimedExpert(["red"]), retreat="bogus")


def test_reset_retreat_closes_gripper_at_home_and_retries(env):
    """reset 후퇴: 제자리에서 그리퍼를 시작 값까지 닫아 넘기고, 다시 연 뒤의 닫기를 확인해 맞는 공으로 성공한다."""
    from calib import SIM_HOME_NORM

    policy = AimedExpert(["black", "red"])
    guard = ColorGuard(policy, retreat="reset")
    obs, _ = env.reset(seed=0, options={"target": "red"})
    guard.reset(env)
    success = False
    for _ in range(MAX_STEPS):
        a = guard.act(obs)
        if guard.retreat == [] and guard.stats["rejections"] == 1 and policy.n_resets == 2 and not hasattr(guard, "_handoff"):
            guard._handoff = a  # 후퇴의 마지막 행동
        obs, _, success, trunc, _ = env.step(a)
        if success or trunc:
            break
    assert guard.stats["rejections"] == 1 and policy.n_resets == 2
    assert [c["seen"] for c in guard.stats["checks"]] == ["black", "red"]
    assert abs(guard._handoff[5] - SIM_HOME_NORM[5]) < 1e-3 and success


def test_guard_rejects_close_when_color_unknown():
    """잡기 시도 순간(20 미만)까지 색을 확인하지 못하면 닫지 않는다(명령한 색을 확인한 경우에만 잡는다)."""
    from types import SimpleNamespace

    guard = ColorGuard(Scripted([60] * 3 + [30, 5]))
    guard.reset(SimpleNamespace(task="grab the red ball"))
    obs = {"observation.images.top": np.full((480, 640, 3), 128, np.uint8),
           "observation.state": np.full(6, 60.0, np.float32)}
    out = [guard.act(obs) for _ in range(5)]
    assert out[3][5] == 30  # 20 이상이면 다음 스텝에 다시 본다
    assert out[4][5] > 40 and guard.stats["rejections"] == 1
    assert guard.stats["checks"][0]["seen"] is None and guard.stats["checks"][0]["rejected"]
