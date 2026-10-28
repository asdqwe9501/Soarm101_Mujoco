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


def test_guard_gives_up_after_max_retries(env):
    guard = ColorGuard(AimedExpert(["sky blue"]), max_retries=1)  # 시도 시간 제한(600스텝) 안에 끝나도록 1회
    info = run(env, guard, "red")
    assert guard.stats["rejections"] == 1
    # 재시도를 다 쓴 뒤에는 그대로 통과시키므로 그때 처음으로 시도가 기록된다
    assert info["first_attempt"] == "other" and info["attempt_color"] == "sky blue"


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
