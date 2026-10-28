import numpy as np

from calib import SIM_HOME_NORM
from run.loop import HOLD_STEPS, TaskCtx, run_task


class FakeBackend:
    """관절 상태가 행동을 그대로 따라가는 가짜 로봇."""

    realtime = False

    def __init__(self):
        self.state = SIM_HOME_NORM.astype(np.float32).copy()

    def observe(self):
        return {"observation.state": self.state.copy()}

    def send(self, action):
        self.state = np.asarray(action, np.float32).copy()


class ScriptPolicy:
    def __init__(self, actions):
        self.actions = actions

    def reset(self, ctx):
        self.task, self.i = ctx.task, 0

    def act(self, obs):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


def pose(grip, away=0.0):
    a = SIM_HOME_NORM.astype(np.float32).copy()
    a[0] += away
    a[5] = grip
    return a


def test_done_after_grasp_release_and_home():
    acts = [pose(60, 30)] * 10 + [pose(5, 30)] * 10 + [pose(60, 30)] * 10 + [pose(1.5)] * 40
    rec = run_task(FakeBackend(), ScriptPolicy(acts), "Grab the red ball.", 900)
    assert rec["end"] == "done" and rec["grasped"] and rec["released"]
    assert rec["steps"] == 30 + HOLD_STEPS


def test_not_done_without_release_or_when_away():
    # 잡기만 하고 놓지 않음 → 제자리에 와도 끝나지 않는다
    acts = [pose(60, 30)] * 10 + [pose(5)] * 100
    assert run_task(FakeBackend(), ScriptPolicy(acts), "x red", 120)["end"] == "max_steps"
    # 놓았지만 제자리에서 먼 곳에 머묾
    acts = [pose(60, 30)] * 10 + [pose(5, 30)] * 10 + [pose(60, 30)] * 100
    assert run_task(FakeBackend(), ScriptPolicy(acts), "x red", 120)["end"] == "max_steps"


def test_task_context_passes_command():
    p = ScriptPolicy([pose(1.5)])
    run_task(FakeBackend(), p, "get the blue ball", 1)
    assert p.task == "get the blue ball" and TaskCtx("a").task == "a"
