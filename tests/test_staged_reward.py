import pytest

from calib import SIM_HOME_NORM
from expert.scripted import ScriptedExpert
from rl.staged_reward import ATTEMPT_TIMEOUT_STEPS, StagedReward
from sim.env import MAX_STEPS, BallSortEnv


def expert_actions(env, seed, aim=None):
    ex = ScriptedExpert(env, seed=seed, dart_sigma=0.0)
    real = env.target
    env.target = aim or real
    ex.reset()
    env.target = real
    return (ex.act()[0] for _ in range(MAX_STEPS))


def run(env, actions_fn, seed=0, target="red"):
    env.reset(seed=seed, options={"target": target})
    sr = StagedReward(env)
    sr.reset()
    total, nonzero, info, n = 0.0, [], {}, 0
    for a in actions_fn(env):
        env.step(a)
        r, done, info = sr.step(a)
        n += 1
        total += r
        if r:
            nonzero.append(r)
        if done:
            break
    return total, nonzero, info, n


@pytest.fixture(scope="module")
def env():
    e = BallSortEnv(render=False)
    yield e
    e.close()


def test_correct_full_sequence_gets_seven(env):
    total, nonzero, info, _ = run(env, lambda e: expert_actions(e, 0))
    assert info["success"] and info["first_attempt"] == "target" and info["stage"] == 4
    assert total == pytest.approx(7.0)
    assert sorted(nonzero) == [1.0, 1.0, 2.0, 3.0]  # 각 단계 한 번씩


def test_wrong_ball_attempt_is_minus_one_and_ends(env):
    total, _, info, _ = run(env, lambda e: expert_actions(e, 0, aim="black"), target="red")
    assert info["first_attempt"] == "other" and info["attempt_color"] == "black"
    assert info["end_reason"] == "wrong_attempt" and total == pytest.approx(-1.0)


def test_no_attempt_times_out(env):
    total, _, info, n = run(env, lambda e: (SIM_HOME_NORM.copy() for _ in range(MAX_STEPS)))
    assert info["end_reason"] == "attempt_timeout" and total == pytest.approx(-0.5)
    assert n == ATTEMPT_TIMEOUT_STEPS


def test_closing_far_from_balls_is_miss(env):
    def open_then_close(e):
        a = SIM_HOME_NORM.copy()
        for g in [60.0] * 30 + [0.0] * 30:
            a[5] = g
            yield a.copy()

    total, _, info, _ = run(env, open_then_close)
    assert info["first_attempt"] == "none" and info["end_reason"] == "miss_attempt"
    assert total == pytest.approx(-0.5)


def test_regrasp_on_target_does_not_double_count(env):
    """맞는 공에 시도한 직후 같은 자리에서 다시 열었다 닫아도 +1은 한 번만."""

    def with_regrasp(e):
        sr_probe = {"done": False}
        for a in expert_actions(e, 3):
            yield a
            if not sr_probe["done"] and a[5] < 20:  # 첫 닫기 직후
                sr_probe["done"] = True
                for g in [60.0] * 5 + [0.0] * 5:
                    b = a.copy()
                    b[5] = g
                    yield b

    total, nonzero, info, _ = run(env, with_regrasp, seed=3)
    assert nonzero.count(1.0) <= 2  # 시도 +1 한 번, 복귀 +1 한 번
    assert total <= 7.0 + 1e-9
