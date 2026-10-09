import numpy as np

from calib import SIM_HOME_NORM
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
    np.testing.assert_allclose(ex.traj[-1], SIM_HOME_NORM, atol=1e-6)


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
