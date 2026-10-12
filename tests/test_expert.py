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


def test_correction_labels_point_to_true_pre_pose():
    """보정 모드: 빗나간 구간의 라벨은 실행 명령보다 항상 올바른 공 위 자세에 가깝거나 같다."""
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=1, correction=True)
    for s in range(200, 205):
        env.reset(seed=s)
        ex.reset()
        assert ex.n_steps < 800
        off = np.flatnonzero(ex._off_mask)
        assert len(off) > 30
        pre = ex.pre_pose[:5]
        prev_grip = None
        for t in range(ex.n_steps):
            cur = env.joint_norm()
            executed, clean = ex.act()
            if ex._off_mask[t]:
                # (a) 팔 관절 라벨은 현재값과 올바른 자세 사이(오버슈트 없음)
                lo = np.minimum(cur[:5], pre) - 1e-6
                hi = np.maximum(cur[:5], pre) + 1e-6
                assert np.all((clean[:5] >= lo) & (clean[:5] <= hi)), (s, t)
                # (b) 그리퍼 라벨은 최소저크 궤적 값이며 스텝 간 급변 없음
                assert abs(clean[5] - ex.traj[t][5]) < 1e-9, (s, t)
                if prev_grip is not None:
                    assert abs(clean[5] - prev_grip) < 5.0, (s, t)
                prev_grip = clean[5]
                # (c) L2 조건
                d_label = np.linalg.norm(clean[:5] - pre)
                d_exec = np.linalg.norm(executed[:5] - pre)
                assert d_label <= d_exec + 1e-9, (s, t, d_label, d_exec)
            if not ex._off_mask[t]:
                prev_grip = None
            env.step(executed)
        # 빗나간 지점은 실제로 joint 공간에서 올바른 자세와 떨어져 있어야 한다
        assert np.linalg.norm(ex.traj[off[-1], :5] - pre) > 2.0


def test_correction_success_rate():
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=0, correction=True)
    wins = sum(run_episode(env, ex, s) for s in range(100, 120))
    assert wins >= 19, wins


# ---------- Plan 5 ----------
import pytest

from sim.env import COLORS


def run_episode_opts(env, ex, seed, options):
    _, info = env.reset(seed=seed, options=options)
    ex.reset()
    for _ in range(900):
        executed, _ = ex.act()
        _, _, terminated, truncated, info = env.step(executed)
        if terminated or truncated:
            break
    return info["success"]


@pytest.mark.parametrize("scene", ["three", "two", "prebinned"])
def test_expert_success_per_scene_randomized(scene):
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=0)
    wins = sum(run_episode_opts(env, ex, s, {"scene": scene, "randomize": True}) for s in range(300, 310))
    assert wins >= 9, (scene, wins)


def test_wrong_approach_labels_point_to_true_pre_pose():
    """잘못된 공 접근 구간의 라벨은 실행 명령보다 항상 명령한 색 공 위 자세에 가깝거나 같다."""
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=2, wrong_approach=True)
    for s in range(210, 215):
        env.reset(seed=s)
        ex.reset()
        off = np.flatnonzero(ex._off_mask)
        assert len(off) > 30
        pre = ex.pre_pose[:5]
        wrong = env.ball_pos(ex.wrong_color)[:2]
        min_d = np.inf
        for t in range(ex.n_steps):
            cur = env.joint_norm()
            executed, clean = ex.act()
            if ex._off_mask[t]:
                d_label = np.linalg.norm(clean[:5] - pre)
                d_exec = np.linalg.norm(executed[:5] - pre)
                assert d_label <= d_exec + 1e-9, (s, t, d_label, d_exec)
                assert abs(clean[5] - ex.traj[t][5]) < 1e-9
            env.step(executed)
            tip = env.data.site_xpos[env._site_id]
            if ex._off_mask[t]:
                min_d = min(min_d, np.hypot(*(tip[:2] - wrong)))
        assert ex.wrong_color != env.target
        assert min_d < 0.03, (s, min_d)  # 실제로 잘못된 공 근처까지 간다


def test_wrong_approach_success_rate():
    env = BallSortEnv(render=False)
    ex = ScriptedExpert(env, seed=0, wrong_approach=True)
    wins = sum(run_episode(env, ex, s) for s in range(100, 120))
    assert wins >= 19, wins


def test_wrong_approach_falls_back_to_correction_without_other_ball():
    env = BallSortEnv(render=False)
    env.reset(seed=220, options={"scene": "two", "aside": "black", "target": "red"})
    # 남은 다른 공(sky blue)도 책상 아래로 치워 명령한 공만 남긴다
    adr = env._ball_adr["sky blue"]
    env.data.qpos[adr : adr + 3] = [0.25, 0.0, -0.72]
    ex = ScriptedExpert(env, seed=0, wrong_approach=True)
    ex.reset()
    assert ex.wrong_color is None and ex._off_mask.any()


def test_default_and_correction_outputs_unchanged():
    """기본 모드와 보정 모드 출력은 wrong_approach 추가 전과 같다 (변경 전 코드로 잰 값)."""
    env = BallSortEnv(render=False)
    for kw, n_steps, total in (({}, 439, 13729.367476287505), ({"correction": True}, 496, 13606.478812688401)):
        env.reset(seed=230)
        ex = ScriptedExpert(env, seed=5, **kw)
        ex.reset()
        a = np.array([np.concatenate(ex.act()) for _ in range(50)])
        assert ex.n_steps == n_steps
        assert a.sum() == pytest.approx(total, abs=1e-6)
