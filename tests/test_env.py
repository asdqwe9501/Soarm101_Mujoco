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
    away = np.array(HOME_NORM)
    away[1] += 20  # 제자리에서 벗어난 자세라 아직 성공이 아니다
    for _ in range(15):
        env.step(away)
    assert not env.at_home()
    _teleport(env, info["target"], [b[0], b[1], 0.025])
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
