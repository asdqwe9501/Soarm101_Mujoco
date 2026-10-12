import numpy as np
import pytest

from calib import SIM_HOME_NORM as HOME_NORM  # 시뮬레이션 제자리 자세
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


def test_keep_scene_previous_ball_does_not_block_success(env):
    env.reset(seed=9, options={"target": "red"})
    b = env.bin_pos()
    _teleport(env, "red", [b[0] + 0.021, b[1], 0.025])
    env.step(HOME_NORM)
    env.reset(options={"keep_scene": True, "target": "black"})
    _teleport(env, "black", [b[0] - 0.021, b[1], 0.025])
    _, reward, _, _, info = env.step(HOME_NORM)
    assert info["target_in_bin"] and not info["wrong_in_bin"] and info["success"]
    assert reward == pytest.approx(2.0)


def test_color_positions_are_balanced(env):
    """공 위치를 색에 고정된 순서로 배정하면 특정 색이 한쪽에 몰린다. 색마다 좌우 분포가 비슷해야 한다."""
    ys = {c: [] for c in COLORS}
    for s in range(2000):
        env.reset(seed=s)
        for c in COLORS:
            ys[c].append(env.ball_pos(c)[1])
    means = [np.mean(v) for v in ys.values()]
    right = [np.mean(np.array(v) < 0.02) for v in ys.values()]
    # 고정 순서 배정 시: 평균 y 차이 약 0.013, 오른쪽 비율 차이 약 0.057
    assert max(means) - min(means) < 0.006, means
    assert max(right) - min(right) < 0.03, right


# ---------- Plan 5: 시각 무작위화와 장면 구성 ----------
def _same_image(a, b):
    # GPU 렌더링은 부하에 따라 극소수 픽셀이 1 차이 날 수 있다
    d = np.abs(a.astype(int) - b.astype(int))
    assert d.max() <= 2 and (d > 0).mean() < 1e-4, (d.max(), (d > 0).mean())


def _visual_state(e):
    t = e.model.geom("table").id
    return (
        e.model.light_diffuse.copy(),
        e.model.vis.headlight.diffuse.copy(),
        e.model.geom_rgba[t].copy(),
        e.model.geom_rgba[e.model.geom("floor").id].copy(),
    )


@pytest.fixture
def renv():
    e = BallSortEnv(render=True)
    yield e
    e.close()


def test_randomize_same_seed_same_render_and_different_seed_differs(renv):
    o1, _ = renv.reset(seed=11, options={"randomize": True})
    v1 = _visual_state(renv)
    o2, _ = renv.reset(seed=11, options={"randomize": True})
    _same_image(o1["observation.images.front"], o2["observation.images.front"])
    _same_image(o1["observation.images.top"], o2["observation.images.top"])
    renv.reset(seed=12, options={"randomize": True})
    v3 = _visual_state(renv)
    assert not np.allclose(v1[0], v3[0])  # 조명
    assert not np.allclose(v1[2], v3[2])  # 책상 색


def test_randomize_off_restores_defaults(renv):
    o0, _ = renv.reset(seed=13)
    v0 = _visual_state(renv)
    balls0 = renv.model.geom_rgba[[renv.model.geom("ball_" + c.replace(" ", "_")).id for c in COLORS]].copy()
    renv.reset(seed=14, options={"randomize": True})
    o1, _ = renv.reset(seed=13)
    for a, b in zip(v0, _visual_state(renv)):
        np.testing.assert_array_equal(a, b)
    balls1 = renv.model.geom_rgba[[renv.model.geom("ball_" + c.replace(" ", "_")).id for c in COLORS]]
    np.testing.assert_array_equal(balls0, balls1)
    _same_image(o0["observation.images.front"], o1["observation.images.front"])


def test_randomize_ball_colors_change_little(env):
    ids = [env.model.geom("ball_" + c.replace(" ", "_")).id for c in COLORS]
    env.reset(seed=15)
    base = env.model.geom_rgba[ids].copy()
    for s in range(20):
        env.reset(seed=s, options={"randomize": True})
        assert np.all(np.abs(env.model.geom_rgba[ids] - base) <= 0.05 + 1e-9)


def test_randomize_does_not_change_layout(env):
    env.reset(seed=16)
    p0 = {c: env.ball_pos(c) for c in COLORS}
    env.reset(seed=16, options={"randomize": True})
    for c in COLORS:
        np.testing.assert_allclose(env.ball_pos(c), p0[c], atol=1e-9)


def _on_table(env, c):
    p = env.ball_pos(c)
    return 0.0 <= p[0] <= 0.4 and -0.3 <= p[1] <= 0.3 and abs(p[2] - 0.02) < 0.005 and not env.ball_in_bin(c)


@pytest.mark.parametrize("scene", ["two", "prebinned"])
def test_reduced_scenes_keep_target_on_table(env, scene):
    for s in range(20):
        _, info = env.reset(seed=s, options={"scene": scene})
        t = info["target"]
        assert _on_table(env, t), (s, t)
        others = [c for c in COLORS if c != t]
        n_table = sum(_on_table(env, c) for c in others)
        assert n_table == 1, (s, scene)
        aside = [c for c in others if not _on_table(env, c)][0]
        if scene == "two":
            assert env.ball_pos(aside)[2] < -0.5  # 책상 아래
        else:
            assert env.ball_in_bin(aside)


def test_scene_aside_option(env):
    _, info = env.reset(seed=17, options={"scene": "two", "aside": "red", "target": "black"})
    assert env.ball_pos("red")[2] < -0.5 and info["target"] == "black"
    with pytest.raises(ValueError):
        env.reset(seed=17, options={"scene": "two", "aside": "red", "target": "red"})


def test_prebinned_ball_does_not_block_success(env):
    _, info = env.reset(seed=18, options={"scene": "prebinned", "aside": "sky blue", "target": "red"})
    assert not info["wrong_in_bin"]
    b = env.bin_pos()
    sb = env.ball_pos("sky blue")
    # 미리 들어 있던 공의 반대쪽에 명령한 공을 넣는다
    d = (b[:2] - sb[:2]) / max(np.hypot(*(b[:2] - sb[:2])), 1e-6)
    _teleport(env, "red", [*(b[:2] + 0.021 * d), 0.025])
    _, reward, terminated, _, info = env.step(HOME_NORM)
    assert info["target_in_bin"] and not info["wrong_in_bin"] and info["success"] and terminated
    assert reward == pytest.approx(2.0)


def test_two_scene_aside_ball_does_not_block_success(env):
    _, info = env.reset(seed=19, options={"scene": "two", "aside": "black", "target": "red"})
    b = env.bin_pos()
    _teleport(env, "red", [b[0], b[1], 0.025])
    _, reward, terminated, _, info = env.step(HOME_NORM)
    assert info["success"] and terminated and reward == pytest.approx(2.0)
