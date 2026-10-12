import numpy as np
import pytest
import torch

from sim.env import BallSortEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CK = "outputs/train/sft_v3/checkpoints/last/pretrained_model"


@pytest.fixture(scope="module")
def setup():
    from policies.dsrl import NoiseSteeredSmolVLA

    env = BallSortEnv(render=True)
    steer = NoiseSteeredSmolVLA(CK, seed=0)
    obs, _ = env.reset(seed=0, options={"target": "red"})
    steer.reset(env)
    yield env, steer, obs
    env.close()


def test_feature_shape_and_finite(setup):
    env, steer, obs = setup
    f = steer.features(obs)
    assert f.shape == (steer.feat_dim,) and f.dtype == np.float32 and np.all(np.isfinite(f))


def test_features_depend_on_command(setup):
    env, steer, obs = setup
    f_red = steer.features(obs)
    steer.task = "grab the black ball and put it in the box"
    f_black = steer.features(obs)
    steer.reset(env)
    assert np.abs(f_red - f_black).max() > 1e-4


def test_chunk_shape_and_latent_effect(setup):
    from policies.dsrl import LATENT_DIM

    env, steer, obs = setup
    zero = steer.chunk(obs, np.zeros(LATENT_DIM, np.float32))
    plus = steer.chunk(obs, np.full(LATENT_DIM, 1.5, np.float32))
    assert zero.shape == (50, 6) and zero.dtype == np.float32
    assert np.abs(zero - plus).mean() > 0.5  # 노이즈가 실제로 행동을 바꾼다


def test_gaussian_latent_matches_base_policy_scale(setup):
    from policies.dsrl import LATENT_DIM

    env, steer, obs = setup
    rng = np.random.default_rng(0)
    ends = np.stack([steer.chunk(obs, rng.standard_normal(LATENT_DIM).astype(np.float32))[-1] for _ in range(4)])
    assert np.all(np.isfinite(ends)) and np.all(np.abs(ends[:, :5]) < 100)


def test_blocks_standardized(setup):
    env, steer, obs = setup
    f = steer.features(obs)
    assert steer.feat_dim == 960 + 512 + 960 + 6
    for name in ("wrist", "top", "lang"):
        b = f[steer.blocks[name]]
        assert abs(b.mean()) < 1e-3 and abs(b.std() - 1.0) < 1e-2, name
    state = steer.adapter.pre(steer.adapter._batch(obs))["observation.state"][0, :6].float().cpu().numpy()
    np.testing.assert_allclose(f[steer.blocks["state"]], state, atol=1e-6)


def test_language_block_separates_similar_colors(setup):
    env, steer, obs = setup
    steer.task = "grab the red ball and put it in the box"
    f_red = steer.features(obs)[steer.blocks["lang"]]
    steer.task = "grab the sky blue ball and put it in the box"
    f_sky = steer.features(obs)[steer.blocks["lang"]]
    steer.reset(env)
    assert np.linalg.norm(f_red - f_sky) > 0.1 * np.linalg.norm(f_red)


def test_overhead_block_depends_on_layout(setup):
    env, steer, obs = setup
    f0 = steer.features(obs)[steer.blocks["top"]]
    obs1, _ = env.reset(seed=1, options={"target": "red"})
    steer.reset(env)
    f1 = steer.features(obs1)[steer.blocks["top"]]
    assert np.linalg.norm(f0 - f1) > 0.05 * np.linalg.norm(f0)
