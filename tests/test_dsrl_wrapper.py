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
