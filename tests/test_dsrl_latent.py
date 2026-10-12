import numpy as np

from policies.dsrl_constants import ACT_DIM, CHUNK, LATENT_DIM, STEER_DIM
from policies.dsrl_latent import BASIS, SteerLatent


def test_basis_orthonormal_and_layout():
    assert BASIS.shape == (LATENT_DIM, STEER_DIM) == (300, 12)
    assert np.allclose(BASIS.T @ BASIS, np.eye(STEER_DIM), atol=1e-6)
    for col in range(STEER_DIM):
        j = col // 2
        nz = np.flatnonzero(np.abs(BASIS[:, col]) > 1e-12)
        assert set(nz) <= {t * ACT_DIM + j for t in range(CHUNK)}
    # k=0: 상수, k=1: 천천히 변하는 코사인
    k0 = BASIS[np.arange(CHUNK) * ACT_DIM + 2, 4]
    assert np.allclose(k0, 1 / np.sqrt(CHUNK))
    k1 = BASIS[np.arange(CHUNK) * ACT_DIM + 2, 5]
    assert k1[0] > 0 > k1[-1] and np.all(np.diff(k1) < 0)


def test_projection_recovers_z():
    sl = SteerLatent(seed=0)
    z = np.random.default_rng(1).uniform(-2, 2, STEER_DIM).astype(np.float32)
    lat = sl.to_latent(z)
    assert lat.shape == (LATENT_DIM,) and lat.dtype == np.float32
    assert np.allclose(BASIS.T @ lat, z, atol=1e-5)


def test_fresh_complement_noise_each_call():
    sl = SteerLatent(seed=0)
    z = np.zeros(STEER_DIM, np.float32)
    assert not np.allclose(sl.to_latent(z), sl.to_latent(z))


def test_standard_normal_z_gives_standard_normal_latent():
    sl = SteerLatent(seed=0)
    rng = np.random.default_rng(2)
    lat = np.stack([sl.to_latent(rng.standard_normal(STEER_DIM)) for _ in range(4000)])
    assert np.abs(lat.mean(0)).max() < 0.1
    assert np.abs(lat.var(0) - 1).max() < 0.1
    cov = np.cov(lat, rowvar=False)
    off = cov[~np.eye(LATENT_DIM, dtype=bool)]
    assert np.abs(off).max() < 0.1
