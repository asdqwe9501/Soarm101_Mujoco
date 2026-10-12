import numpy as np
import torch

from rl.sac import SAC, ReplayBuffer


def test_buffer_roundtrip():
    b = ReplayBuffer(3, 2, 10)
    for i in range(12):
        b.add(np.full(3, i), np.zeros(2), float(i), np.full(3, i + 1), False)
    assert len(b) == 10
    s = b.sample(4)
    assert s["obs"].shape == (4, 3) and s["act"].shape == (4, 2)


def test_learns_contextual_bandit():
    """특징 부호에 따라 행동 부호를 맞히면 보상 1. 색 선택을 흉내 낸 문제."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    sac = SAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=64, gamma=0.0)
    buf = ReplayBuffer(2, 1, 5000)
    for t in range(3000):
        s = rng.choice([-1.0, 1.0])
        o = np.array([s, 1.0], np.float32)
        a = sac.act(o)
        r = 1.0 if np.sign(a[0]) == s else 0.0
        buf.add(o, a, r, o, True)
        if len(buf) > 256:
            sac.update(buf.sample(256))
    hits = [np.sign(sac.act(np.array([s, 1.0], np.float32), deterministic=True)[0]) == s for s in (-1.0, 1.0)]
    assert all(hits)


def test_save_load(tmp_path):
    sac = SAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=16)
    o = np.array([0.3, -0.2], np.float32)
    a = sac.act(o, deterministic=True)
    sac.save(tmp_path / "s.pt")
    sac2 = SAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=16)
    sac2.load(tmp_path / "s.pt")
    np.testing.assert_allclose(sac2.act(o, deterministic=True), a, atol=1e-6)
