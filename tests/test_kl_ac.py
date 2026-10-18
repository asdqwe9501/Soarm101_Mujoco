import numpy as np
import torch

from rl.kl_ac import KLRegularizedAC
from rl.sac import ReplayBuffer


def test_initial_policy_is_prior():
    ac = KLRegularizedAC(obs_dim=4, act_dim=3, act_scale=2.0, device="cpu", hidden=16)
    o = torch.randn(8, 4)
    mean, log_std = ac._dist(o)
    assert torch.all(mean == 0) and torch.all(log_std == 0)
    assert torch.allclose(ac._kl(mean, log_std), torch.zeros(8))


def test_actor_frozen_during_critic_warmup():
    torch.manual_seed(0)
    ac = KLRegularizedAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=16, critic_warmup=5)
    buf = ReplayBuffer(2, 1, 100)
    for _ in range(50):
        buf.add(np.random.randn(2), np.random.randn(1), 1.0, np.random.randn(2), True)
    before = [p.clone() for p in ac.actor.parameters()]
    for _ in range(5):
        ac.update(buf.sample(16))
    assert all(torch.equal(a, b) for a, b in zip(before, ac.actor.parameters()))
    ac.update(buf.sample(16))
    assert not all(torch.equal(a, b) for a, b in zip(before, ac.actor.parameters()))


def test_learns_contextual_bandit():
    """특징 부호에 따라 행동 부호를 맞히면 보상 1. 색 선택을 흉내 낸 문제."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ac = KLRegularizedAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=64, gamma=0.0, beta=0.1)
    buf = ReplayBuffer(2, 1, 5000)
    for _ in range(3000):
        s = rng.choice([-1.0, 1.0])
        o = np.array([s, 1.0], np.float32)
        a = ac.act(o)
        buf.add(o, a, 1.0 if np.sign(a[0]) == s else 0.0, o, True)
        if len(buf) > 256:
            ac.update(buf.sample(256))
    hits = [np.mean([np.sign(ac.act(np.array([s, 1.0], np.float32))[0]) == s for _ in range(100)]) for s in (-1.0, 1.0)]
    assert min(hits) > 0.9


def test_save_load(tmp_path):
    ac = KLRegularizedAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=16)
    torch.nn.init.normal_(ac.actor[-1].weight)
    o = np.array([0.3, -0.2], np.float32)
    a = ac.act(o, deterministic=True)
    ac.save(tmp_path / "k.pt")
    assert torch.load(tmp_path / "k.pt")["kind"] == "kl"
    ac2 = KLRegularizedAC(obs_dim=2, act_dim=1, act_scale=2.0, device="cpu", hidden=16)
    ac2.load(tmp_path / "k.pt")
    np.testing.assert_allclose(ac2.act(o, deterministic=True), a, atol=1e-6)
