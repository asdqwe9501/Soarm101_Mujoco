import torch

from rl.discriminator import Discriminator


def _batch(n, shift):
    o = torch.randn(n, 4)
    o2 = o.clone()
    o2[:, 0] += shift  # 전문가는 첫 특징을 +1만큼 옮기고, 정책은 -1만큼 옮긴다
    return {"obs": o, "obs2": o2}


def test_learns_expert_transition_and_reward_is_nonpositive(tmp_path):
    torch.manual_seed(0)
    d = Discriminator(obs_dim=4, device="cpu", hidden=32, lr=1e-3, r1_coef=1.0, dropout=0.0)
    for _ in range(400):
        stats = d.update(_batch(64, 1.0), _batch(64, -1.0))
    assert stats["d_acc_expert"] > 0.9 and stats["d_acc_policy"] > 0.9
    e, p = _batch(100, 1.0), _batch(100, -1.0)
    r_e, r_p = d.reward(e["obs"], e["obs2"]), d.reward(p["obs"], p["obs2"])
    assert torch.all(r_e <= 0) and torch.all(r_p <= 0) and r_e.mean() > r_p.mean() + 0.5

    d.save(tmp_path / "d.pt")
    d2 = Discriminator(obs_dim=4, device="cpu", hidden=32, dropout=0.0)
    d2.load(tmp_path / "d.pt")
    assert torch.allclose(d2.reward(e["obs"], e["obs2"]), r_e)
