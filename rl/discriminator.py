"""IRL 보상 추론기: 상태 전이 (s, s')가 전문가 시범의 것인지 판별하고, 그 확률에서 보상을 만든다.

보상 r(s, s') = log D(s, s') ≤ 0. 에피소드는 성공(또는 시간 제한)으로만 끝나므로 음수 보상은
"전문가처럼 움직여 빨리 성공 상태로 가라"는 뜻이 되고, 오래 버티는 쪽으로 치우치지 않는다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LANG_SLICE = slice(960 + 512, 960 + 512 + 960)  # NoiseSteeredSmolVLA.features: 손목 960 | 위 512 | 명령문 960 | 상태 6


def mismatch_batch(expert, colors, rng, n):
    """명령어 바꿔치기 음성 표본: 전문가 전이는 그대로 두고 명령문 특징만 다른 색 명령의 것으로 바꾼다.

    colors: 전이마다 명령한 색 (len = 전이 수). 판별기가 '움직임과 명령의 색이 맞는가'를 보게 만든다.
    """
    i = rng.integers(0, len(colors), n)
    j = rng.integers(0, len(colors), 4 * n)
    o, o2 = expert["obs"][i].astype(np.float32), expert["obs2"][i].astype(np.float32)
    for k in range(n):
        cand = j[colors[j] != colors[i[k]]]
        src = expert["obs"][cand[k % len(cand)], LANG_SLICE]
        o[k, LANG_SLICE] = src
        o2[k, LANG_SLICE] = src
    return {"obs": torch.as_tensor(o), "obs2": torch.as_tensor(o2)}


def pair_colors(d):
    """expert_pairs.npz → 전이마다 명령한 색."""
    from policies.color_guard import command_color

    ep_colors = {e: command_color(str(t)) for e, t in zip(np.unique(d["episode"]), d["tasks"])}
    return np.array([ep_colors[e] for e in d["episode"]])


class Discriminator:
    def __init__(self, obs_dim, device="cuda", hidden=256, lr=1e-4, r1_coef=10.0, dropout=0.2):
        self.device = torch.device(device)
        self.r1_coef = r1_coef
        self.net = nn.Sequential(
            nn.LayerNorm(2 * obs_dim), nn.Linear(2 * obs_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1),
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    def _logit(self, o, o2):
        return self.net(torch.cat([o, o2], -1)).squeeze(-1)

    def update(self, expert, policy):
        """expert/policy: {"obs", "obs2"} 텐서 배치. 반환: 손실과 각 쪽 정답률."""
        e_o, e_o2 = expert["obs"].to(self.device), expert["obs2"].to(self.device)
        p_o, p_o2 = policy["obs"].to(self.device), policy["obs2"].to(self.device)
        self.net.train()
        e_in = torch.cat([e_o, e_o2], -1).requires_grad_(True)
        e_logit = self.net(e_in).squeeze(-1)
        p_logit = self._logit(p_o, p_o2)
        bce = F.binary_cross_entropy_with_logits(e_logit, torch.ones_like(e_logit)) + \
            F.binary_cross_entropy_with_logits(p_logit, torch.zeros_like(p_logit))
        grad, = torch.autograd.grad(e_logit.sum(), e_in, create_graph=True)
        r1 = grad.pow(2).sum(-1).mean()
        loss = bce + 0.5 * self.r1_coef * r1
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"d_loss": bce.item(), "d_r1": r1.item(), "d_acc_expert": (e_logit > 0).float().mean().item(),
                "d_acc_policy": (p_logit < 0).float().mean().item()}

    @torch.no_grad()
    def reward(self, o, o2):
        self.net.eval()
        return F.logsigmoid(self._logit(o.to(self.device), o2.to(self.device)))

    def save(self, path):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device)["net"])
