"""KL 정규화 actor-critic: 정책이 기반 노이즈 분포 N(0, I)에서 멀어지는 만큼 β·KL 벌점을 받는다.

SAC의 엔트로피 보너스 대신 사전분포 KL을 쓴다. 최적 정책은 π(z|s) ∝ N(z; 0, I)·exp(Q(s, z)/β)라서
Q 차이가 β보다 충분히 크지 않으면 기반 분포에 머문다. actor 마지막 층을 0으로 초기화해 시작 시 π = N(0, I)다.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F

from rl.sac import _Critic, _mlp


class KLRegularizedAC:
    kind = "kl"

    def __init__(self, obs_dim, act_dim, act_scale, device="cuda", hidden=512, actor_lr=1e-4, critic_lr=3e-4,
                 gamma=0.99, tau=0.005, beta=1.0, critic_warmup=0):
        self.act_dim, self.act_scale = act_dim, float(act_scale)
        self.device = torch.device(device)
        self.gamma, self.tau, self.beta, self.critic_warmup = gamma, tau, float(beta), int(critic_warmup)
        self.updates = 0
        self.actor = _mlp(obs_dim, hidden, 2 * act_dim).to(self.device)
        torch.nn.init.zeros_(self.actor[-1].weight)
        torch.nn.init.zeros_(self.actor[-1].bias)
        self.q1 = _Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q2 = _Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)

    def _dist(self, obs):
        mean, log_std = self.actor(obs).chunk(2, -1)
        return mean, log_std.clamp(-5.0, 1.0)

    @staticmethod
    def _kl(mean, log_std):
        """KL(N(mean, std²) ‖ N(0, I)), 차원 합."""
        return 0.5 * ((2 * log_std).exp() + mean ** 2 - 1 - 2 * log_std).sum(-1)

    def _sample(self, mean, log_std):
        z = mean + log_std.exp() * torch.randn_like(mean)
        return z.clamp(-self.act_scale, self.act_scale)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, log_std = self._dist(o)
        a = mean.clamp(-self.act_scale, self.act_scale) if deterministic else self._sample(mean, log_std)
        return a[0].cpu().numpy()

    def update(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            m2, ls2 = self._dist(b["obs2"])
            a2 = self._sample(m2, ls2)
            q_t = torch.min(self.q1_t(b["obs2"], a2), self.q2_t(b["obs2"], a2)) - self.beta * self._kl(m2, ls2)
            y = b["rew"] + self.gamma * (1 - b["done"]) * q_t
        q1, q2 = self.q1(b["obs"], b["act"]), self.q2(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.updates += 1

        mean, log_std = self._dist(b["obs"])
        kl = self._kl(mean, log_std)
        out = {"critic_loss": critic_loss.item(), "q_mean": q1.mean().item(), "kl": kl.mean().item(),
               "actor_loss": float("nan")}
        if self.updates > self.critic_warmup:
            a = self._sample(mean, log_std)
            q_pi = torch.min(self.q1(b["obs"], a), self.q2(b["obs"], a))
            actor_loss = (self.beta * kl - q_pi).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            out["actor_loss"] = actor_loss.item()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return out

    def save(self, path):
        torch.save({"kind": self.kind, "actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(), "q1_t": self.q1_t.state_dict(), "q2_t": self.q2_t.state_dict(),
                    "updates": self.updates, "beta": self.beta}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.q1.load_state_dict(d["q1"])
        self.q2.load_state_dict(d["q2"])
        self.q1_t.load_state_dict(d["q1_t"])
        self.q2_t.load_state_dict(d["q2_t"])
        self.updates = d["updates"]
