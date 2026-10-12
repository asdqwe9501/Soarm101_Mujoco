"""Minimal SAC with tanh-squashed actor, twin critics and automatic entropy tuning."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, dtype=np.float32):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype)
        self.act = np.zeros((capacity, act_dim), dtype)
        self.rew = np.zeros(capacity, np.float32)
        self.obs2 = np.zeros((capacity, obs_dim), dtype)
        self.done = np.zeros(capacity, np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, done):
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.obs2[i] = o2
        self.done[i] = float(done)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch, device="cpu"):
        idx = np.random.randint(0, self.size, size=batch)
        t = lambda x: torch.as_tensor(x[idx], device=device).float()
        return {"obs": t(self.obs), "act": t(self.act), "rew": t(self.rew),
                "obs2": t(self.obs2), "done": t(self.done)}


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.LayerNorm(inp), nn.Linear(inp, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))


class _Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.obs_norm = nn.LayerNorm(obs_dim)
        self.net = nn.Sequential(nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, o, a):
        return self.net(torch.cat([self.obs_norm(o), a], -1)).squeeze(-1)


class SAC:
    def __init__(self, obs_dim, act_dim, act_scale, device="cuda", hidden=512, lr=3e-4,
                 gamma=0.99, tau=0.005, target_entropy=None):
        self.act_dim, self.act_scale = act_dim, float(act_scale)
        self.device = torch.device(device)
        self.gamma, self.tau = gamma, tau
        self.target_entropy = -float(act_dim) if target_entropy is None else float(target_entropy)
        self.actor = _mlp(obs_dim, hidden, 2 * act_dim).to(self.device)
        self.q1 = _Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q2 = _Critic(obs_dim, act_dim, hidden).to(self.device)
        self.q1_t = copy.deepcopy(self.q1)
        self.q2_t = copy.deepcopy(self.q2)
        for p in list(self.q1_t.parameters()) + list(self.q2_t.parameters()):
            p.requires_grad_(False)
        self.log_alpha = torch.zeros((), device=self.device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def _pi(self, obs, deterministic=False):
        mean, log_std = self.actor(obs).chunk(2, -1)
        log_std = log_std.clamp(-5.0, 2.0)
        std = log_std.exp()
        u = mean if deterministic else mean + std * torch.randn_like(mean)
        th = torch.tanh(u)
        a = self.act_scale * th
        logp = (-0.5 * ((u - mean) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        # tanh and scale correction
        logp = logp - (2 * (np.log(2.0) - u - F.softplus(-2 * u))).sum(-1) - self.act_dim * np.log(self.act_scale)
        return a, logp

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self._pi(o, deterministic)
        return a[0].cpu().numpy()

    def update(self, batch):
        b = {k: v.to(self.device) for k, v in batch.items()}
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, logp2 = self._pi(b["obs2"])
            q_t = torch.min(self.q1_t(b["obs2"], a2), self.q2_t(b["obs2"], a2)) - alpha * logp2
            y = b["rew"] + self.gamma * (1 - b["done"]) * q_t
        q1, q2 = self.q1(b["obs"], b["act"]), self.q2(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        a, logp = self._pi(b["obs"])
        q_pi = torch.min(self.q1(b["obs"], a), self.q2(b["obs"], a))
        actor_loss = (alpha * logp - q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for net, tgt in ((self.q1, self.q1_t), (self.q2, self.q2_t)):
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "alpha": self.log_alpha.exp().item(), "q_mean": q1.mean().item(),
                "entropy": -logp.mean().item()}

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(), "q1_t": self.q1_t.state_dict(),
                    "q2_t": self.q2_t.state_dict(), "log_alpha": self.log_alpha.detach().cpu()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(d["actor"])
        self.q1.load_state_dict(d["q1"])
        self.q2.load_state_dict(d["q2"])
        self.q1_t.load_state_dict(d["q1_t"])
        self.q2_t.load_state_dict(d["q2_t"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))
