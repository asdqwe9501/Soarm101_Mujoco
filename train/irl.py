"""IRL 학습 루프 (Plan 8): 판별기가 시범에서 보상을 추론하고, KL 정규화 노이즈 정책이 그 보상으로 학습한다.

python -m train.irl --checkpoint <sft 복사본> --expert data/irl/expert_pairs.npz --out outputs/irl/<run> --episodes N

- 사용자 정의 단계별 보상은 학습에 쓰지 않는다. 기록(metrics.csv)과 평가에만 쓴다.
- 에피소드 종료: env 성공 또는 900스텝(ChunkEnv staged_termination=False).
- actor-critic 표본의 보상은 매번 현재 판별기로 다시 계산한다: r = log D(s, s').
"""
import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from policies.dsrl import NoiseSteeredSmolVLA
from policies.dsrl_constants import EXEC_STEPS, LATENT_SCALE, STEER_DIM
from rl.chunk_env import ChunkEnv
from rl.discriminator import Discriminator
from rl.kl_ac import KLRegularizedAC
from rl.sac import ReplayBuffer
from sim.env import COLORS, BallSortEnv
from train.dsrl import episode_options

SEED_BASE = 800_000  # DSRL(100000~), 평가(10000~), 시범(700000~)과 겹치지 않음
WINDOW = 50
REWARD_MIN = -10.0
FIELDS = ["episode", "env_steps_total", "wall_time_s", "mean_irl_return", "mean_staged_return",
          "first_attempt_target_rate", "first_attempt_other_rate", "miss_or_timeout_rate", "success_rate",
          "critic_loss", "actor_loss", "kl", "d_loss", "d_acc_expert", "d_acc_policy"]


def to_batch(d, idx):
    return {"obs": torch.as_tensor(d["obs"][idx]).float(), "obs2": torch.as_tensor(d["obs2"][idx]).float()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expert", required=True, help="tools/irl_expert_pairs.py 출력")
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--init-negatives", default=None, help="판별기 사전학습용 정책 쌍(예: DSRL v3 buffer.npz)")
    ap.add_argument("--d-pretrain", type=int, default=3000)
    ap.add_argument("--d-steps", type=int, default=20, help="에피소드마다 판별기 갱신 횟수")
    ap.add_argument("--d-recent", type=int, default=5000, help="판별기 음성 표본: 리플레이 버퍼의 최근 전이 수")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--critic-warmup", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-buffer", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # ReplayBuffer.sample이 전역 난수를 쓴다
    rng_d = np.random.default_rng(args.seed)

    steer = NoiseSteeredSmolVLA(args.checkpoint, device="cuda", seed=args.seed)
    ce = ChunkEnv(BallSortEnv(render=True), steer, EXEC_STEPS, STEER_DIM, LATENT_SCALE, seed=args.seed,
                  staged_termination=False)
    agent = KLRegularizedAC(steer.feat_dim, STEER_DIM, LATENT_SCALE, device="cuda", beta=args.beta,
                            critic_warmup=args.critic_warmup)
    disc = Discriminator(steer.feat_dim, device="cuda")
    buf = ReplayBuffer(steer.feat_dim, STEER_DIM, 100_000, dtype=np.float16)
    expert = dict(np.load(args.expert))
    print(f"expert pairs: {len(expert['obs'])}", flush=True)

    d_stats = {}
    if args.init_negatives:
        neg = dict(np.load(args.init_negatives))
        for i in range(args.d_pretrain):
            d_stats = disc.update(to_batch(expert, rng_d.integers(0, len(expert["obs"]), args.batch)),
                                  to_batch(neg, rng_d.integers(0, len(neg["obs"]), args.batch)))
            if (i + 1) % 500 == 0:
                print(f"d pretrain {i + 1}/{args.d_pretrain}: {d_stats}", flush=True)
        del neg

    def agent_update():
        b = buf.sample(args.batch)
        b["rew"] = disc.reward(b["obs"], b["obs2"]).clamp(min=REWARD_MIN).cpu()
        return agent.update(b)

    def disc_update():
        lo = max(0, len(buf) - args.d_recent) if len(buf) < buf.capacity else 0
        idx = rng_d.integers(lo, len(buf), args.batch)
        pol = {"obs": torch.as_tensor(buf.obs[idx]).float(), "obs2": torch.as_tensor(buf.obs2[idx]).float()}
        return disc.update(to_batch(expert, rng_d.integers(0, len(expert["obs"]), args.batch)), pol)

    metrics_path = out / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.DictWriter(f, FIELDS).writeheader()
    hist = deque(maxlen=WINDOW)
    last_upd, env_steps_total, t0 = {}, 0, time.time()

    for ep in range(1, args.episodes + 1):
        rng = np.random.default_rng(SEED_BASE + ep)
        color = COLORS[rng.integers(len(COLORS))]
        opts = episode_options(color, "train", True, np.random.default_rng([SEED_BASE + ep, 1]))
        o, _ = ce.reset(seed=SEED_BASE + ep, options=opts)
        staged_ret, irl_ret, t_ep = 0.0, 0.0, time.time()
        while True:
            o2, r_staged, term, trunc, info = ce.step(agent.act(o))
            buf.add(o, ce.last_latent, 0.0, o2, term)  # 보상은 표본을 뽑을 때 판별기로 계산
            irl_ret += float(disc.reward(torch.as_tensor(o)[None], torch.as_tensor(o2)[None]).clamp(min=REWARD_MIN))
            o, staged_ret = o2, staged_ret + r_staged
            if len(buf) >= args.min_buffer:
                last_upd = agent_update()
            if term or trunc:
                break
        if len(buf) >= args.batch:
            for _ in range(args.d_steps):
                d_stats = disc_update()
        env_steps_total += info["env_steps"]
        hist.append({"irl": irl_ret, "staged": staged_ret, "first": info.get("first_attempt"),
                     "reason": info.get("end_reason"), "success": info["env_success"]})
        print(f"ep {ep} [{color}] {opts['command']!r} irl {irl_ret:+.2f} staged {staged_ret:+.1f} "
              f"first {info.get('first_attempt')} success {info['env_success']} steps {info['env_steps']} "
              f"time {time.time() - t_ep:.1f}s d_acc {d_stats.get('d_acc_expert', float('nan')):.2f}/"
              f"{d_stats.get('d_acc_policy', float('nan')):.2f} kl {last_upd.get('kl', float('nan')):.3f}", flush=True)
        if ep % args.save_every == 0 or ep == args.episodes:
            agent.save(out / "last.pt")
            agent.save(out / f"agent_{ep}.pt")
            disc.save(out / f"disc_{ep}.pt")
            h = list(hist)
            m = lambda f: float(np.mean([f(e) for e in h]))
            row = {"episode": ep, "env_steps_total": env_steps_total, "wall_time_s": round(time.time() - t0, 1),
                   "mean_irl_return": m(lambda e: e["irl"]), "mean_staged_return": m(lambda e: e["staged"]),
                   "first_attempt_target_rate": m(lambda e: e["first"] == "target"),
                   "first_attempt_other_rate": m(lambda e: e["first"] == "other"),
                   "miss_or_timeout_rate": m(lambda e: e["reason"] in ("miss_attempt", "attempt_timeout")),
                   "success_rate": m(lambda e: e["success"]),
                   **{k: last_upd.get(k, float("nan")) for k in ("critic_loss", "actor_loss", "kl")},
                   **{k: d_stats.get(k, float("nan")) for k in ("d_loss", "d_acc_expert", "d_acc_policy")}}
            with open(metrics_path, "a", newline="") as f:
                csv.DictWriter(f, FIELDS).writerow(row)


if __name__ == "__main__":
    main()
