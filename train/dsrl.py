"""DSRL 학습 루프: SAC가 고정된 SmolVLA의 flow-matching 노이즈(잠재 행동)를 고른다.

python -m train.dsrl --checkpoint <sft> --out outputs/dsrl/<run> --episodes N --warmup-episodes W [--resume]
"""

import argparse
import csv
import os
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from policies.dsrl import NoiseSteeredSmolVLA
from policies.dsrl_constants import EXEC_STEPS, LATENT_SCALE, STEER_DIM
from rl.chunk_env import ChunkEnv
from rl.sac import SAC, ReplayBuffer
from sim.commands import command_for_tier, sample_command
from sim.env import COLORS, BallSortEnv

SEED_BASE = 100_000  # 평가 seed(10000~)와 겹치지 않음
WINDOW = 50  # metrics.csv 이동 평균 창
BUFFER_EVERY = 50  # 버퍼 파일은 크므로 50 에피소드마다(그리고 종료 시)만 저장
FIELDS = ["episode", "env_steps_total", "wall_time_s", "mean_return", "first_attempt_target_rate",
          "first_attempt_other_rate", "miss_or_timeout_rate", "stage>=1", "stage>=2", "stage>=3",
          "success_rate", "alpha", "critic_loss", "actor_loss"]


def episode_options(color, commands, randomize, rng):
    """env.reset options: canonical이면 기존 고정 문장, basic/train이면 명령문을 넣는다."""
    opts = {"target": color}
    if commands == "basic":
        opts["command"] = command_for_tier(color, "basic", rng)
    elif commands == "train":
        opts["command"] = sample_command(color, rng, split="train")
    if randomize:
        opts["randomize"] = True
    return opts


def save_buffer(buf, path):
    tmp = str(path) + ".tmp.npz"
    n = buf.size
    np.savez(tmp, obs=buf.obs[:n], act=buf.act[:n], rew=buf.rew[:n], obs2=buf.obs2[:n],
             done=buf.done[:n], ptr=buf.ptr)
    os.replace(tmp, path)


def load_buffer(buf, path):
    d = np.load(path)
    n = len(d["rew"])
    for k in ["obs", "act", "rew", "obs2", "done"]:
        getattr(buf, k)[:n] = d[k]
    buf.size, buf.ptr = n, int(d["ptr"])


def read_progress(path):
    """metrics.csv의 마지막 행 → (완료 에피소드 수, 누적 env 스텝, 경과 시간)."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return 0, 0, 0.0
    r = rows[-1]
    return int(r["episode"]), int(r["env_steps_total"]), float(r["wall_time_s"])


def summarize(episode, env_steps, wall, hist, last_upd):
    h = list(hist)
    m = lambda f: float(np.mean([f(e) for e in h])) if h else 0.0
    return {
        "episode": episode, "env_steps_total": env_steps, "wall_time_s": round(wall, 1),
        "mean_return": m(lambda e: e["ret"]),
        "first_attempt_target_rate": m(lambda e: e["first_attempt"] == "target"),
        "first_attempt_other_rate": m(lambda e: e["first_attempt"] == "other"),
        "miss_or_timeout_rate": m(lambda e: e["end_reason"] in ("miss_attempt", "attempt_timeout")),
        "stage>=1": m(lambda e: e["stage"] >= 1), "stage>=2": m(lambda e: e["stage"] >= 2),
        "stage>=3": m(lambda e: e["stage"] >= 3), "success_rate": m(lambda e: e["end_reason"] == "success"),
        "alpha": last_upd.get("alpha", float("nan")), "critic_loss": last_upd.get("critic_loss", float("nan")),
        "actor_loss": last_upd.get("actor_loss", float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, required=True, help="총 에피소드 수 (재개 시 이어서 이 수까지)")
    ap.add_argument("--warmup-episodes", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--min-buffer", type=int, default=1000)
    ap.add_argument("--capacity", type=int, default=100_000)
    ap.add_argument("--max-env-steps", type=int, default=None, help="스모크용: 이 env 스텝에서 truncated 처리")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--commands", choices=["canonical", "basic", "train"], default="canonical",
                    help="명령문: canonical(예전 고정 문장), basic(Grab the X ball.), train(학습 배분대로)")
    ap.add_argument("--randomize", action="store_true", help="조명/배경 무작위화")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)  # ReplayBuffer.sample이 전역 난수를 쓴다

    steer = NoiseSteeredSmolVLA(args.checkpoint, device="cuda", seed=args.seed)
    ce = ChunkEnv(BallSortEnv(render=True), steer, EXEC_STEPS, STEER_DIM, LATENT_SCALE, seed=args.seed)
    sac = SAC(steer.feat_dim, STEER_DIM, LATENT_SCALE, device="cuda")
    buf = ReplayBuffer(steer.feat_dim, STEER_DIM, args.capacity, dtype=np.float16)

    metrics_path = out / "metrics.csv"
    start_ep, env_steps_total, wall0 = 0, 0, 0.0
    if args.resume:
        sac.load(out / "last.pt")
        if (out / "buffer.npz").exists():
            load_buffer(buf, out / "buffer.npz")
        start_ep, env_steps_total, wall0 = read_progress(metrics_path)
        print(f"resume: episode {start_ep}, buffer {len(buf)}, env steps {env_steps_total}", flush=True)
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    hist = deque(maxlen=WINDOW)
    last_upd = {}
    t_start = time.time() - wall0
    ep = start_ep

    def checkpoint(save_buf):
        sac.save(out / "last.pt")
        sac.save(out / f"sac_{ep}.pt")
        row = summarize(ep, env_steps_total, time.time() - t_start, hist, last_upd)
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        if save_buf:
            save_buffer(buf, out / "buffer.npz")

    last_saved = ep
    try:
        while ep < args.episodes:
            rng = np.random.default_rng(SEED_BASE + ep)
            color = COLORS[rng.integers(len(COLORS))]
            warm = ep < args.warmup_episodes
            t_ep = time.time()
            opts = episode_options(color, args.commands, args.randomize, np.random.default_rng([SEED_BASE + ep, 1]))
            o, _ = ce.reset(seed=SEED_BASE + ep, options=opts)
            ret, n_rl = 0.0, 0
            while True:
                a = (np.clip(rng.standard_normal(STEER_DIM), -LATENT_SCALE, LATENT_SCALE).astype(np.float32)
                     if warm else sac.act(o))
                o2, r, term, trunc, info = ce.step(a)
                if args.max_env_steps is not None and info["env_steps"] >= args.max_env_steps and not term:
                    trunc = True
                buf.add(o, ce.last_latent, r, o2, term)  # truncated는 부트스트랩한다
                o, ret, n_rl = o2, ret + r, n_rl + 1
                if not warm and len(buf) >= args.min_buffer:
                    last_upd = sac.update(buf.sample(args.batch))
                if term or trunc:
                    break
            ep += 1
            env_steps_total += info["env_steps"]
            reason = info["end_reason"] or "truncated"
            hist.append({"ret": ret, "first_attempt": info["first_attempt"], "stage": info["stage"],
                         "end_reason": reason})
            dt = time.time() - t_ep
            print(f"ep {ep} [{color}{' warm' if warm else ''}] {opts.get('command', '')!r} ret {ret:+.1f} {reason} stage {info['stage']} "
                  f"steps {info['env_steps']} rl {n_rl} time {dt:.1f}s ({dt / n_rl:.2f}s/rl) buf {len(buf)}",
                  flush=True)
            if ep % args.save_every == 0:
                checkpoint(ep % BUFFER_EVERY == 0)
                last_saved = ep
    finally:
        if ep > last_saved:
            checkpoint(True)
        elif ep > start_ep:
            save_buffer(buf, out / "buffer.npz")
        reasons = Counter(e["end_reason"] for e in hist)
        print(f"done: episode {ep}, end_reason (last {len(hist)}): {dict(reasons)}", flush=True)


if __name__ == "__main__":
    main()
