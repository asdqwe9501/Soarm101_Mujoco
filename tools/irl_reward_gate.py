"""IRL 관문: 판별기가 찾아낸 보상이 '명령한 색 공을 넣는 것'을 높게 치는지 확인한다.

판별기를 전문가 쌍(양성)과 정책 쌍(음성, 예: DSRL v3 버퍼)으로 학습한다.
그다음 학습에 쓰지 않은 같은 장면의 전문가 시범과 잘못된 색 시범에 대해 에피소드 평균 보상을 비교한다(AUC).

python tools/irl_reward_gate.py --expert data/irl/expert_pairs.npz --policy-buffer outputs/dsrl/v3/buffer.npz \
    --gate-expert data/irl/gate_expert.npz --gate-wrong data/irl/gate_wrong.npz --out docs/results/irl_gate.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.discriminator import Discriminator  # noqa: E402


def sampler(d, rng, batch):
    idx = rng.integers(0, len(d["obs"]), batch)
    return {"obs": torch.as_tensor(d["obs"][idx]).float(), "obs2": torch.as_tensor(d["obs2"][idx]).float()}


def episode_rewards(disc, d):
    r = disc.reward(torch.as_tensor(d["obs"]).float(), torch.as_tensor(d["obs2"]).float()).cpu().numpy()
    return {int(e): float(r[d["episode"] == e].mean()) for e in np.unique(d["episode"])}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--policy-buffer", required=True)
    ap.add_argument("--gate-expert", required=True)
    ap.add_argument("--gate-wrong", required=True)
    ap.add_argument("--updates", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None, help="학습한 판별기 저장 경로")
    args = ap.parse_args()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    expert, policy = dict(np.load(args.expert)), dict(np.load(args.policy_buffer))
    disc = Discriminator(expert["obs"].shape[1], device="cuda")
    for i in range(args.updates):
        stats = disc.update(sampler(expert, rng, args.batch), sampler(policy, rng, args.batch))
        if (i + 1) % 500 == 0:
            print(f"update {i + 1}: {stats}", flush=True)

    ge, gw = dict(np.load(args.gate_expert)), dict(np.load(args.gate_wrong))
    re, rw = episode_rewards(disc, ge), episode_rewards(disc, gw)
    paired = [e for e in re if e in rw]  # 같은 장면(같은 seed)에서 맞는 색 vs 잘못된 색
    result = {
        "auc_correct_vs_wrong": auc(list(re.values()), list(rw.values())),
        "paired_correct_higher": float(np.mean([re[e] > rw[e] for e in paired])) if paired else None,
        "n_paired": len(paired),
        "mean_reward_correct": float(np.mean(list(re.values()))),
        "mean_reward_wrong": float(np.mean(list(rw.values()))),
        "final_d_stats": stats, "updates": args.updates,
        "n_expert_pairs": len(expert["obs"]), "n_policy_pairs": len(policy["obs"]),
    }
    print(json.dumps(result, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    if args.save:
        disc.save(args.save)


if __name__ == "__main__":
    main()
