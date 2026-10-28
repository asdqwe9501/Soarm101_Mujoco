"""정책이 실제로 닫는 순간의 손목 프레임을 모아 색 가드 분류기를 조정/시험한다.

python tools/policy_close_frames.py --layouts 200 --seed 70000 --out outputs/guard_frames_policy

반사실 시험과 같은 배치(배치마다 세 색 명령)에서 IRL v2 정책(가드 없음)을 첫 잡기 시도까지 돌리고,
그리퍼 명령이 40 아래로 내려간 뒤 20 아래로 내려갈 때까지의 모든 스텝(가드가 판별하는 구간) 손목 프레임을 저장한다.
라벨 aimed: 그 순간 집게가 노린 공(StagedReward와 같은 판정, 없으면 none). 시뮬레이터 정답이므로 시험용으로만 쓴다.
출력: <out>/L<layout>_<command>_s<step>.png 와 <out>/labels.csv (file,layout,command,step,cmd,aimed)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.staged_reward import GRIP_CLOSE_CMD, GRIP_OPEN_CMD, StagedReward  # noqa: E402
from sim.env import COLORS, MAX_STEPS, BallSortEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/train/sft_aux2_irl/pretrained_model")
    ap.add_argument("--actor", default="outputs/irl/irl2/last.pt")
    ap.add_argument("--layouts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=70000)
    ap.add_argument("--out", default="outputs/guard_frames_policy")
    args = ap.parse_args()

    from policies.dsrl import DSRLPolicyAdapter

    policy = DSRLPolicyAdapter(args.checkpoint, args.actor)
    env = BallSortEnv(render=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for li in range(args.layouts):
        for color in COLORS:
            obs, _ = env.reset(seed=args.seed + li, options={"target": color, "command": f"Grab the {color} ball."})
            policy.reset(env)
            sr = StagedReward(env)
            sr.reset()
            armed = False
            for t in range(MAX_STEPS):
                a = policy.act(obs)
                if a[5] > GRIP_OPEN_CMD:
                    armed = True
                elif armed:
                    aimed = sr._aimed_ball() or "none"
                    fn = f"L{li:03d}_{color.replace(' ', '_')}_s{t:03d}.png"
                    Image.fromarray(obs["observation.images.top"]).save(out / fn)
                    rows.append([fn, li, color, t, round(float(a[5]), 1), aimed])
                    if a[5] < GRIP_CLOSE_CMD:
                        break
                obs, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break
        print(li, len(rows), flush=True)
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "layout", "command", "step", "cmd", "aimed"])
        w.writerows(rows)
    print(len(rows), "frames ->", out)


if __name__ == "__main__":
    main()
