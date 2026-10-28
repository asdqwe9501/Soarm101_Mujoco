"""가드 재시도 진단: 가드가 제자리로 후퇴시킨 상태에서 정책이 다시 잡기를 시도하는가.

조건마다 같은 장면(seed)에서 시작 상태만 바꿔 정책을 실행하고, 첫 잡기 시도까지 걸린 스텝과 결과를 센다.
- fresh:  일반 리셋 상태(제자리, 그리퍼 닫힘)
- park:   가드 home 후퇴를 흉내 낸 상태(그리퍼를 열었다가 제자리로 가며 GRIP_PARK까지 닫음)
- park_open: 제자리에서 그리퍼를 연 채로 넘겨줌
- park_closed: 제자리에서 그리퍼를 시작 자세 값(SIM_HOME_NORM[5])까지 닫아 넘겨줌

python tools/guard_retry_diag.py --checkpoint <ck> --actor <pt> --n 30 --out docs/results/erran/guard_retry_diag.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calib import SIM_HOME_NORM  # noqa: E402
from expert.scripted import min_jerk  # noqa: E402
from policies.color_guard import GRIP_OPEN, GRIP_PARK, RETREAT_STEPS  # noqa: E402

LIMIT = 450  # 가드 후퇴 뒤 시도 제한(600)까지 남는 대략의 스텝


def prephase(env, end_grip):
    """현재 자세에서 그리퍼를 열고(10스텝), 제자리 자세로 가며 그리퍼를 end_grip으로 맞춘다(RETREAT_STEPS)."""
    obs = None
    start = env.joint_norm().astype(np.float32)
    opened = start.copy()
    opened[5] = GRIP_OPEN
    for _ in range(10):
        obs, *_ = env.step(opened)
    end = np.asarray(SIM_HOME_NORM, np.float32).copy()
    end[5] = end_grip
    for i in range(RETREAT_STEPS):
        obs, *_ = env.step(min_jerk(opened, end, (i + 1) / RETREAT_STEPS).astype(np.float32))
    return obs


def main():
    from policies.dsrl import DSRLPolicyAdapter
    from rl.staged_reward import StagedReward
    from sim.commands import command_for_tier
    from sim.env import COLORS, BallSortEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--conds", default="fresh,park,park_open,park_closed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pol = DSRLPolicyAdapter(args.checkpoint, args.actor)
    env = BallSortEnv(render=True)
    sr = StagedReward(env)
    rng = np.random.default_rng(0)
    results = {}
    grip = {"park": GRIP_PARK, "park_open": GRIP_OPEN, "park_closed": float(SIM_HOME_NORM[5])}
    for cond in args.conds.split(","):
        outs = []
        for k in range(args.n):
            color = COLORS[k % 3]
            obs, _ = env.reset(seed=args.seed + k, options={"target": color,
                                                            "command": command_for_tier(color, "basic", rng)})
            if cond != "fresh":
                obs = prephase(env, grip[cond])
            pol.reset(env)
            sr.reset()
            out, step = "timeout", LIMIT
            for t in range(LIMIT):
                a = pol.act(obs)
                obs, *_ = env.step(a)
                _, done, info = sr.step(a)
                if info["first_attempt"] is not None:
                    out = info["first_attempt"] if info["first_attempt"] != "none" else "none"
                    step = t + 1
                    break
                if done:
                    out = info["end_reason"]
                    break
            outs.append({"seed": args.seed + k, "color": color, "outcome": out, "step": step})
            print(cond, k, color, out, step, flush=True)
        c = Counter(o["outcome"] for o in outs)
        results[cond] = {"counts": dict(c), "median_step_to_attempt": float(np.median([o["step"] for o in outs if o["outcome"] in ("target", "other", "none")] or [np.nan])), "episodes": outs}
        print(cond, results[cond]["counts"], results[cond]["median_step_to_attempt"], flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
