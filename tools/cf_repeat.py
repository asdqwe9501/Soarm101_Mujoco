"""반사실 시험 오류의 재현 시험: 같은 장면 + 같은 명령문을 노이즈만 바꿔 K번 반복한다.

오류 사례(첫 시도가 다른 색)와, 비교용으로 같은 수의 무작위 정답 사례를 반복한다.
반복해도 계속 틀리면 장면이 원인인 체계적 오류이고, 대부분 맞으면 노이즈에 따른 확률적 오류다.

python tools/cf_repeat.py --records docs/results/cf_rec_aux2_heldout.json --policy smolvla --checkpoint <ck> \
    --repeats 10 --out docs/results/cf_repeat_aux2_heldout.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from counterfactual_eval import RandomLatentPolicy, run_attempt  # noqa: E402


def main():
    from rl.staged_reward import StagedReward
    from sim.env import COLORS, BallSortEnv

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="counterfactual_eval.py --records 출력")
    ap.add_argument("--policy", choices=["smolvla", "dsrl"], default="smolvla")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--actor", default=None)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = json.load(open(args.records))["records"]
    wrong = [r for r in recs if r["outcome"] in COLORS and r["outcome"] != r["color"]]
    right = [r for r in recs if r["outcome"] == r["color"]]
    rng = np.random.default_rng(args.seed)
    control = [right[i] for i in rng.choice(len(right), min(len(wrong), len(right)), replace=False)]

    if args.policy == "dsrl":
        from policies.dsrl import DSRLPolicyAdapter
        pol = DSRLPolicyAdapter(args.checkpoint, args.actor)
    else:
        pol = RandomLatentPolicy(args.checkpoint, seed=args.seed + 1)
    env = BallSortEnv(render=True)
    sr = StagedReward(env)

    cases = []
    for group, rs in (("error", wrong), ("control", control)):
        for r in rs:
            outs = []
            for _ in range(args.repeats):
                obs, _ = env.reset(seed=r["seed"], options={"target": r["color"], "command": r["command"]})
                assert np.allclose(env.ball_pos(r["color"])[:2], r["balls"][r["color"]], atol=1e-3)  # 같은 장면
                pol.reset(env)
                sr.reset()
                outs.append(run_attempt(env, sr, pol.act, obs)[0])
            case = dict(r, group=group, repeats=outs,
                        repeat_correct=sum(o == r["color"] for o in outs) / len(outs),
                        repeat_same_wrong=sum(o == r["outcome"] for o in outs) / len(outs) if group == "error" else None)
            cases.append(case)
            print(group, r["layout"], r["color"], repr(r["command"]), "first:", r["outcome"], "repeats:", outs, flush=True)
    summary = {
        "records": args.records, "policy": args.policy, "actor": args.actor, "repeats": args.repeats,
        "n_error_cases": len(wrong), "n_control_cases": len(control),
        "error_cases_repeat_correct_mean": float(np.mean([c["repeat_correct"] for c in cases if c["group"] == "error"])) if wrong else None,
        "control_cases_repeat_correct_mean": float(np.mean([c["repeat_correct"] for c in cases if c["group"] == "control"])) if control else None,
        "cases": cases,
    }
    print({k: v for k, v in summary.items() if k != "cases"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
