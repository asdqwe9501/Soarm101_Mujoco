"""잡기 실패 원인 분석: 전체 작업 에피소드마다 닫기 순간의 집게-공 상대 위치와 공의 움직임을 기록해 실패를 나눈다.

python tools/grasp_fail_analysis.py --checkpoint <ck> --actor <pt> --guard --commands basic --n-per-color 50 \
    --out docs/results/grasp/grasp_irl2_guard_basic.json --video-dir outputs/grasp/videos_basic

닫기 = 그리퍼 명령이 40을 넘었다가 20 아래로 내려간 순간(StagedReward와 같은 판정).
닫기 순간 공의 위치를 집게 기준으로 나눈다(tools/grasp_tolerance.py와 같은 축).
- axis: 닫히는 축(고정 집게 쪽 +). 움직이는 집게가 공을 쓸어 오므로 이 방향 오차에는 비교적 강하다.
- width: 집게 폭 방향. 집게 끝이 좁아(8~10mm) 1~1.5cm 넘게 어긋나면 공이 옆으로 짜여 나간다.

실패 분류(성공하지 못한 에피소드):
- wrong_ball: 다른 색 공이 들리거나 통에 들어감
- no_close: 목표 공 근처(수평 3.5cm, 집게 높이 7cm 이하)에서 한 번도 닫지 않음 (closes_elsewhere: 다른 곳에서는 닫음)
- pushed: 처음 목표 공 근처에서 닫기 전에 공이 1.5cm 넘게 밀려남
- too_high: 목표 공 근처에서 닫았으나 집게 끝이 너무 높음(> 2.5cm, 공 중심 2cm·시연 1.2cm)
- offset_miss: 목표 공 근처에서 닫았으나 공이 1.5cm 넘게 어긋나 들리지 않음
- closed_not_lifted: 1.5cm 이내로 닫았는데도 들리지 않음
- dropped: 들렸다가 통 밖에 떨어짐
- held_timeout: 들고 있는 채로 시간이 끝남
- return_fail: 통에 넣었지만 제자리 복귀 전에 시간이 끝남
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.evaluate import _options  # noqa: E402
from rl.staged_reward import ATTEMPT_MAX_Z, ATTEMPT_RADIUS, GRIP_CLOSE_CMD, GRIP_OPEN_CMD, LIFT_Z  # noqa: E402
from sim.env import COLORS, FPS, MAX_STEPS, BallSortEnv  # noqa: E402

OFFSET_OK = 0.015  # tools/grasp_tolerance.py: 1.5cm까지 10/12 성공
PUSHED = 0.015
TOO_HIGH = 0.025  # 공 중심 0.02, 시연의 닫는 높이 GRASP_Z 0.012


def close_record(env, t):
    tip = env.data.site_xpos[env._site_id].copy()
    gid = env.model.body("gripper").id
    u = -env.data.xmat[gid].reshape(3, 3)[:2, 0]
    u /= np.linalg.norm(u)
    w = np.array([-u[1], u[0]])
    b = env.ball_pos(env.target)
    d = b[:2] - tip[:2]
    dist = {c: float(np.hypot(*(env.ball_pos(c)[:2] - tip[:2]))) for c in COLORS if not env.ball_in_bin(c)}
    return {"step": t, "tip_z": float(tip[2]), "ball_z": float(b[2]), "offset": float(np.linalg.norm(d)),
            "axis": float(d @ u), "width": float(d @ w), "nearest": min(dist, key=dist.get),
            "near_target": bool(tip[2] <= ATTEMPT_MAX_Z and np.linalg.norm(d) <= ATTEMPT_RADIUS),
            "ball_xy": b[:2].round(4).tolist()}


def run(env, policy, obs, frames):
    policy.reset(env)
    target = env.target
    start_xy = env.ball_pos(target)[:2].copy()
    armed, closes, events = False, [], []
    held, ever_lifted, other_lifted, steps = False, False, False, 0
    info = {}
    for t in range(MAX_STEPS):
        a = policy.act(obs)
        if a[5] > GRIP_OPEN_CMD:
            armed = True
        elif armed and a[5] < GRIP_CLOSE_CMD:
            armed = False
            closes.append(close_record(env, t))
        obs, _, term, trunc, info = env.step(a)
        steps = t + 1
        if frames is not None:
            frames.append(obs["observation.images.front"])
        z = env.ball_pos(target)[2]
        if z > LIFT_Z and not held:
            held, ever_lifted = True, True
            events.append(("lift", t))
        elif held and z < 0.03:
            held = False
            events.append(("in_bin" if env.ball_in_bin(target) else "drop", t))
        other_lifted |= any(env.ball_pos(c)[2] > LIFT_Z for c in COLORS if c != target and not env.ball_in_bin(c))
        if term or trunc:
            break
    near = [c for c in closes if c["near_target"]]
    first_near = near[0] if near else None
    ep = {"success": bool(info.get("success")), "steps": steps, "n_closes": len(closes), "closes": closes,
          "events": events, "ever_lifted": ever_lifted, "start_xy": start_xy.round(4).tolist()}
    if ep["success"]:
        ep["category"] = "success"
    elif info.get("wrong_in_bin") or other_lifted:
        ep["category"] = "wrong_ball"
    elif info.get("target_in_bin"):
        ep["category"] = "return_fail"
    elif ever_lifted:
        ep["category"] = "held_timeout" if held else "dropped"
    elif first_near is None:
        ep["category"] = "no_close" if not closes else "closes_elsewhere"
    elif np.hypot(*(np.array(first_near["ball_xy"]) - start_xy)) > PUSHED:
        ep["category"] = "pushed"
    elif first_near["tip_z"] > TOO_HIGH:
        ep["category"] = "too_high"
    elif first_near["offset"] > OFFSET_OK:
        ep["category"] = "offset_miss"
    else:
        ep["category"] = "closed_not_lifted"
    return ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--guard", action="store_true")
    ap.add_argument("--commands", default="basic")
    ap.add_argument("--n-per-color", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-dir", default=None, help="실패 분류마다 영상 최대 2개")
    args = ap.parse_args()

    from policies.dsrl import DSRLPolicyAdapter
    policy = DSRLPolicyAdapter(args.checkpoint, args.actor)
    if args.guard:
        from policies.color_guard import ColorGuard
        policy = ColorGuard(policy)
    env = BallSortEnv(render=True)
    cmd_rng = np.random.default_rng(args.seed + 7)
    eps, saved = [], Counter()
    k = 0
    for color in COLORS:
        for _ in range(args.n_per_color):
            obs, _ = env.reset(seed=args.seed + k, options=_options(color, args.commands, cmd_rng))
            frames = [] if args.video_dir else None
            ep = dict(run(env, policy, obs, frames), episode=k, color=color, command=env.task)
            if args.guard:
                ep["guard_checks"] = policy.stats["checks"]
            eps.append(ep)
            cat = ep["category"]
            if frames and cat != "success" and saved[cat] < 2:
                Path(args.video_dir).mkdir(parents=True, exist_ok=True)
                name = f"ep{k:04d}_{color.replace(' ', '_')}_{cat}.mp4"
                imageio.mimsave(Path(args.video_dir) / name, frames, fps=FPS)
                with open(Path(args.video_dir) / "videos.txt", "a", encoding="utf-8") as f:
                    f.write(f"{name}\t{env.task}\n")
                saved[cat] += 1
            fn = ep["closes"][0] if ep["closes"] else None
            print(k, color, cat, "closes", ep["n_closes"],
                  f"first off {100 * fn['offset']:.1f}cm axis {100 * fn['axis']:+.1f} width {100 * fn['width']:+.1f}" if fn else "",
                  flush=True)
            k += 1

    def first_near(e):
        return next((c for c in e["closes"] if c["near_target"]), None)

    cats = Counter(e["category"] for e in eps)
    offs = defaultdict(list)
    for e in eps:
        c = first_near(e)
        if c:
            offs[e["category"]].append(c)
    off_summary = {cat: {"n": len(v), "offset_cm_median": round(100 * float(np.median([c["offset"] for c in v])), 2),
                         "abs_width_cm_median": round(100 * float(np.median([abs(c["width"]) for c in v])), 2),
                         "abs_axis_cm_median": round(100 * float(np.median([abs(c["axis"]) for c in v])), 2),
                         "tip_z_cm_median": round(100 * float(np.median([c["tip_z"] for c in v])), 2)}
                   for cat, v in offs.items()}
    summary = {"checkpoint": args.checkpoint, "actor": args.actor, "guard": args.guard, "commands": args.commands,
               "n": len(eps), "categories": dict(cats), "first_close_near_target": off_summary, "episodes": eps}
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
