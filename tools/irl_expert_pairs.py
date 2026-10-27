"""IRL 시범 데이터: 시연 스크립트를 실시간으로 실행하며 RL 한 스텝(50 env 스텝)마다 (s, s') 특징 쌍을 기록한다.

python tools/irl_expert_pairs.py --checkpoint <ck> --episodes 300 --seed 700000 --out data/irl/expert_pairs.npz
--aim wrong: 명령하지 않은 색 공을 집어 넣는 시범(관문 시험용 음성 표본). 움직임 스타일은 같고 색만 틀린다.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert.scripted import ScriptedExpert  # noqa: E402
from policies.dsrl import NoiseSteeredSmolVLA  # noqa: E402
from policies.dsrl_constants import EXEC_STEPS  # noqa: E402
from sim.env import COLORS, MAX_STEPS, BallSortEnv  # noqa: E402
from train.dsrl import episode_options  # noqa: E402


def run_episode(env, steer, seed, aim, rng, commands="train"):
    color = COLORS[rng.integers(len(COLORS))]
    obs, _ = env.reset(seed=seed, options=episode_options(color, commands, True, np.random.default_rng([seed, 1])))
    steer.reset(env)
    ex = ScriptedExpert(env, seed=seed)
    if aim == "wrong":
        env.target = [c for c in COLORS if c != color][rng.integers(len(COLORS) - 1)]
    ex.reset()
    env.target = color  # 명령과 성공 판정은 원래 색 기준
    feats, done, success = [steer.features(obs)], False, False
    for t in range(1, MAX_STEPS + 1):
        obs, _, success, trunc, info = env.step(ex.act()[0])
        if success or trunc or t % EXEC_STEPS == 0:
            feats.append(steer.features(obs))
        if aim == "wrong" and info["wrong_in_bin"] and env.at_home():
            break  # 잘못된 공을 넣고 복귀하면 끝(그 뒤는 정지 장면)
        if success or trunc:
            break
    wrong_done = aim == "wrong" and info["wrong_in_bin"]
    return np.stack(feats), success, wrong_done, env.task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--aim", choices=["target", "wrong"], default="target")
    ap.add_argument("--commands", choices=["train", "irl", "heldout"], default="train",
                    help="명령문 분할 (irl: 학습 + IRL 확장 표현, heldout: 평가 전용 표현 — 관문 시험용)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    env = BallSortEnv(render=True)
    steer = NoiseSteeredSmolVLA(args.checkpoint, device="cuda")
    obs, obs2, done, ep_id, tasks = [], [], [], [], []
    n_ok = 0
    for i in range(args.episodes):
        seed = args.seed + i
        f, success, wrong_done, task = run_episode(env, steer, seed, args.aim, np.random.default_rng(seed), args.commands)
        ok = success if args.aim == "target" else wrong_done
        print(f"ep {i} seed {seed} {task!r} pairs {len(f) - 1} {'ok' if ok else 'FAILED'}", flush=True)
        if not ok:
            continue
        n_ok += 1
        obs.append(f[:-1])
        obs2.append(f[1:])
        d = np.zeros(len(f) - 1, np.float32)
        d[-1] = 1.0 if args.aim == "target" else 0.0  # 성공만 종료 상태
        done.append(d)
        ep_id.append(np.full(len(f) - 1, i))
        tasks.append(task)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, obs=np.concatenate(obs).astype(np.float16), obs2=np.concatenate(obs2).astype(np.float16),
             done=np.concatenate(done), episode=np.concatenate(ep_id), tasks=np.array(tasks))
    print(f"saved {args.out}: episodes {n_ok}/{args.episodes}, pairs {sum(len(o) for o in obs)}", flush=True)


if __name__ == "__main__":
    main()
