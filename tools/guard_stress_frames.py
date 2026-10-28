"""색 가드 스트레스 프레임: 공끼리 가까운 배치에서 일부러 빗나가게(이웃 공 쪽으로) 닫는 순간의 손목 프레임.

python tools/guard_stress_frames.py --episodes 600 --seed 90000 --out outputs/guard_frames_stress

공 사이가 가까운(MAX_GAP 이하) 배치만 골라, 시연 스크립트가 노릴 공 X의 위치를 이웃 공 쪽(또는 무작위 방향)으로
0~OFF_MAX만큼 옮겨 잡게 한다. 그리퍼 명령이 40 아래로 내려간 뒤 20 아래로 내려갈 때까지의 프레임을 저장한다.
라벨 aimed: 그 순간 집게가 노린 공(StagedReward 판정, 없으면 none).
라벨 lifted: 닫은 뒤 시연을 이어 갔을 때 실제로 들린 공(LIFT_Z 위, 없으면 none). 색 정확도의 실제 기준.
출력: <out>/E<ep>_<aim>_s<step>.png 와 <out>/labels.csv (file,episode,aim,neighbor,offset,step,cmd,aimed,lifted)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from expert.scripted import ScriptedExpert  # noqa: E402
from rl.staged_reward import GRIP_CLOSE_CMD, GRIP_OPEN_CMD, LIFT_Z, StagedReward  # noqa: E402
from sim.env import COLORS, BallSortEnv  # noqa: E402

MAX_GAP = 0.10  # 가장 가까운 이웃 공까지 거리 상한
OFF_MAX = 0.035  # 노리는 위치를 옮기는 최대 거리 (= StagedReward 시도 반경)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=90000)
    ap.add_argument("--out", default="outputs/guard_frames_stress")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    env = BallSortEnv(render=True)
    sr = StagedReward(env)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, ep, s = [], 0, args.seed
    while ep < args.episodes:
        s += 1
        aim = COLORS[rng.integers(3)]
        obs, _ = env.reset(seed=s, options={"target": aim})
        pos = {c: env.ball_pos(c)[:2].copy() for c in COLORS}
        gaps = {c: np.hypot(*(pos[c] - pos[aim])) for c in COLORS if c != aim}
        nb = min(gaps, key=gaps.get)
        if gaps[nb] > MAX_GAP:
            continue
        u = (pos[nb] - pos[aim]) / gaps[nb]
        if rng.random() < 0.3:  # 일부는 무작위 방향
            th = rng.uniform(0, 2 * np.pi)
            u = np.array([np.cos(th), np.sin(th)])
        off = rng.uniform(0, OFF_MAX) * u
        real = env.ball_pos

        def shifted(c, real=real):
            p = real(c).copy()
            if c == aim:
                p[:2] += off
            return p

        ex = ScriptedExpert(env, seed=s, dart_sigma=0.0)
        env.ball_pos = shifted
        ex.reset()
        env.ball_pos = real
        sr.reset()
        armed, n0 = False, len(rows)
        for t in range(ex.traj.shape[0]):
            a, _ = ex.act()
            if a[5] > GRIP_OPEN_CMD:
                armed = True
            elif armed:
                aimed = sr._aimed_ball() or "none"
                fn = f"E{ep:04d}_{aim.replace(' ', '_')}_s{t:03d}.png"
                Image.fromarray(obs["observation.images.top"]).save(out / fn)
                rows.append([fn, ep, aim, nb, round(float(np.linalg.norm(off)), 4), t, round(float(a[5]), 1), aimed])
                if a[5] < GRIP_CLOSE_CMD:
                    break
            obs, *_ = env.step(a)
        for _ in range(75):  # 닫기 마무리(1초) + 들기(1.5초)
            a, _ = ex.act()
            env.step(a)
        up = [c for c in COLORS if env.ball_pos(c)[2] > LIFT_Z]
        for r in rows[n0:]:
            r.append(up[0] if up else "none")
        ep += 1
        if ep % 50 == 0:
            print(ep, len(rows), flush=True)
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "episode", "aim", "neighbor", "offset", "step", "cmd", "aimed", "lifted"])
        w.writerows(rows)
    print(len(rows), "frames ->", out)


if __name__ == "__main__":
    main()
