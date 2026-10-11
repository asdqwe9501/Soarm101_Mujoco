"""잡기 허용 오차 측정: 시연 스크립트가 계획할 때만 목표 공 위치를 반경 r만큼 어긋나게 해서 성공 수를 센다.

실행 경로는 어긋난 위치로 가지만 공은 실제 위치에 있으므로, 정책이 공에서 r만큼 빗나가게
내려앉았을 때 시뮬레이션 잡기가 얼마나 버티는지를 잰다. 방향은 0..2pi를 n등분, 장면은 seed=방향 번호.

사용: python tools/grasp_tolerance.py --radii 0.5 1.0 1.5 2.0 2.5 [--n-dirs 12]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트

import numpy as np

from expert.scripted import ScriptedExpert
from sim.env import MAX_STEPS, BallSortEnv


def run_offset(env, offset_xy, seed):
    """목표 공을 offset_xy[m]만큼 어긋난 곳에 있다고 믿고 계획한 시연 1회. 성공 여부를 돌려준다."""
    env.reset(seed=seed)
    ex = ScriptedExpert(env, seed=seed, dart_sigma=0.0)
    real = env.ball_pos

    def planned(color):
        p = real(color)
        if color == env.target:
            p[:2] += offset_xy
        return p

    env.ball_pos = planned  # 인스턴스 속성으로 가린다: 계획(reset) 동안에만
    try:
        ex.reset()
    finally:
        del env.ball_pos
    info = {"success": False}
    for _ in range(MAX_STEPS):
        _, _, te, tr, info = env.step(ex.act()[0])
        if te or tr:
            break
    return bool(info["success"])


def measure(env, radius, n_dirs=12):
    wins = 0
    for s in range(n_dirs):
        a = 2 * np.pi * s / n_dirs
        wins += run_offset(env, radius * np.array([np.cos(a), np.sin(a)]), seed=s)
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5], help="[cm]")
    ap.add_argument("--n-dirs", type=int, default=12)
    args = ap.parse_args()
    env = BallSortEnv(render=False)
    for r in args.radii:
        print(f"grasp offset {r:.1f} cm: {measure(env, r / 100, args.n_dirs)}/{args.n_dirs} success", flush=True)


if __name__ == "__main__":
    main()
