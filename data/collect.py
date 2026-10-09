"""시연 스크립트를 실행해 LeRobot v3 데이터셋으로 저장한다. 성공한 에피소드만 남긴다."""

import argparse
import json
from collections import Counter

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from calib import FEATURE_NAMES
from expert.scripted import ScriptedExpert
from sim.env import COLORS, FPS, IMG_H, IMG_W, MAX_STEPS, BallSortEnv

_IMG = {"dtype": "video", "shape": (IMG_H, IMG_W, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "action": {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": FEATURE_NAMES},
    "observation.images.front": _IMG,
    "observation.images.top": _IMG,
}


def collect(repo_id, root, n_per_color, seed=0):
    ds = LeRobotDataset.create(repo_id, FPS, FEATURES, root=root, robot_type="so_follower", use_videos=True)
    env = BallSortEnv(render=True)
    expert = ScriptedExpert(env, seed=seed)
    saved, discarded, per_color = 0, 0, Counter()
    ep_seed = seed
    try:
        for color in COLORS:
            while per_color[color] < n_per_color:
                obs, info = env.reset(seed=ep_seed, options={"target": color})
                ep_seed += 1
                expert.reset()
                for _ in range(MAX_STEPS):
                    executed, clean = expert.act()
                    ds.add_frame(
                        {
                            "observation.state": obs["observation.state"],
                            "action": clean.astype(np.float32),
                            "observation.images.top": obs["observation.images.top"],
                            "observation.images.front": obs["observation.images.front"],
                            "task": info["task"],
                        }
                    )
                    obs, _, terminated, truncated, info = env.step(executed)
                    if terminated or truncated:
                        break
                if info["success"]:
                    ds.save_episode()
                    saved += 1
                    per_color[color] += 1
                    print(f"[{saved}] {color} saved (seed {ep_seed - 1})", flush=True)
                else:
                    ds.clear_episode_buffer()
                    discarded += 1
    finally:
        ds.finalize()
        env.close()
    return {"saved": saved, "discarded": discarded, "per_color": dict(per_color)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/so101_ballsort_sim")
    ap.add_argument("--root", default="data/so101_ballsort_sim")
    ap.add_argument("--n-per-color", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(collect(args.repo_id, args.root, args.n_per_color, args.seed), indent=2))


if __name__ == "__main__":
    main()
