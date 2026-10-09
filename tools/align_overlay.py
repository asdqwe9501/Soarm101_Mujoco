"""실제 데이터셋 프레임과 시뮬레이션 렌더링을 50% 겹쳐 PNG로 저장한다.

사용: python tools/align_overlay.py --episode 0 --frames 0 150 300
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 저장소 루트

import imageio.v3 as iio
import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from sim.env import BallSortEnv

REPO_ID = "taeyoungss/SmolVla_grab_blackball"
ROOT = Path.home() / ".cache/huggingface/lerobot/taeyoungss/SmolVla_grab_blackball"
OUT = Path("outputs/align")


def to_hwc_uint8(img):
    a = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if a.shape[0] == 3:
        a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = (a * 255).clip(0, 255).astype(np.uint8)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 150, 300])
    args = ap.parse_args()

    ds = LeRobotDataset(REPO_ID, root=ROOT, episodes=[args.episode], video_backend="pyav")
    env = BallSortEnv(render=True)
    env.reset(seed=0)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.frames:
        item = ds[f]
        q = env.calib.norm_to_rad(item["observation.state"].numpy())
        env.data.qpos[:6] = q
        mujoco.mj_forward(env.model, env.data)
        obs = env._obs()
        for key in ["observation.images.top", "observation.images.front"]:
            real = to_hwc_uint8(item[key])
            blend = (0.5 * real + 0.5 * obs[key]).astype(np.uint8)
            name = key.split(".")[-1]
            iio.imwrite(OUT / f"ep{args.episode}_f{f}_{name}.png", np.concatenate([real, obs[key], blend], axis=1))
        print(f, np.round(item["observation.state"].numpy(), 1))
    env.close()


if __name__ == "__main__":
    main()
