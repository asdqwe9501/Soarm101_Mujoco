"""시연 데이터에서 손목 카메라 프레임을 뽑아 색 가드 조정/시험용으로 저장한다.

- preclose: 행동의 그리퍼 명령이 40을 넘었다가 처음 20 아래로 내려가는 프레임 (StagedReward의 잡기 시도 순간)
- approach: 같은 에피소드에서 그 이전 구간(30~80%) 중 무작위 프레임 (이동 중, 정답은 None 또는 명령 색)

출력: <out>/<kind>_<color>_<dataset>_ep<idx>_f<frame>.png 와 <out>/labels.csv (file,kind,color,dataset,episode,frame)
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np
from PIL import Image

from rl.staged_reward import GRIP_CLOSE_CMD, GRIP_OPEN_CMD

PREFIX = "grab the "


def preclose_index(grip):
    armed = False
    for i, g in enumerate(grip):
        if g > GRIP_OPEN_CMD:
            armed = True
        elif armed and g < GRIP_CLOSE_CMD:
            return i
    return None


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["data/so101_ballsort_sim", "data/so101_ballsort_sim_corr"])
    ap.add_argument("--per-color", type=int, default=60)
    ap.add_argument("--out", default="outputs/guard_frames")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows, count = [], {}
    per_ds = int(np.ceil(args.per_color / len(args.datasets)))
    for root in args.datasets:
        name = Path(root).name
        ds = LeRobotDataset(f"local/{name}", root=root, video_backend="pyav")
        ep_idx = np.asarray(ds.hf_dataset["episode_index"])
        actions = np.stack(ds.hf_dataset["action"])
        n_ds = {}
        for ep in rng.permutation(ds.meta.total_episodes):
            meta = ds.meta.episodes[int(ep)]
            color = meta["tasks"][0][len(PREFIX):].split(" ball")[0]
            if n_ds.get(color, 0) >= per_ds or count.get(color, 0) >= args.per_color:
                continue
            frames = np.nonzero(ep_idx == ep)[0]
            k = preclose_index(actions[frames, 5])
            if k is None:
                continue
            j = int(rng.uniform(0.3, 0.8) * k)
            for kind, f in (("preclose", k), ("approach", j)):
                img = ds[int(frames[f])]["observation.images.top"]
                arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                fn = f"{kind}_{color.replace(' ', '_')}_{name}_ep{int(ep):04d}_f{f:03d}.png"
                Image.fromarray(arr).save(out / fn)
                rows.append([fn, kind, color, name, int(ep), f])
            n_ds[color] = n_ds.get(color, 0) + 1
            count[color] = count.get(color, 0) + 1
            if all(v >= per_ds for v in n_ds.values()) and len(n_ds) == 3:
                break
    with open(out / "labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "kind", "color", "dataset", "episode", "frame"])
        w.writerows(rows)
    print(count, len(rows), "frames ->", out)


if __name__ == "__main__":
    main()
