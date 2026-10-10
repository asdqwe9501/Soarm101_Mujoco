"""관절 상태(observation.state)를 0으로 바꾼 데이터셋 사본을 만든다 (상태 지름길 차단 실험용).

영상은 하드링크로 공유하고, parquet의 state 열과 stats.json의 state 통계만 바꾼다.
사용: python tools/make_zero_state_dataset.py data/so101_ballsort_sim data/so101_ballsort_sim_nostate
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(src, dst):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    dst.mkdir(parents=True)
    subprocess.run(["cp", "-rl", str(src / "videos"), str(dst / "videos")], check=True)
    shutil.copytree(src / "meta", dst / "meta")
    for f in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        df["observation.state"] = [np.zeros(6, dtype=np.float32) for _ in range(len(df))]
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    s = stats["observation.state"]
    for k in s:
        if k == "count":
            continue
        s[k] = [1.0] * 6 if k == "std" else [0.0] * 6
    stats_path.write_text(json.dumps(stats, indent=4))
    print("wrote", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])
