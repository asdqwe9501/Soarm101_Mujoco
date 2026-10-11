"""로컬 LeRobot 데이터셋 여러 개를 하나로 합친다.

사용: python tools/merge_datasets.py data/so101_ballsort_sim_all data/so101_ballsort_sim data/so101_ballsort_sim_corr
"""

import sys
from pathlib import Path

from lerobot.datasets.dataset_tools import merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(out, *srcs):
    out = Path(out)
    datasets = [LeRobotDataset(f"local/{Path(s).name}", root=s) for s in srcs]
    merged = merge_datasets(datasets, output_repo_id=f"local/{out.name}", output_dir=out)
    print("episodes", merged.meta.total_episodes, "frames", merged.meta.total_frames, "tasks", merged.meta.total_tasks)


if __name__ == "__main__":
    main(*sys.argv[1:])
