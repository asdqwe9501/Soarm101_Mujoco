import csv
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
CK = "outputs/train/sft_v3/checkpoints/last/pretrained_model"


def test_two_episode_smoke_and_adapter(tmp_path):
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, "-m", "train.dsrl", "--checkpoint", CK, "--out", str(out),
         "--episodes", "2", "--warmup-episodes", "1", "--save-every", "1", "--max-env-steps", "200"],
        check=True,
    )
    assert (out / "last.pt").exists() and (out / "metrics.csv").exists() and (out / "buffer.npz").exists()

    # 재개: 에피소드 번호가 metrics.csv에서 이어진다
    subprocess.run(
        [sys.executable, "-m", "train.dsrl", "--checkpoint", CK, "--out", str(out),
         "--episodes", "3", "--warmup-episodes", "1", "--save-every", "1", "--max-env-steps", "100", "--resume"],
        check=True,
    )
    rows = list(csv.DictReader(open(out / "metrics.csv")))
    assert [int(r["episode"]) for r in rows] == [1, 2, 3]

    from eval.evaluate import evaluate
    from policies.dsrl import DSRLPolicyAdapter

    pol = DSRLPolicyAdapter(CK, str(out / "last.pt"))
    r = evaluate(pol, n_per_color=1, seed=0, render=True)
    assert r["n"] == 3 and 0.0 <= r["success_rate"] <= 1.0
