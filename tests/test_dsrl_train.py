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


def test_episode_options_commands_and_randomize():
    import numpy as np
    from sim.commands import TRAIN_TEMPLATES, normalize
    from train.dsrl import episode_options

    rng = np.random.default_rng(0)
    assert episode_options("red", "canonical", False, rng) == {"target": "red"}
    assert episode_options("sky blue", "basic", True, rng) == {
        "target": "sky blue", "command": "Grab the blue ball.", "randomize": True}
    o = episode_options("black", "train", False, np.random.default_rng(1))
    assert "randomize" not in o and "black" in o["command"]
    assert any(normalize(o["command"]) == t.format(color=w)
               for t in TRAIN_TEMPLATES for w in ("black", "black-colored"))
    # 같은 seed면 같은 명령
    assert o == episode_options("black", "train", False, np.random.default_rng(1))


def test_upper_bound_command_for():
    import numpy as np
    sys.path.insert(0, "tools")
    from dsrl_upper_bound import command_for

    rng = np.random.default_rng(0)
    assert command_for("red", "canonical", rng) is None
    assert command_for("sky blue", "basic", rng) == "Grab the blue ball."
    assert "red" in command_for("red", "heldout", rng)
