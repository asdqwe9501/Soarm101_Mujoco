import json

from data.collect import collect


def test_collect_small_dataset(tmp_path):
    root = tmp_path / "ds"
    stats = collect("local/test_ballsort", root, n_per_color=1, seed=0)
    assert stats["saved"] == 3

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test_ballsort", root=root, video_backend="pyav")
    assert ds.num_episodes == 3
    assert set(ds.meta.tasks.index) == {
        "grab the red ball and put it in the box",
        "grab the black ball and put it in the box",
        "grab the sky blue ball and put it in the box",
    }
    item = ds[0]
    assert tuple(item["observation.state"].shape) == (6,)
    assert tuple(item["action"].shape) == (6,)
    assert tuple(item["observation.images.top"].shape) == (3, 480, 640)

    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["fps"] == 30 and info["robot_type"] == "so_follower"
    assert info["features"]["action"]["names"][0] == "shoulder_pan.pos"


def test_same_layout_gives_identical_first_frames_for_three_commands(tmp_path):
    root = tmp_path / "ds_same"
    stats = collect("local/test_same", root, n_per_color=1, seed=7, same_layout=True)
    assert stats["saved"] == 3

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test_same", root=root, video_backend="pyav")
    starts = [int(i) for i in ds.meta.episodes["dataset_from_index"]]
    firsts = [ds[i] for i in starts]
    assert len({f["task"] for f in firsts}) == 3
    for f in firsts[1:]:
        assert (f["observation.state"] == firsts[0]["observation.state"]).all()
        diff = (f["observation.images.front"] - firsts[0]["observation.images.front"]).abs().mean()
        assert diff < 0.01  # 같은 배치 (영상 압축 오차만 허용)


# ---------- Plan 5 ----------
from collections import Counter

from data.collect import _scene_mix_schedule
from sim.env import COLORS


def test_scene_mix_schedule_never_commands_aside_ball():
    per_color = Counter()
    scenes = Counter()
    seen = set()
    for color, ep_seed, scene, aside in _scene_mix_schedule(per_color, 300, 1000):
        assert color != aside
        assert (aside is None) == (scene == "three")
        if ep_seed not in seen:
            seen.add(ep_seed)
            scenes[scene] += 1
        per_color[color] += 1
    assert all(per_color[c] == 300 for c in COLORS)
    n = sum(scenes.values())
    assert abs(scenes["three"] / n - 0.6) < 0.06 and abs(scenes["two"] / n - 0.2) < 0.06, scenes


def test_collect_mixed_options(tmp_path):
    root = tmp_path / "ds_mix"
    stats = collect(
        "local/test_mix", root, n_per_color=2, seed=40, same_layout=True,
        randomize=True, mode_mix=True, scene_mix=True,
    )
    assert stats["saved"] == 6 and stats["per_color"] == {c: 2 for c in COLORS}
    assert sum(stats["per_mode"].values()) == 6 and sum(stats["per_scene"].values()) == 6
