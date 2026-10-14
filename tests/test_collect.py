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


def test_aux_labels_record_target_xy(tmp_path):
    root = tmp_path / "ds_aux"
    collect("local/test_aux", root, n_per_color=1, seed=11, aux_labels=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test_aux", root=root, video_backend="pyav")
    assert "target_xy" in ds.meta.features and tuple(ds.meta.features["target_xy"]["shape"]) == (2,)
    item = ds[0]
    xy = item["target_xy"].numpy()
    assert 0.05 < xy[0] < 0.40 and -0.30 < xy[1] < 0.30  # 책상 위 작업 공간


def test_paraphrase_uses_varied_training_commands(tmp_path):
    from sim.commands import COLOR_WORDS, TRAIN_TEMPLATES, normalize

    root = tmp_path / "ds_para"
    stats = collect("local/test_para", root, n_per_color=2, seed=21, paraphrase=True)
    # 색별 2개: basic round(1.2)=1, similar round(0.6)=1, rare 0
    for c in ("red", "black", "sky blue"):
        assert stats["per_tier"].get(f"{c}/basic") == 1 and stats["per_tier"].get(f"{c}/similar") == 1
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/test_para", root=root, video_backend="pyav")
    tasks = list(ds.meta.tasks.index)
    allowed = {t.format(color=w) for t in TRAIN_TEMPLATES for ws in COLOR_WORDS.values() for w in ws}
    assert {normalize(t) for t in tasks} <= allowed and len(tasks) >= 3
    assert "Grab the blue ball." in tasks  # 하늘색 공은 blue로 부른다
