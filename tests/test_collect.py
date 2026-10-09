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
