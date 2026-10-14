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


def _per_color_schedule(per_color, n_per_color, seed):
    """색별로 n개가 모일 때까지 매번 새 배치(seed)를 쓴다."""
    ep_seed = seed
    for color in COLORS:
        while per_color[color] < n_per_color:
            yield color, ep_seed
            ep_seed += 1


def _same_layout_schedule(per_color, n_per_color, seed):
    """배치(seed) 하나에 세 색을 차례로 쓴다. 이미 다 모인 색은 건너뛴다."""
    ep_seed = seed
    while any(per_color[c] < n_per_color for c in COLORS):
        for color in COLORS:
            if per_color[color] < n_per_color:
                yield color, ep_seed
        ep_seed += 1


SCENE_P = {"three": 0.6, "two": 0.2, "prebinned": 0.2}
MODES = ("default", "correction", "wrong_approach")


def _scene_mix_schedule(per_color, n_per_color, seed):
    """배치(seed)마다 장면을 고르고(three 60%, two 20%, prebinned 20%), 책상 위에 남은 색에만 명령을 붙인다.
    yield (color, ep_seed, scene, aside). aside는 치우거나 통에 넣은 공(three면 None)."""
    ep_seed = seed
    while any(per_color[c] < n_per_color for c in COLORS):
        rng = np.random.default_rng([ep_seed, 5])
        scene = str(rng.choice(list(SCENE_P), p=list(SCENE_P.values())))
        aside = None if scene == "three" else COLORS[rng.integers(len(COLORS))]
        for color in COLORS:
            if color != aside and per_color[color] < n_per_color:
                yield color, ep_seed, scene, aside
        ep_seed += 1


def collect(
    repo_id, root, n_per_color, seed=0, correction=False, same_layout=False,
    randomize=False, mode_mix=False, scene_mix=False, modes=MODES, aux_labels=False,
):
    """same_layout=True면 배치(seed) 하나마다 세 색 명령을 모두 수행해, 같은 장면에 다른 명령이 붙은 시연을 만든다.
    randomize: 조명·배경·공 색 무작위화. mode_mix: 에피소드마다 실행 방식(기본/보정/잘못된 공 접근)을 1/3씩.
    scene_mix: 배치마다 장면 구성(three/two/prebinned)을 고른다(same_layout 필요).
    modes: mode_mix가 고를 실행 방식 목록(기본은 세 가지 모두).
    aux_labels: 프레임마다 명령한 색 공의 (x, y)를 target_xy로 기록(보조 학습 정답 전용, 정책 입력 아님)."""
    if any(m not in MODES for m in modes):
        raise ValueError(f"unknown mode in {modes}")
    if scene_mix and not same_layout:
        raise ValueError("scene_mix requires same_layout")
    features = dict(FEATURES)
    if aux_labels:
        features["target_xy"] = {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}
    ds = LeRobotDataset.create(repo_id, FPS, features, root=root, robot_type="so_follower", use_videos=True)
    env = BallSortEnv(render=True)
    expert = ScriptedExpert(env, seed=seed, correction=correction)
    mode_rng = np.random.default_rng([seed, 7])
    saved, discarded, per_color = 0, 0, Counter()
    per_scene, per_mode = Counter(), Counter()
    try:
        if scene_mix:
            schedule = _scene_mix_schedule(per_color, n_per_color, seed)
        elif same_layout:
            schedule = ((c, s, "three", None) for c, s in _same_layout_schedule(per_color, n_per_color, seed))
        else:
            schedule = ((c, s, "three", None) for c, s in _per_color_schedule(per_color, n_per_color, seed))
        for color, ep_seed, scene, aside in schedule:
            options = {"target": color}
            if randomize:
                options["randomize"] = True
            if scene != "three":
                options.update(scene=scene, aside=aside)
            obs, info = env.reset(seed=ep_seed, options=options)
            mode = "correction" if correction else "default"
            if mode_mix:
                mode = modes[mode_rng.integers(len(modes))]
                expert.correction = mode == "correction"
                expert.wrong_approach = mode == "wrong_approach"
            expert.reset()
            for _ in range(MAX_STEPS):
                executed, clean = expert.act()
                frame = {
                    "observation.state": obs["observation.state"],
                    "action": clean.astype(np.float32),
                    "observation.images.top": obs["observation.images.top"],
                    "observation.images.front": obs["observation.images.front"],
                    "task": info["task"],
                }
                if aux_labels:
                    frame["target_xy"] = env.ball_pos(env.target)[:2].astype(np.float32)
                ds.add_frame(frame)
                obs, _, terminated, truncated, info = env.step(executed)
                if terminated or truncated:
                    break
            if info["success"]:
                ds.save_episode()
                saved += 1
                per_color[color] += 1
                per_scene[scene] += 1
                per_mode[mode] += 1
                print(f"[{saved}] {color} saved (seed {ep_seed}, scene {scene}, mode {mode})", flush=True)
            else:
                ds.clear_episode_buffer()
                discarded += 1
    finally:
        ds.finalize()
        env.close()
    return {
        "saved": saved,
        "discarded": discarded,
        "per_color": dict(per_color),
        "per_scene": dict(per_scene),
        "per_mode": dict(per_mode),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="local/so101_ballsort_sim")
    ap.add_argument("--root", default="data/so101_ballsort_sim")
    ap.add_argument("--n-per-color", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correction", action="store_true", help="보정 시연 모드(빗나갔다가 올바른 자세로 보정)")
    ap.add_argument("--same-layout", action="store_true", help="같은 배치에 세 색 명령을 모두 수집")
    ap.add_argument("--randomize", action="store_true", help="조명·배경·공 색 무작위화")
    ap.add_argument("--mode-mix", action="store_true", help="에피소드마다 기본/보정/잘못된 공 접근을 1/3씩")
    ap.add_argument("--aux-labels", action="store_true", help="프레임마다 목표 공 (x, y)를 target_xy로 기록")
    ap.add_argument("--modes", default=",".join(MODES), help="--mode-mix가 고를 실행 방식(쉼표 구분)")
    ap.add_argument("--scene-mix", action="store_true", help="배치마다 three 60%%, two 20%%, prebinned 20%% (--same-layout 필요)")
    args = ap.parse_args()
    stats = collect(
        args.repo_id, args.root, args.n_per_color, args.seed, args.correction, args.same_layout,
        args.randomize, args.mode_mix, args.scene_mix, tuple(args.modes.split(",")), args.aux_labels,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
