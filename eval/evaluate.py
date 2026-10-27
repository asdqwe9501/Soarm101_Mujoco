"""정책 성공률 평가. 정책 프로토콜: reset(env), act(obs) -> action(6,)"""

import argparse
import json
from collections import Counter
from pathlib import Path

import imageio
import numpy as np

from expert.scripted import ScriptedExpert
from sim.env import COLORS, FPS, MAX_STEPS, BallSortEnv

LIFTED_Z = 0.05


class ExpertPolicy:
    """특권 정보를 쓰는 시연 스크립트를 정책 프로토콜로 감싼 것 (평가기 검증용)."""

    def __init__(self, seed=0, dart_sigma=0.0):
        self.seed, self.dart_sigma = seed, dart_sigma
        self.expert = None

    def reset(self, env):
        if self.expert is None or self.expert.env is not env:
            self.expert = ScriptedExpert(env, seed=self.seed, dart_sigma=self.dart_sigma)
        self.expert.reset()

    def act(self, obs):
        return self.expert.act()[0]


def classify_failure(info, max_target_z):
    if info["wrong_in_bin"]:
        return "wrong_ball"
    if info["target_in_bin"]:
        return "return_fail"
    return "release_fail" if max_target_z > LIFTED_Z else "grasp_fail"


def run_episode(env, policy, obs, frames=None):
    policy.reset(env)
    max_z, info = 0.0, None
    for _ in range(MAX_STEPS):
        obs, _, terminated, truncated, info = env.step(policy.act(obs))
        max_z = max(max_z, env.ball_pos(env.target)[2])
        if frames is not None:
            frames.append(obs["observation.images.front"])
        if terminated or truncated:
            break
    return obs, info, max_z


def _options(color, commands, cmd_rng, **extra):
    """평가 명령문: canonical(예전 고정 문장), basic(Grab the X ball.), train(학습 배분), heldout(처음 듣는 표현)."""
    from sim.commands import command_for_tier, sample_command

    opts = {"target": color, **extra}
    if commands == "basic":
        opts["command"] = command_for_tier(color, "basic", cmd_rng)
    elif commands != "canonical":
        opts["command"] = sample_command(color, cmd_rng, split=commands)
    return opts


def _save_video(video_dir, name, frames, command):
    """영상 저장 + videos.txt에 파일 이름과 명령문을 한 줄로 남긴다."""
    Path(video_dir).mkdir(parents=True, exist_ok=True)
    imageio.mimsave(Path(video_dir) / name, frames, fps=FPS)
    with open(Path(video_dir) / "videos.txt", "a", encoding="utf-8") as f:
        f.write(f"{name}\t{command}\n")


def evaluate(policy, n_per_color, seed=0, render=False, video_dir=None, max_videos=10, commands="canonical",
             success_videos_per_color=2):
    """video_dir: 실패 영상 최대 max_videos개 + 성공 영상 색별 success_videos_per_color개를 저장한다."""
    env = BallSortEnv(render=render or video_dir is not None)
    wins, per_color, failures, n_videos = 0, Counter(), Counter(), 0
    success_saved = Counter()
    cmd_rng = np.random.default_rng(seed + 7)
    ep = 0
    for color in COLORS:
        for _ in range(n_per_color):
            obs, _ = env.reset(seed=seed + ep, options=_options(color, commands, cmd_rng))
            frames = [] if video_dir is not None else None
            _, info, max_z = run_episode(env, policy, obs, frames)
            if info["success"]:
                wins += 1
                per_color[color] += 1
                if frames and success_saved[color] < success_videos_per_color:
                    _save_video(video_dir, f"ep{ep:04d}_{color.replace(' ', '_')}_success.mp4", frames, env.task)
                    success_saved[color] += 1
            else:
                kind = classify_failure(info, max_z)
                failures[kind] += 1
                if frames and n_videos < max_videos:
                    _save_video(video_dir, f"ep{ep:04d}_{color.replace(' ', '_')}_{kind}.mp4", frames, env.task)
                    n_videos += 1
            ep += 1
    env.close()
    n = n_per_color * len(COLORS)
    return {
        "n": n,
        "success_rate": wins / n,
        "per_color": {c: per_color[c] / n_per_color for c in COLORS},
        "failures": {k: failures[k] for k in ["wrong_ball", "grasp_fail", "release_fail", "return_fail"]},
    }


def evaluate_sequential(policy, n_trials, seed=0, render=False, commands="canonical"):
    """공 3개를 무작위 순서로 연속 명령한다. 한 시험 안에서는 장면을 유지한다."""
    env = BallSortEnv(render=render)
    rng = np.random.default_rng(seed)
    cmd_rng = np.random.default_rng(seed + 7)
    all_ok, cmd_ok = 0, 0
    for t in range(n_trials):
        order = list(rng.permutation(COLORS))
        ok = True
        for i, color in enumerate(order):
            if i == 0:
                obs, _ = env.reset(seed=seed + t, options=_options(color, commands, cmd_rng))
            else:
                obs, _ = env.reset(options=_options(color, commands, cmd_rng, keep_scene=True))
            _, info, _ = run_episode(env, policy, obs)
            cmd_ok += info["success"]
            if not info["success"]:
                ok = False
                break
        all_ok += ok
    env.close()
    return {"n": n_trials, "all_success_rate": all_ok / n_trials, "per_command_rate": cmd_ok / (3 * n_trials)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["expert", "smolvla", "dsrl"], default="expert")
    ap.add_argument("--checkpoint", default=None, help="--policy smolvla/dsrl일 때 pretrained_model 경로")
    ap.add_argument("--actor", default=None, help="--policy dsrl일 때 SAC 체크포인트(.pt)")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--zero-state", action="store_true", help="상태를 0으로 학습한 체크포인트용")
    ap.add_argument("--n-per-color", type=int, default=50)
    ap.add_argument("--sequential", type=int, default=0, help="연속 명령 시험 횟수 (0이면 생략)")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--commands", choices=["canonical", "basic", "train", "heldout"], default="canonical",
                    help="명령문 종류 (새 모델은 basic/heldout 권장)")
    ap.add_argument("--guard", action="store_true", help="손목 카메라 색 확인 + 재시도(ColorGuard)로 감싼다")
    args = ap.parse_args()
    if args.policy == "smolvla":
        from policies.smolvla_policy import SmolVLAPolicyAdapter

        policy = SmolVLAPolicyAdapter(args.checkpoint, n_action_steps=args.n_action_steps, zero_state=args.zero_state)
    elif args.policy == "dsrl":
        from policies.dsrl import DSRLPolicyAdapter

        policy = DSRLPolicyAdapter(args.checkpoint, args.actor)
    else:
        policy = ExpertPolicy()
    if args.guard:
        from policies.color_guard import ColorGuard

        policy = ColorGuard(policy)
    render = args.policy in ("smolvla", "dsrl") or args.guard
    result = {"single": evaluate(policy, args.n_per_color, args.seed, render=render, video_dir=args.video_dir, commands=args.commands)}
    if args.sequential:
        result["sequential"] = evaluate_sequential(policy, args.sequential, args.seed, render=render, commands=args.commands)
    result["commands"] = args.commands
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
