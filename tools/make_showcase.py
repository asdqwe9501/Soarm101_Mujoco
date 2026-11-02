"""포트폴리오용 시연 영상: 최종 구성(IRL v2 + 색 가드 v2)으로 정면·손목 영상을 나란히 놓고 명령문·상태 자막을 넣는다.

python tools/make_showcase.py --out portfolio/media

- counterfactual: 같은 배치에서 명령만 바꿔 세 색을 각각 집는다(세 화면을 나란히).
- sequential: 한 장면에서 세 명령을 연달아 수행한다(처음 듣는 표현 포함).
- guard: 공이 가까운 배치에서 가드가 틀린 색 잡기를 거부하는 장면(처음 20초).
정책은 확률적이므로 성공한 시행을 고른다(성능 수치는 벤치마크 결과를 따른다). mp4와 GIF(축소, 2배속)를 저장한다.
"""
import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.env import FPS, MAX_STEPS, BallSortEnv  # noqa: E402

CK = "outputs/train/sft_aux2_irl/pretrained_model"
ACTOR = "outputs/irl/irl2/last.pt"
W, H = 320, 240
BAR = 44
try:
    FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
    SMALL = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def panel(obs, title, status, color=(255, 255, 255)):
    """정면 | 손목 영상(각 320×240) + 위쪽 자막 막대."""
    front = Image.fromarray(obs["observation.images.front"]).resize((W, H))
    wrist = Image.fromarray(obs["observation.images.top"]).resize((W, H))
    img = Image.new("RGB", (2 * W, H + BAR), (20, 20, 20))
    img.paste(front, (0, BAR))
    img.paste(wrist, (W, BAR))
    d = ImageDraw.Draw(img)
    d.text((8, 4), title, font=FONT, fill=(255, 255, 255))
    d.text((8, 24), status, font=SMALL, fill=color)
    d.text((W - 48, BAR + 4), "front", font=SMALL, fill=(255, 255, 255))
    d.text((2 * W - 48, BAR + 4), "wrist", font=SMALL, fill=(255, 255, 255))
    return np.asarray(img)


def run(env, policy, obs, title, frames, guard_note=True):
    """명령 하나를 끝까지 실행하며 프레임을 쌓는다. 반환: (성공, 마지막 obs, 거부 횟수)."""
    policy.reset(env)
    n_checks, note, note_until = 0, "", -1
    info = {}
    for t in range(MAX_STEPS):
        a = policy.act(obs)
        checks = policy.stats["checks"]
        if len(checks) > n_checks:
            n_checks = len(checks)
            c = checks[-1]
            seen = c["seen"] or "unknown"
            note = (f"color guard: saw {seen} -> REJECTED, retrying" if c["rejected"]
                    else f"color guard: saw {seen} -> confirmed, grasping")
            note_until = t + 45
        obs, _, term, trunc, info = env.step(a)
        status = note if (guard_note and t <= note_until) else f"t = {t / FPS:4.1f} s"
        col = (255, 120, 120) if "REJECTED" in status else (140, 230, 140) if "confirmed" in status else (200, 200, 200)
        frames.append(panel(obs, title, status, col))
        if term or trunc:
            break
    ok = bool(info.get("success"))
    rej = policy.stats["rejections"]
    end = ("SUCCESS: in the box, back home" if ok
           else f"time out - wrong ball never grasped ({rej} rejections)" if rej else "failed")
    frames += [panel(obs, title, end, (140, 230, 140) if ok else (255, 120, 120))] * FPS
    return ok, obs, rej


def save(frames, out, name):
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out / f"{name}.mp4", frames, fps=FPS, macro_block_size=1)
    k = 3 if frames[0].shape[0] < 400 else 4  # 세 화면을 쌓은 영상은 프레임을 더 줄인다
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::k]]
    imageio.mimsave(out / f"{name}.gif", small, duration=1000 * k / FPS / 2, loop=0)  # 2배속
    print("saved", name, len(frames), "frames", flush=True)


def counterfactual(env, policy, out, seed, tries, seeds=20):
    """같은 배치에서 세 명령. 세 화면을 위아래로 쌓는다. 세 명령이 모두 성공한 배치만 저장한다."""
    cmds = [("red", "Grab the red ball."), ("black", "Grab the black ball."), ("sky blue", "Grab the blue ball.")]
    for s in range(seed, seed + seeds):
        clips = []
        for color, cmd in cmds:
            for _ in range(tries):
                frames = []
                obs, _ = env.reset(seed=s, options={"target": color, "command": cmd})
                ok, _, _ = run(env, policy, obs, f'Same scene, command: "{cmd}"', frames)
                if ok:
                    break
            if not ok:
                break
            clips.append(frames)
        print("counterfactual seed", s, len(clips), "/ 3", flush=True)
        if len(clips) == 3:
            n = max(len(c) for c in clips)
            clips = [c + [c[-1]] * (n - len(c)) for c in clips]
            save([np.concatenate([c[i] for c in clips], 0) for i in range(n)], out, "counterfactual_same_scene")
            return
    print("no all-success counterfactual scene")


def sequential(env, policy, out, seed, tries, seeds=20):
    """한 장면에서 세 명령을 연달아. 세 명령이 모두 성공한 시행만 저장한다."""
    cmds = [("black", "Grab the black ball."), ("sky blue", "bring the blue one to the box"),
            ("red", "lift the red ball and drop it in the box")]
    for s in range(seed, seed + seeds):
        for _ in range(tries):
            frames, ok_all = [], True
            for i, (color, cmd) in enumerate(cmds):
                opts = {"target": color, "command": cmd}
                if i:
                    opts["keep_scene"] = True
                obs, _ = env.reset(seed=s if i == 0 else None, options=opts)
                ok, _, _ = run(env, policy, obs, f'Command {i + 1}/3: "{cmd}"', frames)
                ok_all &= ok
                if not ok:
                    break
            print("sequential seed", s, ok_all, flush=True)
            if ok_all:
                save(frames, out, "sequential_three_commands")
                return
    print("no all-success sequential run")


def guard(env, policy, out, seed, color, cmd, tries):
    for k in range(tries):
        frames = []
        obs, _ = env.reset(seed=seed, options={"target": color, "command": cmd})
        ok, _, rej = run(env, policy, obs, f'Balls close together, command: "{cmd}"', frames)
        print("guard try", k, ok, rej, flush=True)
        # 거부 뒤 재시도는 30초 안에 거의 회복하지 못한다(오류 분석 8절). 거부 장면을 보여 주고 20초로 자른다.
        if rej:
            save(frames[:600] + frames[-FPS:], out, "guard_rejects_wrong_color")
            return
    print("no guard rejection captured")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="portfolio/media")
    ap.add_argument("--only", nargs="*", default=["counterfactual", "sequential", "guard"])
    ap.add_argument("--tries", type=int, default=5)
    args = ap.parse_args()
    from policies.color_guard import ColorGuard
    from policies.dsrl import DSRLPolicyAdapter

    policy = ColorGuard(DSRLPolicyAdapter(CK, ACTOR))
    env = BallSortEnv(render=True)
    out = Path(args.out)
    if "counterfactual" in args.only:
        counterfactual(env, policy, out, seed=30003, tries=args.tries)
    if "sequential" in args.only:
        sequential(env, policy, out, seed=30011, tries=args.tries)
    if "guard" in args.only:
        # 오류 분석에서 세 모델이 모두 자주 틀린 배치 18(검정-파랑 간격 7.7cm)
        guard(env, policy, out, seed=30018, color="black", cmd="Grab the black ball.", tries=40)


if __name__ == "__main__":
    main()
