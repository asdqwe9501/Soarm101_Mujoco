"""반복 실행 루프: 제자리 대기 → 명령 입력 → 작업 → 제자리 복귀 → 다음 명령.

시뮬레이션(루프 검증):
  python -m run.loop --backend sim --commands-file cmds.txt
실제 로봇:
  python -m run.loop --backend real --port COM5 --wrist-cam 1 --front-cam 0

- 정책: IRL v2(sft_aux2 복사본 + KL 노이즈 정책) + 색 가드(home 후퇴). --no-guard로 끌 수 있다.
- 정책 입력은 관절 상태(정규화 단위), 손목 영상(observation.images.top), 정면 영상(observation.images.front), 명령문뿐이다.
- 시작과 작업 사이에 SIM_HOME_NORM(시뮬레이션 학습 때의 제자리 자세)으로 천천히 옮긴다.
  실제 데이터셋의 제자리(HOME_NORM)와는 wrist_flex만 28 단위 다르다(calib.py).
- 작업 끝 판정(실제 로봇에는 정답 신호가 없으므로 행동만 본다):
  닫기(잡기) → 열기(놓기)를 거친 뒤 제자리 근처에 HOLD_STEPS 동안 머물면 끝. 또는 --max-steps 경과.
- 실제 로봇 안전: max_relative_target로 한 스텝 이동량을 제한한다. Ctrl+C로 언제든 멈추고 토크를 끈다.
- 색 가드의 ROI·HSV 임계값은 시뮬레이션 기준이다. 실제 카메라로 tools/guard_calibrate.py를 먼저 돌려 확인한다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from calib import SIM_HOME_NORM
from expert.scripted import min_jerk
from policies.color_guard import command_color
from rl.staged_reward import GRIP_CLOSE_CMD, GRIP_OPEN_CMD
from sim.env import FPS, HOME_TOL

CK = "outputs/train/sft_aux2_irl/pretrained_model"
ACTOR = "outputs/irl/irl2/last.pt"
HOME_STEPS = 60  # 제자리로 옮기는 데 2초
HOLD_STEPS = 15  # 놓은 뒤 제자리 근처에 0.5초 머물면 작업 끝
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class TaskCtx:
    """정책·가드의 reset(env)는 env.task(명령문)만 읽는다."""

    def __init__(self, task):
        self.task = task


class SimBackend:
    """BallSortEnv를 실제 로봇처럼 쓴다. 장면은 명령 사이에 유지한다(연속 명령 평가와 같음)."""

    realtime = False

    def __init__(self, seed=0):
        from sim.env import BallSortEnv

        self.env = BallSortEnv(render=True)
        self.obs, _ = self.env.reset(seed=seed)
        self.first = True

    def begin(self, task):
        color = command_color(task)
        if color is None or not self.env.ball_on_table(color):
            return False
        opts = {"target": color, "command": task}
        if not self.first:
            opts["keep_scene"] = True
        self.obs, _ = self.env.reset(options=opts)
        self.first = False
        return True

    def observe(self):
        return self.obs

    def send(self, action):
        self.obs, _, self.success, _, self.info = self.env.step(np.asarray(action, np.float32))

    def truth(self):
        """시뮬레이션 정답(루프 판정 검증용)."""
        return {"target_in_bin": bool(self.info["target_in_bin"]), "wrong_in_bin": bool(self.info["wrong_in_bin"]),
                "at_home": self.env.at_home()}

    def close(self):
        self.env.close()


class RealBackend:
    realtime = True

    def __init__(self, port, wrist_cam, front_cam, robot_id, max_relative_target):
        from lerobot.cameras.opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

        def cam(x):
            return OpenCVCameraConfig(index_or_path=int(x) if str(x).isdigit() else x, fps=FPS, width=640, height=480)

        cfg = SOFollowerRobotConfig(port=port, id=robot_id, max_relative_target=max_relative_target,
                                    cameras={"top": cam(wrist_cam), "front": cam(front_cam)})
        self.robot = SOFollower(cfg)
        self.robot.connect(calibrate=False)

    def begin(self, task):
        return command_color(task) is not None

    def observe(self):
        o = self.robot.get_observation()
        return {"observation.state": np.array([o[f"{j}.pos"] for j in JOINTS], np.float32),
                "observation.images.top": np.asarray(o["top"]), "observation.images.front": np.asarray(o["front"])}

    def send(self, action):
        self.robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, action)})

    def truth(self):
        return {}

    def close(self):
        self.robot.disconnect()


def tick(backend, t0, k):
    """실제 로봇이면 FPS에 맞춰 기다린다."""
    if backend.realtime:
        time.sleep(max(0.0, t0 + k / FPS - time.perf_counter()))


def move_home(backend):
    start = backend.observe()["observation.state"].copy()
    t0 = time.perf_counter()
    for i in range(HOME_STEPS):
        backend.send(min_jerk(start, SIM_HOME_NORM, (i + 1) / HOME_STEPS))
        tick(backend, t0, i + 1)


def near_home(state):
    return bool(np.all(np.abs(np.asarray(state[:5]) - SIM_HOME_NORM[:5]) < HOME_TOL))


def run_task(backend, policy, task, max_steps):
    """명령 하나를 실행한다. 반환: 판정 기록."""
    policy.reset(TaskCtx(task))
    closed = released = False
    hold, armed, steps = 0, False, 0
    reason = "max_steps"
    t0 = time.perf_counter()
    for k in range(max_steps):
        obs = backend.observe()
        a = policy.act(obs)
        backend.send(a)
        steps = k + 1
        if a[5] > GRIP_OPEN_CMD:
            armed = True
            released = released or closed  # 잡은 뒤 다시 열면 놓은 것
        elif armed and a[5] < GRIP_CLOSE_CMD:
            armed, closed = False, True
        hold = hold + 1 if released and near_home(backend.observe()["observation.state"]) else 0
        if hold >= HOLD_STEPS:
            reason = "done"
            break
        tick(backend, t0, k + 1)
    return {"task": task, "steps": steps, "seconds": round(steps / FPS, 1), "end": reason,
            "grasped": closed, "released": released}


def commands(args):
    if args.commands_file:
        for line in Path(args.commands_file).read_text(encoding="utf-8").splitlines():
            if line.strip():
                print(f"> {line.strip()}")
                yield line.strip()
        return
    while True:
        try:
            line = input("명령 입력 (빈 줄이나 q: 종료) > ").strip()
        except EOFError:
            return
        if not line or line.lower() == "q":
            return
        yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "real"], default="sim")
    ap.add_argument("--checkpoint", default=CK)
    ap.add_argument("--actor", default=ACTOR)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--max-steps", type=int, default=900, help="명령 하나의 최대 스텝(30fps 기준 30초)")
    ap.add_argument("--commands-file", default=None, help="한 줄에 명령 하나. 없으면 키보드로 입력")
    ap.add_argument("--log", default="outputs/loop/log.jsonl")
    ap.add_argument("--seed", type=int, default=0, help="sim 장면 seed")
    ap.add_argument("--port", default=None, help="real: 로봇 시리얼 포트 (예: COM5)")
    ap.add_argument("--wrist-cam", default=None, help="real: 손목 카메라 번호나 경로 (데이터셋의 'top')")
    ap.add_argument("--front-cam", default=None, help="real: 정면 카메라 번호나 경로 (데이터셋의 'front')")
    ap.add_argument("--robot-id", default="ty_follower_arm")
    ap.add_argument("--max-relative-target", type=float, default=10.0, help="real: 한 스텝 최대 이동량(정규화 단위)")
    args = ap.parse_args()

    if args.backend == "real":
        missing = [k for k in ("port", "wrist_cam", "front_cam") if getattr(args, k) is None]
        if missing:
            ap.error(f"--backend real에는 {', '.join('--' + m.replace('_', '-') for m in missing)}가 필요합니다")
        backend = RealBackend(args.port, args.wrist_cam, args.front_cam, args.robot_id, args.max_relative_target)
    else:
        backend = SimBackend(args.seed)

    from policies.dsrl import DSRLPolicyAdapter

    policy = DSRLPolicyAdapter(args.checkpoint, args.actor)
    if not args.no_guard:
        from policies.color_guard import ColorGuard

        policy = ColorGuard(policy)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    try:
        move_home(backend)
        for task in commands(args):
            if not backend.begin(task):
                print(f"  명령에서 공 색(red / black / blue)을 찾지 못했거나 그 공이 책상 위에 없습니다: {task!r}")
                continue
            rec = run_task(backend, policy, task, args.max_steps)
            rec.update(backend.truth())
            move_home(backend)
            print(f"  끝: {rec['end']} ({rec['seconds']}초), 잡음 {rec['grasped']}, 놓음 {rec['released']}"
                  + (f", [sim 정답] 통 안 {rec['target_in_bin']}, 다른 공 {rec['wrong_in_bin']}" if "target_in_bin" in rec else ""))
            with open(args.log, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        print("\n중단")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
