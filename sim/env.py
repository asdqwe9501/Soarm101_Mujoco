"""SO101 색깔 공 분류 Gymnasium 환경. MuJoCo를 직접 다루는 유일한 모듈."""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from calib import SIM_HOME_NORM, Calib

SCENE_XML = Path(__file__).resolve().parent / "scene.xml"
COLORS = ("red", "black", "sky blue")
FPS = 30
N_SUBSTEPS = 17
MAX_STEPS = 900
HOME_TOL = 5.0
HOME_NOISE = 1.0

BIN_CENTER = (0.165, -0.205)
BIN_JITTER = 0.02
BIN_R_IN = 0.045
BIN_H = 0.07

BALL_RADIUS = 0.02
BALL_X = (0.12, 0.28)
BALL_Y = (-0.10, 0.20)
BALL_R_MAX = 0.28
BALL_MIN_GAP = 0.07
BALL_BIN_GAP = 0.12
BALL_REST_TIP_GAP = 0.06  # 제자리 자세 그리퍼 끝과의 최소 거리

# 실제 데이터셋의 카메라 키는 이름과 반대다: images.top = 손목 카메라, images.front = 위쪽 카메라
TOP_CAM = "wrist_cam"
FRONT_CAM = "top"  # MuJoCo 위쪽 카메라 이름
IMG_H, IMG_W = 480, 640


def task_text(color: str) -> str:
    return f"grab the {color} ball and put it in the box"


class BallSortEnv(gym.Env):
    metadata = {"render_fps": FPS}

    def __init__(self, render: bool = True, calib: Calib | None = None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.model.opt.timestep = 1.0 / (FPS * N_SUBSTEPS)
        self.data = mujoco.MjData(self.model)
        self.calib = calib or Calib()
        self.render_images = render
        self._renderer = mujoco.Renderer(self.model, IMG_H, IMG_W) if render else None

        self._ball_joint = {c: "ball_" + c.replace(" ", "_") for c in COLORS}
        self._ball_adr = {
            c: self.model.jnt_qposadr[self.model.joint(j).id] for c, j in self._ball_joint.items()
        }
        self._bin_id = self.model.body("bin").id
        self._site_id = self.model.site("gripperframe").id
        self._ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        self.action_space = gym.spaces.Box(-100.0, 100.0, (6,), np.float32)
        spaces = {"observation.state": gym.spaces.Box(-100.0, 100.0, (6,), np.float32)}
        if render:
            img = gym.spaces.Box(0, 255, (IMG_H, IMG_W, 3), np.uint8)
            spaces["observation.images.top"] = img
            spaces["observation.images.front"] = img
        self.observation_space = gym.spaces.Dict(spaces)

        self.target = None
        self.task = None
        self._steps = 0
        self._target_rewarded = False
        self._wrong_penalized = set()
        self._in_bin_at_reset = set()

    # ---------- 특권 정보 (시연 스크립트, 보상, 평가 전용) ----------
    def ball_pos(self, color: str) -> np.ndarray:
        a = self._ball_adr[color]
        return self.data.qpos[a : a + 3].copy()

    def bin_pos(self) -> np.ndarray:
        return self.model.body_pos[self._bin_id].copy()

    def ball_in_bin(self, color: str) -> bool:
        p, b = self.ball_pos(color), self.bin_pos()
        return bool(np.hypot(p[0] - b[0], p[1] - b[1]) < BIN_R_IN and p[2] < BIN_H)

    def joint_norm(self) -> np.ndarray:
        return self.calib.rad_to_norm(self.data.qpos[:6])

    def at_home(self) -> bool:
        return bool(np.all(np.abs(self.joint_norm() - SIM_HOME_NORM) < HOME_TOL))

    # ---------- 내부 ----------
    def _set_ball(self, color, xy):
        a = self._ball_adr[color]
        self.data.qpos[a : a + 3] = [xy[0], xy[1], BALL_RADIUS]
        self.data.qpos[a + 3 : a + 7] = [1, 0, 0, 0]

    def _sample_layout(self, rest_tip_xy):
        rng = self.np_random
        b = np.array(BIN_CENTER) + rng.uniform(-BIN_JITTER, BIN_JITTER, 2)
        self.model.body_pos[self._bin_id][:2] = b
        while True:
            pts = []
            for _ in range(200):
                p = np.array([rng.uniform(*BALL_X), rng.uniform(*BALL_Y)])
                if (
                    np.hypot(*p) <= BALL_R_MAX
                    and np.hypot(*(p - b)) >= BALL_BIN_GAP
                    and np.hypot(*(p - rest_tip_xy)) >= BALL_REST_TIP_GAP
                    and all(np.hypot(*(p - q)) >= BALL_MIN_GAP for q in pts)
                ):
                    pts.append(p)
                    if len(pts) == len(COLORS):
                        return pts

    def _obs(self):
        obs = {"observation.state": self.joint_norm().astype(np.float32)}
        if self.render_images:
            for key, cam in (("observation.images.top", TOP_CAM), ("observation.images.front", FRONT_CAM)):
                self._renderer.update_scene(self.data, camera=cam)
                obs[key] = self._renderer.render().copy()
        return obs

    def _info(self, success=False):
        return {
            "task": self.task,
            "target": self.target,
            "success": success,
            "target_in_bin": self.ball_in_bin(self.target),
            "wrong_in_bin": any(
                self.ball_in_bin(c)
                for c in COLORS
                if c != self.target and c not in self._in_bin_at_reset
            ),
        }

    # ---------- Gymnasium API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if not options.get("keep_scene"):
            mujoco.mj_resetData(self.model, self.data)
            home = SIM_HOME_NORM + self.np_random.uniform(-HOME_NOISE, HOME_NOISE, 6)
            q = self.calib.norm_to_rad(home)
            self.data.qpos[:6] = q
            self.data.ctrl[:6] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
            mujoco.mj_kinematics(self.model, self.data)
            rest_tip = self.data.site_xpos[self._site_id][:2].copy()
            pts = self._sample_layout(rest_tip)
            # 뽑힌 순서대로 색을 배정하면 특정 색이 한쪽에 치우친다. 색-위치 배정을 매번 섞는다.
            order = self.np_random.permutation(len(COLORS))
            for c, p in zip(COLORS, (pts[i] for i in order)):
                self._set_ball(c, p)
            mujoco.mj_forward(self.model, self.data)
            for _ in range(N_SUBSTEPS * 6):  # 0.2초 정착
                mujoco.mj_step(self.model, self.data)

        remaining = [c for c in COLORS if not self.ball_in_bin(c)]
        if not remaining:
            raise RuntimeError("all balls are already in the bin")
        target = options.get("target")
        if target is None:
            target = remaining[self.np_random.integers(len(remaining))]
        if target not in remaining:
            raise ValueError(f"target {target!r} is not on the table")
        self.target, self.task = target, task_text(target)
        self._steps = 0
        self._target_rewarded = False
        self._in_bin_at_reset = {c for c in COLORS if self.ball_in_bin(c)}
        self._wrong_penalized = set()
        return self._obs(), self._info()

    def step(self, action):
        q = self.calib.norm_to_rad(np.asarray(action, dtype=float))
        self.data.ctrl[:6] = np.clip(q, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_step(self.model, self.data, nstep=N_SUBSTEPS)
        self._steps += 1

        reward = 0.0
        if self.ball_in_bin(self.target) and not self._target_rewarded:
            self._target_rewarded = True
            reward += 1.0
        for c in COLORS:
            if (
                c != self.target
                and c not in self._in_bin_at_reset
                and c not in self._wrong_penalized
                and self.ball_in_bin(c)
            ):
                self._wrong_penalized.add(c)
                reward -= 1.0

        info = self._info()
        success = info["target_in_bin"] and not info["wrong_in_bin"] and self.at_home()
        info["success"] = success
        if success:
            reward += 1.0
        truncated = self._steps >= MAX_STEPS and not success
        return self._obs(), reward, success, truncated, info

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
