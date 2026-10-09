"""실제 SO101(LeRobot 정규화 단위)과 MuJoCo 관절각(rad) 사이의 변환."""

import json
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FEATURE_NAMES = [f"{j}.pos" for j in JOINTS]
DEFAULT_CALIB = (
    Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/ty_follower_arm.json"
)

# 실제 데이터셋 에피소드 시작/끝 평균 (정규화 단위)
HOME_NORM = np.array([-2.8, 0.1, 93.7, 71.7, -0.7, 1.5])

TICKS_PER_REV = 4096
MID_TICK = 2047
# 2047 tick(= 실제 캘리브레이션 중앙 = 제자리 자세)에 해당하는 MuJoCo 관절각.
# tools/align_overlay.py로 실제 영상과 겹쳐 보며 보정한다.
MID_RAD = np.array([0.0, -1.70, 1.54, 1.20, 0.0, 0.0])


class Calib:
    def __init__(self, path=DEFAULT_CALIB, mid_rad=MID_RAD):
        c = json.loads(Path(path).read_text())
        self.lo = np.array([c[j]["range_min"] for j in JOINTS], dtype=float)
        self.hi = np.array([c[j]["range_max"] for j in JOINTS], dtype=float)
        self.mid_rad = np.asarray(mid_rad, dtype=float)

    def _norm_to_ticks(self, v):
        v = np.asarray(v, dtype=float)
        span = self.hi - self.lo
        t = np.empty_like(v)
        t[..., :5] = (v[..., :5] + 100) / 200 * span[:5] + self.lo[:5]
        t[..., 5] = v[..., 5] / 100 * span[5] + self.lo[5]
        return t

    def _ticks_to_norm(self, t):
        span = self.hi - self.lo
        v = np.empty_like(t)
        v[..., :5] = (t[..., :5] - self.lo[:5]) / span[:5] * 200 - 100
        v[..., 5] = (t[..., 5] - self.lo[5]) / span[5] * 100
        return v

    def norm_to_rad(self, v):
        return self.mid_rad + (self._norm_to_ticks(v) - MID_TICK) * 2 * np.pi / TICKS_PER_REV

    def rad_to_norm(self, q):
        t = (np.asarray(q, dtype=float) - self.mid_rad) * TICKS_PER_REV / (2 * np.pi) + MID_TICK
        return self._ticks_to_norm(t)
