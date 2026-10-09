"""정답 좌표 + IK로 관절 궤적을 만드는 시연 스크립트.

제자리 → 공 위 → 하강 → 닫기 → 들기 → 통 위 → 열기 → 제자리
"""

import numpy as np

from calib import SIM_HOME_NORM
from expert.ik import IK
from sim.env import FPS

GRIP_OPEN = 60.0
GRIP_CLOSED = 5.0
PRE_Z = 0.07        # 공 위 대기 높이 (gripperframe 기준)
GRASP_Z = 0.012     # 집는 높이 (공 중심 0.02보다 약간 아래)
LIFT_Z = 0.08
RELEASE_Z = 0.09  # 공 아래면 ≈0.08 > 통 높이 0.07
# 각 구간 길이 [s]: 공 위, 하강, 닫기, 들기, 통 위, 열기, 제자리, 대기
DURATIONS = [3.0, 1.5, 1.0, 1.5, 3.0, 1.0, 3.5, 1.0]
GRASP_OFFSET = 0.008  # 고정 집게가 공을 밀지 않도록 gripperframe을 고정 집게 쪽으로 이동
# 로봇 바로 앞의 공은 그리퍼를 수직으로 내리면 어깨와 부딪히므로 손끝을 몸쪽으로 기울인다
NEAR_R = 0.11       # 팬 축으로부터 이 거리보다 가까우면 기울임 [m]
TILT_PER_M = 1000.0  # 1cm 가까워질 때마다 10도
MAX_TILT = 30.0     # [deg]
XY_JITTER = 0.005
DART_RHO = 0.9
# 새 노이즈를 넣지 않는 구간(하강, 닫기). 공이 좁은 집게 가운데에 오도록 남은 노이즈는 DART_RHO로 감쇠한다
QUIET_SEGMENTS = (1, 2)


def min_jerk(a, b, s):
    s = np.clip(s, 0.0, 1.0)
    w = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a + (b - a) * w


class ScriptedExpert:
    def __init__(self, env, seed=0, dart_sigma=0.5):
        self.env = env
        self.ik = IK(env.model)
        self._gripper_body = env.model.body("gripper").id
        self.ik._fk(np.zeros(6))
        self._pan_xy = self.ik.data.xanchor[env.model.joint("shoulder_pan").id][:2].copy()
        self.rng = np.random.default_rng(seed)
        self.dart_sigma = dart_sigma
        self.traj = None
        self.t = 0

    @property
    def n_steps(self):
        return len(self.traj)

    def _pose(self, xyz, q_prev_rad, grip, down=None):
        q, _, _ = self.ik.solve(xyz, q_prev_rad, down=down)
        v = self.env.calib.rad_to_norm(q)
        v[5] = grip
        return v, q

    def _jaw_dir(self, q_rad):
        """gripperframe → 고정 집게 방향(수평 단위벡터)."""
        self.ik._fk(q_rad)
        u = -self.ik.data.xmat[self._gripper_body].reshape(3, 3)[:2, 0]
        return u / np.linalg.norm(u)

    def _down(self, xy):
        """손끝 방향: 기본은 연직 아래, 팬 축에 가까우면 몸쪽으로 기울인다."""
        d = np.asarray(xy) - self._pan_xy
        r = np.linalg.norm(d)
        th = np.radians(np.clip((NEAR_R - r) * TILT_PER_M, 0.0, MAX_TILT))
        return np.array([*(-np.sin(th) * d / r), -np.cos(th)])

    def reset(self):
        env, rng = self.env, self.rng
        ball = env.ball_pos(env.target)
        bin_ = env.bin_pos()
        jit = lambda: rng.uniform(-XY_JITTER, XY_JITTER, 2)
        start = env.joint_norm()
        q0 = env.calib.norm_to_rad(SIM_HOME_NORM)

        down = self._down(ball[:2])
        _, q = self._pose([*ball[:2], PRE_Z], q0, GRIP_OPEN, down)
        g = ball[:2] + GRASP_OFFSET * self._jaw_dir(q)
        pre, q = self._pose([*(g + jit()), PRE_Z], q0, GRIP_OPEN, down)
        grasp, q = self._pose([*g, GRASP_Z], q, GRIP_OPEN, down)
        closed = grasp.copy()
        closed[5] = GRIP_CLOSED
        lift, q = self._pose([*g, LIFT_Z], q, GRIP_CLOSED, down)
        above_bin, q = self._pose([*(bin_[:2] + jit()), RELEASE_Z], q, GRIP_CLOSED)
        release = above_bin.copy()
        release[5] = GRIP_OPEN
        home = SIM_HOME_NORM.copy()

        keys = [start, pre, grasp, closed, lift, above_bin, release, home, home]
        scale = rng.uniform(0.85, 1.15, len(DURATIONS))
        traj, gate = [], []
        for i, (a, b, d, k) in enumerate(zip(keys[:-1], keys[1:], DURATIONS, scale)):
            n = max(2, int(round(d * k * FPS)))
            traj += [min_jerk(a, b, j / (n - 1)) for j in range(n)]
            gate += [0.0 if i in QUIET_SEGMENTS else 1.0] * n
        self.traj = np.array(traj)
        self._gate = np.array(gate)
        self.t = 0
        self._noise = np.zeros(6)

    def act(self):
        i = min(self.t, len(self.traj) - 1)
        clean = self.traj[i].copy()
        self.t += 1
        eps = self.rng.normal(0.0, self.dart_sigma * np.sqrt(1 - DART_RHO**2), 6) * self._gate[i]
        self._noise = DART_RHO * self._noise + eps
        self._noise[5] = 0.0  # 그리퍼에는 노이즈를 넣지 않음
        return clean + self._noise, clean
