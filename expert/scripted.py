"""정답 좌표 + IK로 관절 궤적을 만드는 시연 스크립트.

제자리 → 공 위 → 하강 → 닫기 → 들기 → 통 위 → 열기 → 제자리
"""

import numpy as np

from calib import SIM_HOME_NORM
from expert.ik import IK
from sim.env import COLORS, FPS

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
# 보정 모드: 공 위로 갈 때 일부러 빗나간 지점으로 간 뒤 멈췄다가 올바른 자세로 보정한다
CORR_R = (0.02, 0.04)        # 빗나가는 거리 [m]
CORR_DWELL = (0.3, 0.6)      # 빗나간 지점에서 멈추는 시간 [s]
CORR_DURATION = 1.2          # 보정 이동 시간 [s] (DURATIONS처럼 ±15%)
CORR_MAX_STEP = 3.0          # 빗나간 구간 라벨의 한 스텝 최대 이동 [정규화 단위]. 평소 최고 속도(≈2)보다 약간 크게
CORR_BALL_GAP = 0.04         # 빗나간 지점과 다른 공 사이 최소 수평 거리
CORR_BIN_GAP = 0.09          # 빗나간 지점과 통 중심 사이 최소 수평 거리


def min_jerk(a, b, s):
    s = np.clip(s, 0.0, 1.0)
    w = 10 * s**3 - 15 * s**4 + 6 * s**5
    return a + (b - a) * w


class ScriptedExpert:
    def __init__(self, env, seed=0, dart_sigma=0.5, correction=False):
        self.env = env
        self.ik = IK(env.model)
        self._gripper_body = env.model.body("gripper").id
        self.ik._fk(np.zeros(6))
        self._pan_xy = self.ik.data.xanchor[env.model.joint("shoulder_pan").id][:2].copy()
        self.rng = np.random.default_rng(seed)
        self.dart_sigma = dart_sigma
        self.correction = correction
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
        pre_xy = g + jit()
        pre, q = self._pose([*pre_xy, PRE_Z], q0, GRIP_OPEN, down)
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
        off_mask = np.zeros(len(traj), dtype=bool)
        off = self._offset_pose(pre_xy, bin_, q0) if self.correction else None
        if off is not None:
            # 첫 구간(제자리 → 공 위)을 [제자리 → 빗나간 지점, 멈춤, 보정 이동]으로 바꾼다.
            # 빗나간 구간과 멈춤 동안은 DART 노이즈를 넣지 않는다(빗나감 자체가 교란).
            n0 = max(2, int(round(DURATIONS[0] * scale[0] * FPS)))
            n_dwell = int(round(rng.uniform(*CORR_DWELL) * FPS))
            n_corr = max(2, int(round(CORR_DURATION * rng.uniform(0.85, 1.15) * FPS)))
            head = [min_jerk(start, off, j / (n0 - 1)) for j in range(n0)] + [off.copy()] * n_dwell
            corr = [min_jerk(off, pre, j / (n_corr - 1)) for j in range(n_corr)]
            traj = head + corr + traj[n0:]
            gate = [0.0] * len(head) + [1.0] * n_corr + gate[n0:]
            off_mask = np.zeros(len(traj), dtype=bool)
            off_mask[: len(head)] = True
        self.traj = np.array(traj)
        self._gate = np.array(gate)
        self._off_mask = off_mask
        self.pre_pose = pre
        self.t = 0
        self._noise = np.zeros(6)

    def _offset_pose(self, pre_xy, bin_, q0):
        """올바른 공 위 지점(pre_xy)에서 무작위 방향으로 CORR_R만큼 빗나간 공 위 자세. 못 찾으면 None."""
        env, rng = self.env, self.rng
        others = [env.ball_pos(c)[:2] for c in COLORS if c != env.target]
        for _ in range(50):
            th = rng.uniform(0, 2 * np.pi)
            xy = pre_xy + rng.uniform(*CORR_R) * np.array([np.cos(th), np.sin(th)])
            if np.hypot(*(xy - bin_[:2])) < CORR_BIN_GAP:
                continue
            if any(np.hypot(*(xy - o)) < CORR_BALL_GAP for o in others):
                continue
            q, err, _ = self.ik.solve([*xy, PRE_Z], q0, down=self._down(xy))
            if err < 0.003:
                v = self.env.calib.rad_to_norm(q)
                v[5] = GRIP_OPEN
                return v
        return None

    def act(self):
        i = min(self.t, len(self.traj) - 1)
        clean = self.traj[i].copy()
        self.t += 1
        eps = self.rng.normal(0.0, self.dart_sigma * np.sqrt(1 - DART_RHO**2), 6) * self._gate[i]
        self._noise = DART_RHO * self._noise + eps
        self._noise[5] = 0.0  # 그리퍼에는 노이즈를 넣지 않음
        executed = clean + self._noise
        if self._off_mask[i]:
            # 빗나간 구간 라벨: 현재 관절값에서 올바른 공 위 자세 쪽으로 속도 제한을 둔 한 스텝
            # 한계는 관절마다 max(CORR_MAX_STEP, 실행 명령이 현재 관절값보다 앞선 양): 라벨이 실행 명령보다
            # 느리게 목표로 가지 않으므로 관절마다 라벨이 실행 명령보다 올바른 자세에 가깝거나 같다.
            cur = self.env.joint_norm()
            lim = np.maximum(CORR_MAX_STEP, np.abs(executed[:5] - cur[:5]))
            clean[:5] = cur[:5] + np.clip(self.pre_pose[:5] - cur[:5], -lim, lim)
            clean[5] = GRIP_OPEN
        return executed, clean
