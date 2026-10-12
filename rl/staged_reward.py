"""사용자가 정의한 단계별 보상: 맞는 공에 잡기 시도 +1, 잡음 +2, 통에 넣음 +3, 제자리 복귀 +1.

잡기 시도 = 그리퍼 명령이 GRIP_OPEN_CMD를 넘었다가 처음 GRIP_CLOSE_CMD 아래로 내려간 순간.
그 순간 그리퍼 끝 근처(수평 ATTEMPT_RADIUS, 높이 ATTEMPT_MAX_Z 이하)에서 가장 가까운 공이 노린 공이다.
"""

import numpy as np

from sim.env import COLORS

GRIP_OPEN_CMD = 40.0
GRIP_CLOSE_CMD = 20.0
ATTEMPT_MAX_Z = 0.07
ATTEMPT_RADIUS = 0.035
LIFT_Z = 0.05
ATTEMPT_TIMEOUT_STEPS = 360

R_ATTEMPT = 1.0
R_LIFT = 2.0
R_BIN = 3.0
R_HOME = 1.0
R_WRONG = -1.0
R_MISS = -0.5


class StagedReward:
    def __init__(self, env):
        self.env = env
        self.site = env.model.site("gripperframe").id

    def reset(self):
        self.steps = 0
        self.armed = False  # 그리퍼를 연 뒤 아직 닫지 않음
        self.first_attempt = None
        self.attempt_color = None
        self.stage = 0
        self.candidates = [c for c in COLORS if not self.env.ball_in_bin(c)]

    def _aimed_ball(self):
        tip = self.env.data.site_xpos[self.site]
        if tip[2] > ATTEMPT_MAX_Z:
            return None
        d = {c: np.hypot(*(self.env.ball_pos(c)[:2] - tip[:2])) for c in self.candidates}
        c = min(d, key=d.get)
        return c if d[c] <= ATTEMPT_RADIUS else None

    def _info(self, end_reason=None):
        return {
            "first_attempt": self.first_attempt,
            "attempt_color": self.attempt_color,
            "stage": self.stage,
            "success": end_reason == "success",
            "end_reason": end_reason,
        }

    def step(self, action):
        env, target = self.env, self.env.target
        self.steps += 1
        cmd = float(action[5])
        reward = 0.0

        # 잡기 시도 판정
        if cmd > GRIP_OPEN_CMD:
            self.armed = True
        elif self.armed and cmd < GRIP_CLOSE_CMD:
            self.armed = False
            aimed = self._aimed_ball()
            if aimed is not None and aimed != target:
                if self.first_attempt is None:
                    self.first_attempt, self.attempt_color = "other", aimed
                return R_WRONG, True, self._info("wrong_attempt")
            if self.first_attempt is None:
                if aimed is None:
                    self.first_attempt = "none"
                    return R_MISS, True, self._info("miss_attempt")
                self.first_attempt, self.attempt_color = "target", aimed
            if aimed == target and self.stage == 0:
                self.stage = 1
                reward += R_ATTEMPT

        # 다른 공이 들리거나 통에 들어가면 실패
        for c in self.candidates:
            if c != target and (env.ball_pos(c)[2] > LIFT_Z or env.ball_in_bin(c)):
                return reward + R_WRONG, True, self._info("wrong_ball")

        # 단계 보상
        if self.stage == 1 and env.ball_pos(target)[2] > LIFT_Z:
            self.stage = 2
            reward += R_LIFT
        if self.stage == 2 and env.ball_in_bin(target):
            self.stage = 3
            reward += R_BIN
        if self.stage == 3 and env.at_home():
            self.stage = 4
            return reward + R_HOME, True, self._info("success")

        if self.first_attempt is None and self.steps >= ATTEMPT_TIMEOUT_STEPS:
            return reward + R_MISS, True, self._info("attempt_timeout")
        return reward, False, self._info()
