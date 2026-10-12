"""손목 카메라 색 확인 장치(color guard).

감싼 정책이 그리퍼를 닫기 시작하면(명령이 40을 넘었다가 40 아래로 내려가면) 그 행동을 실행하기 전에
현재 손목 영상(observation.images.top)의 집게 사이 영역 색을 판별한다. 판별이 불확실하면 다음 스텝에 다시 보고,
명령이 20 아래로 내려가는 순간(StagedReward의 잡기 시도 판정 시점)까지 확실한 판별이 없으면 그대로 닫는다.
(20 미만 시점에서야 거부하면 집게가 이미 공에 닿아 있어 다시 열 때 공을 튕긴다. 40 시점은 아직 닿기 전이다.)
- 명령한 색이거나 판별 불가(None) → 그대로 실행
- 다른 색으로 판별 → 닫지 않는다. 그리퍼를 연 채 (1) OPEN_STEPS 동안 제자리에서 열고 (2) 최근 REWIND_STEPS개의
  관절 상태를 거꾸로 되짚어 들어 올린 뒤(공을 쳐서 굴리지 않도록) (3) RETREAT_STEPS 동안 제자리 자세로 돌아가며 그리퍼를 GRIP_PARK까지 닫는다.
  그 다음 감싼 정책을 리셋(새 노이즈)해 다시 시도하게 한다. 최대 max_retries회. 그 뒤에는 그대로 통과시킨다.
- 명령한 색을 확인하고 닫은 뒤에는 더 보지 않는다(통 위에서 놓은 뒤 제자리로 가며 닫는 것은 잡기가 아니다).
쓰는 입력은 손목 영상, 관절 상태, 정책의 행동 명령, 명령문(env.task)뿐이다(시뮬레이터 정답 없음).

관심 영역과 임계값은 시뮬레이션 손목 영상(시연 데이터의 잡기 직전 프레임)으로 정했다.
실제 로봇에서는 카메라 위치·조명·공 색이 달라지므로 ROI와 HSV 임계값을 다시 보정해야 한다.
"""

import re
from collections import deque

import numpy as np
from scipy import ndimage

from calib import SIM_HOME_NORM
from expert.scripted import min_jerk
from rl.staged_reward import GRIP_CLOSE_CMD, GRIP_OPEN_CMD

MAX_RETRIES = 3
OPEN_STEPS = 5
REWIND_STEPS = 60  # 최근 2초의 하강 경로를 되짚는다
REWIND_STRIDE = 2  # 되짚을 때 2배 속도
RETREAT_STEPS = 45  # 1.5초
GRIP_OPEN = 60.0
# 후퇴 끝 그리퍼 명령. 시연의 시작 자세(제자리, 그리퍼 닫힘 ≈1.5)에 가깝게 하되 20 미만으로 내리면
# 잡기 시도로 판정되므로 바로 위에 둔다. 제자리에서 열린 채 시작하면 정책이 '놓은 뒤 복귀'로 보고 곧바로 닫았다.
GRIP_PARK = 21.0

# 집게 사이 공이 보이는 영역 (480×640 손목 영상, 잡기 직전 공 중심 ≈ (x375, y370), 반지름 ≈ 100px)
ROI = (slice(250, 480), slice(255, 495))  # (행 y, 열 x)
MIN_FRAC = 0.15  # 가장 많은 색의 ROI 점유율 하한
DOMINANCE = 2.0  # 두 번째 색보다 이만큼 배 이상 많아야 한다


def _color_masks(img):
    """HSV 규칙으로 색별 픽셀 마스크. 색상 각도는 0~360, 채도·명도는 0~1."""
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    v = rgb.max(-1)
    c = v - rgb.min(-1)
    s = np.where(v > 0, c / np.maximum(v, 1e-6), 0.0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cc = np.maximum(c, 1e-6)
    h = np.where(v == r, ((g - b) / cc) % 6, np.where(v == g, (b - r) / cc + 2, (r - g) / cc + 4)) * 60.0
    return {
        "red": ((h < 15) | (h > 345)) & (s > 0.6) & (v > 0.3),
        "black": v < 0.2,
        "sky blue": (h > 185) & (h < 225) & (s > 0.25) & (v > 0.45),
    }


def _drop_border_blobs(mask):
    """위·왼쪽·오른쪽 가장자리에 닿는 덩어리(테이블 밖 배경, 멀리 있는 물체)를 지운다. 공은 아래쪽 가장자리에만 닿는다."""
    lab, _ = ndimage.label(mask)
    bad = np.union1d(np.union1d(lab[0], lab[:, 0]), lab[:, -1])
    bad = bad[bad > 0]
    return mask & ~np.isin(lab, bad) if len(bad) else mask


def wrist_color_fractions(img):
    """ROI 안에서 색별 픽셀 비율."""
    return {k: float(_drop_border_blobs(m)[ROI].mean()) for k, m in _color_masks(img).items()}


def classify_wrist(img):
    """손목 영상(H×W×3 uint8 RGB) → 'red' | 'black' | 'sky blue' | None(공 없음/불확실)."""
    f = wrist_color_fractions(img)
    (c1, f1), (_, f2) = sorted(f.items(), key=lambda kv: -kv[1])[:2]
    return c1 if f1 >= MIN_FRAC and f1 >= DOMINANCE * f2 else None


def command_color(task):
    m = re.search(r"grab the (.+?) ball", task)
    return m.group(1) if m else None


class ColorGuard:
    """평가기 프로토콜 reset(env), act(obs)를 따르는 정책 감싸개."""

    def __init__(self, policy, max_retries=MAX_RETRIES, retreat_steps=RETREAT_STEPS, home=SIM_HOME_NORM):
        self.policy, self.max_retries, self.retreat_steps = policy, max_retries, retreat_steps
        self.home = np.asarray(home, dtype=np.float32)

    def reset(self, env):
        self.env = env
        self.policy.reset(env)
        self.target = command_color(env.task)
        self.decided = False  # 이번 닫기에 대해 판별을 마쳤는가
        self.confirmed = False  # 명령한 색을 확인하고 닫았다 → 이후(통에 놓기, 제자리 복귀)의 닫기는 보지 않는다
        self.armed = False  # 실행된 행동 기준: 그리퍼를 연 뒤 아직 닫지 않음 (StagedReward와 같음)
        self.park_grip = False  # 후퇴 뒤 정책이 스스로 다시 열 때까지 그리퍼 명령을 GRIP_PARK 이상으로 묶는다
        self.retreat = []
        self.history = deque(maxlen=REWIND_STEPS)  # 최근 관절 상태
        self.t = 0
        self.stats = {"rejections": 0, "checks": []}

    def _retreat_plan(self, state):
        rewind = [np.asarray(state, dtype=np.float32)] * OPEN_STEPS + list(reversed(self.history))[::REWIND_STRIDE]
        plan = [a.copy() for a in rewind]
        for a in plan:
            a[5] = GRIP_OPEN
        start, end = plan[-1], self.home.copy()
        end[5] = GRIP_PARK
        plan += [min_jerk(start, end, (i + 1) / self.retreat_steps) for i in range(self.retreat_steps)]
        return [a.astype(np.float32) for a in plan]

    def _next(self, obs):
        if self.retreat:
            a = self.retreat.pop(0)
            if not self.retreat:
                self.policy.reset(self.env)  # 큐 비우기 + 새 노이즈
                self.park_grip = True
            return a
        a = np.array(self.policy.act(obs), dtype=np.float32)
        if self.park_grip:
            if a[5] > GRIP_OPEN_CMD:
                self.park_grip = False
            else:  # 정책이 다시 열기 전의 닫기는 잡기 시도가 아니다: 20 미만으로 내려가지 않게 한다
                a[5] = max(a[5], GRIP_PARK)
                return a
        if self.armed and not self.decided and not self.confirmed and a[5] < GRIP_OPEN_CMD:
            seen = classify_wrist(obs["observation.images.top"])
            if seen is None and a[5] >= GRIP_CLOSE_CMD:
                return a  # 아직 잡기 시도 전: 다음 스텝에 다시 본다
            self.decided = True
            reject = seen is not None and seen != self.target and self.stats["rejections"] < self.max_retries
            self.stats["checks"].append(
                {"step": self.t, "commanded": self.target, "seen": seen, "rejected": reject}
            )
            self.confirmed = seen == self.target
            if reject:
                self.stats["rejections"] += 1
                self.retreat = self._retreat_plan(obs["observation.state"])
                return self.retreat.pop(0)
        return a

    def act(self, obs):
        self.history.append(np.asarray(obs["observation.state"], dtype=np.float32))
        a = self._next(obs)
        if a[5] > GRIP_OPEN_CMD:
            self.armed, self.decided = True, False
        elif a[5] < GRIP_CLOSE_CMD:
            self.armed = False
        self.t += 1
        return a
