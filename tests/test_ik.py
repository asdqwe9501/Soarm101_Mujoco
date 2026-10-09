import numpy as np
import pytest

from calib import SIM_HOME_NORM, Calib
from expert.ik import IK
from sim.env import BALL_R_MAX, BALL_X, BALL_Y, BIN_CENTER, BIN_JITTER, BallSortEnv


@pytest.fixture(scope="module")
def ik_and_home():
    env = BallSortEnv(render=False)
    return IK(env.model), Calib().norm_to_rad(SIM_HOME_NORM)


def _grid():
    for x in np.linspace(*BALL_X, 5):
        for y in np.linspace(*BALL_Y, 6):
            if np.hypot(x, y) <= BALL_R_MAX:
                yield x, y


@pytest.mark.parametrize("z", [0.012, 0.07])
def test_reach_ball_region(ik_and_home, z):
    ik, home = ik_and_home
    for x, y in _grid():
        q, err, tilt = ik.solve(np.array([x, y, z]), home)
        assert err < 0.003, (x, y, z, err)
        assert tilt < 35, (x, y, z, tilt)


def test_reach_bin_release(ik_and_home):
    ik, home = ik_and_home
    for dx in (-BIN_JITTER, 0, BIN_JITTER):
        for dy in (-BIN_JITTER, 0, BIN_JITTER):
            p = np.array([BIN_CENTER[0] + dx, BIN_CENTER[1] + dy, 0.09])
            q, err, tilt = ik.solve(p, home)
            assert err < 0.005 and tilt < 45, (p, err, tilt)


def test_keeps_roll_and_gripper(ik_and_home):
    ik, home = ik_and_home
    q, _, _ = ik.solve(np.array([0.2, 0.0, 0.05]), home)
    assert q[4] == home[4] and q[5] == home[5]
