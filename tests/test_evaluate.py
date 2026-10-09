from calib import SIM_HOME_NORM
from eval.evaluate import ExpertPolicy, classify_failure, evaluate, evaluate_sequential


class StayHome:
    def reset(self, env):
        pass

    def act(self, obs):
        return SIM_HOME_NORM


def test_classify_failure():
    assert classify_failure({"wrong_in_bin": True, "target_in_bin": False}, max_target_z=0.1) == "wrong_ball"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": False}, max_target_z=0.02) == "grasp_fail"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": False}, max_target_z=0.08) == "release_fail"
    assert classify_failure({"wrong_in_bin": False, "target_in_bin": True}, max_target_z=0.08) == "return_fail"


def test_evaluate_stay_home_fails_everything():
    r = evaluate(StayHome(), n_per_color=1, seed=0)
    assert r["n"] == 3 and r["success_rate"] == 0.0 and r["failures"]["grasp_fail"] == 3


def test_evaluate_expert():
    r = evaluate(ExpertPolicy(), n_per_color=2, seed=100)
    assert r["success_rate"] >= 5 / 6


def test_sequential_expert():
    r = evaluate_sequential(ExpertPolicy(), n_trials=2, seed=200)
    assert r["n"] == 2 and 0.0 <= r["all_success_rate"] <= 1.0
