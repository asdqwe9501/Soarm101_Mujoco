import numpy as np

from calib import SIM_HOME_NORM
from expert.scripted import ScriptedExpert
from rl.chunk_env import ChunkEnv
from rl.staged_reward import ATTEMPT_TIMEOUT_STEPS
from sim.env import BallSortEnv


class HomeSteer:
    feat_dim = 4

    def reset(self, env):
        self.env = env

    def features(self, obs):
        return np.ones(4, np.float32)

    def chunk(self, obs, latent):
        return np.tile(SIM_HOME_NORM.astype(np.float32), (50, 1))


class ExpertSteer(HomeSteer):
    """latent는 무시하고 시연 스크립트의 실행 행동을 묶음으로 내준다. aim 색을 노린다."""

    def __init__(self, aim=None):
        self.aim = aim

    def reset(self, env):
        self.env = env
        self.ex = ScriptedExpert(env, seed=0, dart_sigma=0.0)
        real = env.target
        env.target = self.aim or real
        self.ex.reset()
        env.target = real

    def chunk(self, obs, latent):
        # 앞 10개만 실행되므로 시연 스크립트는 10스텝만 앞으로 간다(나머지 40개는 미리보기)
        t0 = self.ex.t
        chunk = np.stack([self.ex.act()[0] for _ in range(50)]).astype(np.float32)
        self.ex.t = t0 + 10
        return chunk


def make(steer):
    return ChunkEnv(BallSortEnv(render=False), steer, exec_steps=10, steer_dim=12, latent_scale=2.0)


def rollout(ce, seed=0, target="red"):
    ce.reset(seed=seed, options={"target": target})
    total, n = 0.0, 0
    while True:
        _, r, term, trunc, info = ce.step(np.zeros(12, np.float32))
        total += r
        n += 1
        if term or trunc:
            return total, info, n


def test_spaces_and_reset():
    ce = make(HomeSteer())
    f, _ = ce.reset(seed=0, options={"target": "red"})
    assert f.shape == (4,) and ce.action_space.shape == (12,) and ce.action_space.high.max() == 2.0


def test_full_success_through_chunks():
    total, info, _ = rollout(make(ExpertSteer()))
    assert info["success"] and abs(total - 7.0) < 1e-9


def test_wrong_attempt_stops_at_that_env_step():
    ce = make(ExpertSteer(aim="black"))
    total, info, _ = rollout(ce)
    assert info["end_reason"] == "wrong_attempt" and abs(total + 1.0) < 1e-9
    # 판정 직후 멈췄는지: 같은 장면을 StagedReward만으로 돌려 얻은 스텝 수와 같아야 한다
    from rl.staged_reward import StagedReward

    env = BallSortEnv(render=False)
    env.reset(seed=0, options={"target": "red"})
    sr = StagedReward(env)
    sr.reset()
    steer = ExpertSteer(aim="black")
    steer.reset(env)
    n = 0
    while True:
        a = steer.ex.act()[0]
        env.step(a)
        n += 1
        if sr.step(a)[1]:
            break
    assert info["env_steps"] == n


def test_attempt_timeout_through_chunks():
    _, info, n = rollout(make(HomeSteer()))
    assert info["end_reason"] == "attempt_timeout" and info["env_steps"] == ATTEMPT_TIMEOUT_STEPS and n == ATTEMPT_TIMEOUT_STEPS // 10


def test_action_is_clipped_to_scale():
    ce = make(HomeSteer())
    ce.reset(seed=0)
    ce.step(np.full(12, 50.0, np.float32))
    assert ce.last_latent.shape == (12,) and np.all(ce.last_latent == 2.0)


def test_step_maps_z_to_300_dim_latent_on_basis():
    from policies.dsrl_latent import BASIS

    class Recorder(HomeSteer):
        def chunk(self, obs, latent):
            self.latent = latent
            return super().chunk(obs, latent)

    steer = Recorder()
    ce = make(steer)
    ce.reset(seed=0)
    z = np.linspace(-1.5, 1.5, 12).astype(np.float32)
    ce.step(z)
    assert steer.latent.shape == (300,) and np.allclose(BASIS.T @ steer.latent, z, atol=1e-5)


def test_irl_mode_records_staged_but_ends_only_on_env_success_or_time():
    ce = ChunkEnv(BallSortEnv(render=False), ExpertSteer(aim="black"), exec_steps=10, steer_dim=12,
                  latent_scale=2.0, staged_termination=False)
    total, info, _ = rollout(ce)
    assert info["end_reason"] == "wrong_attempt" and abs(total + 1.0) < 1e-9  # 기록은 그대로, 벌점은 한 번
    assert not info["env_success"] and info["env_steps"] == 900  # 잘못된 시도로 끝나지 않고 시간 제한까지

    ce = ChunkEnv(BallSortEnv(render=False), ExpertSteer(), exec_steps=10, steer_dim=12, latent_scale=2.0,
                  staged_termination=False)
    total, info, _ = rollout(ce)
    assert info["env_success"] and info["success"] and abs(total - 7.0) < 1e-9
