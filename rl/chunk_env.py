"""저차원 조종 벡터 z(→ 300차원 노이즈 잠재값)를 행동으로 받는 RL 환경. 한 스텝 = SmolVLA 행동 묶음 앞부분 실행, 보상 = StagedReward."""

import gymnasium as gym
import numpy as np

from policies.dsrl_constants import EXEC_STEPS, LATENT_SCALE, STEER_DIM
from policies.dsrl_latent import SteerLatent
from rl.staged_reward import StagedReward


class ChunkEnv(gym.Env):
    def __init__(self, env, steer, exec_steps=EXEC_STEPS, steer_dim=STEER_DIM, latent_scale=LATENT_SCALE, seed=0,
                 staged_termination=True):
        """staged_termination=False(IRL): 단계별 보상은 기록만 하고, 종료는 env 성공 또는 시간 제한으로만 한다."""
        self.env, self.steer, self.exec_steps = env, steer, exec_steps
        self.staged_termination = staged_termination
        self.reward_fn = StagedReward(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (steer.feat_dim,), np.float32)
        self.action_space = gym.spaces.Box(-latent_scale, latent_scale, (steer_dim,), np.float32)
        self.steer_latent = SteerLatent(seed)
        self._obs = None
        self.last_latent = None

    def reset(self, seed=None, options=None):
        self._obs, info = self.env.reset(seed=seed, options=options)
        self.reward_fn.reset()
        self.steer.reset(self.env)
        self._env_steps = 0
        self._staged_done, self._staged_info = False, {}
        return self.steer.features(self._obs), info

    def step(self, z):
        z = np.clip(np.asarray(z, np.float32), self.action_space.low, self.action_space.high)
        self.last_latent = z  # SAC 행동(z). 리플레이 버퍼에 저장된다.
        actions = self.steer.chunk(self._obs, self.steer_latent.to_latent(z))
        reward, terminated, truncated, info = 0.0, False, False, {}
        for a in actions[: self.exec_steps]:
            self._obs, _, env_success, env_trunc, _ = self.env.step(a)
            self._env_steps += 1
            if not self._staged_done:
                r, self._staged_done, self._staged_info = self.reward_fn.step(a)
                reward += r
            terminated = self._staged_done if self.staged_termination else env_success
            truncated = env_trunc and not terminated
            if terminated or truncated:
                break
        info = dict(self._staged_info)
        info["env_steps"] = self._env_steps
        info["env_success"] = bool(env_success)
        return self.steer.features(self._obs), float(reward), terminated, truncated, info
