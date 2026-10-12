"""노이즈(잠재 행동)를 행동으로 받는 RL 환경. 한 스텝 = SmolVLA 행동 묶음 앞부분 실행, 보상 = StagedReward."""

import gymnasium as gym
import numpy as np

from policies.dsrl_constants import EXEC_STEPS, LATENT_DIM, LATENT_SCALE
from rl.staged_reward import StagedReward


class ChunkEnv(gym.Env):
    def __init__(self, env, steer, exec_steps=EXEC_STEPS, latent_dim=LATENT_DIM, latent_scale=LATENT_SCALE):
        self.env, self.steer, self.exec_steps = env, steer, exec_steps
        self.reward_fn = StagedReward(env)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (steer.feat_dim,), np.float32)
        self.action_space = gym.spaces.Box(-latent_scale, latent_scale, (latent_dim,), np.float32)
        self._obs = None
        self.last_latent = None

    def reset(self, seed=None, options=None):
        self._obs, info = self.env.reset(seed=seed, options=options)
        self.reward_fn.reset()
        self.steer.reset(self.env)
        self._env_steps = 0
        return self.steer.features(self._obs), info

    def step(self, latent):
        latent = np.clip(np.asarray(latent, np.float32), self.action_space.low, self.action_space.high)
        self.last_latent = latent
        actions = self.steer.chunk(self._obs, latent)
        reward, terminated, truncated, info = 0.0, False, False, {}
        for a in actions[: self.exec_steps]:
            self._obs, _, _, env_trunc, _ = self.env.step(a)
            self._env_steps += 1
            r, done, info = self.reward_fn.step(a)
            reward += r
            terminated = done
            truncated = env_trunc and not done
            if terminated or truncated:
                break
        info["env_steps"] = self._env_steps
        return self.steer.features(self._obs), float(reward), terminated, truncated, info
