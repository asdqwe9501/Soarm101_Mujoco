"""저차원 on-manifold 노이즈 조종 (torch 없음).

SAC 행동 z (STEER_DIM,)는 관절별 완만한 시간 성분 12개(B의 열)만 조종하고,
나머지 직교 여공간은 매번 새 N(0, I) 노이즈로 채운다:
    latent = B z + (I - B Bᵀ) ε,  ε ~ N(0, I_300)
z ~ N(0, I_12)이면 latent ~ N(0, I_300)으로 기본 분포와 정확히 같다.
"""

import numpy as np

from policies.dsrl_constants import ACT_DIM, CHUNK, LATENT_DIM, STEER_DIM


def _basis():
    t = np.arange(CHUNK)
    dct = [np.ones(CHUNK), np.cos(np.pi * (t + 0.5) / CHUNK)]  # DCT-II k=0, k=1
    B = np.zeros((LATENT_DIM, STEER_DIM))
    for j in range(ACT_DIM):
        for k, v in enumerate(dct):
            B[t * ACT_DIM + j, 2 * j + k] = v / np.linalg.norm(v)  # NoiseSteeredSmolVLA.chunk의 reshape(50, 6) 배치
    return B


BASIS = _basis()


class SteerLatent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def to_latent(self, z):
        eps = self.rng.standard_normal(LATENT_DIM)
        lat = BASIS @ np.asarray(z, np.float64) + eps - BASIS @ (BASIS.T @ eps)
        return lat.astype(np.float32)
