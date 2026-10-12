"""DSRL용 노이즈 조종 SmolVLA 래퍼.

작은 RL 정책이 flow-matching 초기 노이즈를 고른다. 이 래퍼는
(a) 그 정책의 입력 특징(features)과 (b) 주어진 노이즈 잠재값에 대한 행동 묶음(chunk)을 제공한다.
SmolVLA 내부 행동 큐(select_action)는 쓰지 않는다.
"""

import numpy as np
import torch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from policies.dsrl_constants import ACT_DIM, CHUNK, EXEC_STEPS, LATENT_DIM, LATENT_SCALE, MAX_DIM, STEER_DIM  # noqa: F401
from policies.dsrl_latent import SteerLatent
from policies.smolvla_policy import SmolVLAPolicyAdapter

N_REAL_CAMERAS = 2  # camera1(손목), camera2(위쪽). 그 뒤에 empty_cameras=1 빈 카메라가 붙는다.
TOKEN_GRID = 8  # 이미지당 64 토큰 = 8x8. SmolVLM connector pixel_shuffle 출력은 행 우선(row*8 + col).
POOL_GRID = 4  # 위쪽 카메라를 4x4 칸으로 평균 풀링
PROJ_DIM = 32  # 칸마다 고정 무작위 투영 960 -> 32
PROJ_SEED = 1234  # 투영 행렬 생성용 고정 시드


def _standardize(x):
    """파라미터 없는 블록 단위 layer norm."""
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-6)


class NoiseSteeredSmolVLA:
    def __init__(self, checkpoint, device="cuda", seed=0):
        self.device = device
        self.adapter = SmolVLAPolicyAdapter(checkpoint, device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        hidden = self.adapter.policy.model.vlm_with_expert.config.text_config.hidden_size
        proj = np.random.default_rng(PROJ_SEED).standard_normal((hidden, PROJ_DIM)) / np.sqrt(hidden)
        self.proj = torch.as_tensor(proj, dtype=torch.float32, device=device)
        n_top = POOL_GRID * POOL_GRID * PROJ_DIM
        sizes = {"wrist": hidden, "top": n_top, "lang": hidden, "state": ACT_DIM}
        self.blocks, start = {}, 0
        for name, n in sizes.items():
            self.blocks[name] = slice(start, start + n)
            start += n
        self.feat_dim = start  # 960 + 512 + 960 + 6 = 2438

    @property
    def task(self):
        return self.adapter.task

    @task.setter
    def task(self, value):
        self.adapter.task = value

    def reset(self, env):
        self.adapter.reset(env)  # policy.reset() + 명령문 기억

    @torch.inference_mode()
    def features(self, obs):
        """블록별로 표준화한 특징을 이어 붙인다 (위치는 self.blocks).

        wrist(960): 손목 이미지 토큰 평균. top(512): 위쪽 이미지 8x8 토큰을 4x4 칸으로 평균 풀링하고
        칸마다 고정 무작위 행렬로 960->32 투영. lang(960): 유효 언어 토큰 평균.
        wrist/top/lang은 각 블록 안에서 평균 0, 표준편차 1로 표준화한다.
        state(6): 전처리된 observation.state 앞 6개(정규화 단위, 표준화하지 않음).
        """
        policy = self.adapter.policy
        batch = self.adapter.pre(self.adapter._batch(obs))
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        embs, _, _ = policy.model.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        # 토큰 순서: [이미지 블록 × len(images)] → 언어(lang_len) → 상태(1). 특수 토큰 없음(add_image_special_tokens=False).
        n_lang = lang_tokens.shape[1]
        n_img_total = embs.shape[1] - n_lang - 1
        assert not policy.config.add_image_special_tokens and n_img_total % len(images) == 0
        n_img = n_img_total // len(images)
        embs = embs[0].float()
        assert n_img == TOKEN_GRID * TOKEN_GRID
        wrist = embs[:n_img].mean(0)
        k = TOKEN_GRID // POOL_GRID
        top = embs[n_img : 2 * n_img].reshape(POOL_GRID, k, POOL_GRID, k, -1).mean((1, 3))  # (4, 4, 960)
        top = (top.reshape(POOL_GRID * POOL_GRID, -1) @ self.proj).flatten()
        lang = embs[n_img_total : n_img_total + n_lang][lang_masks[0].bool()].mean(0)
        parts = [_standardize(wrist), _standardize(top), _standardize(lang), batch[OBS_STATE][0, :ACT_DIM].float()]
        return torch.cat(parts).cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def chunk(self, obs, latent):
        """latent (300,) → 정규화 단위 관절 목표 묶음 (50, 6)."""
        noise = torch.randn((1, CHUNK, MAX_DIM), generator=self.gen, device=self.device)
        noise[0, :, :ACT_DIM] = torch.as_tensor(
            np.asarray(latent, np.float32).reshape(CHUNK, ACT_DIM), device=self.device
        )
        batch = self.adapter.pre(self.adapter._batch(obs))
        actions = self.adapter.policy.predict_action_chunk(batch, noise=noise)
        actions = self.adapter.post(actions)
        return actions[0].float().cpu().numpy().astype(np.float32)


class DSRLPolicyAdapter:
    """평가기 프로토콜(reset(env), act(obs)). 큐가 비면 features → actor(결정적) z → SteerLatent → chunk 앞 EXEC_STEPS개."""

    def __init__(self, checkpoint, actor_path, device="cuda"):
        from rl.sac import SAC

        self.steer = NoiseSteeredSmolVLA(checkpoint, device=device)
        self.sac = SAC(self.steer.feat_dim, STEER_DIM, LATENT_SCALE, device=device)
        self.steer_latent = SteerLatent(seed=0)
        self.sac.load(actor_path)
        self.queue = []

    def reset(self, env):
        self.steer.reset(env)
        self.queue = []

    def act(self, obs):
        if not self.queue:
            z = self.sac.act(self.steer.features(obs), deterministic=True)
            self.queue = list(self.steer.chunk(obs, self.steer_latent.to_latent(z))[:EXEC_STEPS])
        return self.queue.pop(0)
