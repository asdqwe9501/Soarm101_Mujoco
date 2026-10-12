"""DSRL용 노이즈 조종 SmolVLA 래퍼.

작은 RL 정책이 flow-matching 초기 노이즈를 고른다. 이 래퍼는
(a) 그 정책의 입력 특징(features)과 (b) 주어진 노이즈 잠재값에 대한 행동 묶음(chunk)을 제공한다.
SmolVLA 내부 행동 큐(select_action)는 쓰지 않는다.
"""

import numpy as np
import torch
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from policies.dsrl_constants import ACT_DIM, CHUNK, EXEC_STEPS, LATENT_DIM, LATENT_SCALE, MAX_DIM  # noqa: F401
from policies.smolvla_policy import SmolVLAPolicyAdapter

N_REAL_CAMERAS = 2  # camera1(손목), camera2(위쪽). 그 뒤에 empty_cameras=1 빈 카메라가 붙는다.


class NoiseSteeredSmolVLA:
    def __init__(self, checkpoint, device="cuda", seed=0):
        self.device = device
        self.adapter = SmolVLAPolicyAdapter(checkpoint, device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        hidden = self.adapter.policy.model.vlm_with_expert.config.text_config.hidden_size
        self.feat_dim = 3 * hidden + ACT_DIM

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
        """[손목 이미지 토큰 평균, 위쪽 이미지 토큰 평균, 언어 토큰 평균, 정규화 관절 6개]."""
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
        parts = [embs[i * n_img : (i + 1) * n_img].mean(0) for i in range(N_REAL_CAMERAS)]
        lang = embs[n_img_total : n_img_total + n_lang]
        m = lang_masks[0].bool()
        parts.append(lang[m].mean(0))
        parts.append(batch[OBS_STATE][0, :ACT_DIM].float())
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
