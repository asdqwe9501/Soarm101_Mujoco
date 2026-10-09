"""SmolVLA 체크포인트를 평가기 정책 프로토콜(reset(env), act(obs))로 감싼다."""

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.front": "observation.images.camera2",
}


class SmolVLAPolicyAdapter:
    def __init__(self, checkpoint, device="cuda", n_action_steps=None):
        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": RENAME_MAP},
            },
        )
        self.task = None

    def reset(self, env):
        self.policy.reset()
        self.task = env.task

    def _batch(self, obs):
        batch = {"observation.state": torch.from_numpy(obs["observation.state"]).unsqueeze(0), "task": [self.task]}
        for key in ("observation.images.top", "observation.images.front"):
            img = torch.from_numpy(obs[key]).permute(2, 0, 1).float().div(255.0)
            batch[key] = img.unsqueeze(0)
        return batch

    @torch.inference_mode()
    def act(self, obs):
        action = self.policy.select_action(self.pre(self._batch(obs)))
        action = self.post(action)
        return action.squeeze(0).float().cpu().numpy().astype(np.float32)
