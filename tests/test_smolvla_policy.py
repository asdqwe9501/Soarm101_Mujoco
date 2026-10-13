import numpy as np
import pytest
import torch

from sim.env import BallSortEnv

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_adapter_returns_finite_joint_targets():
    from policies.smolvla_policy import SmolVLAPolicyAdapter

    env = BallSortEnv(render=True)
    obs, _ = env.reset(seed=0)
    policy = SmolVLAPolicyAdapter("lerobot/smolvla_base")
    policy.reset(env)
    a1 = policy.act(obs)
    obs, *_ = env.step(a1)
    a2 = policy.act(obs)
    for a in (a1, a2):
        assert a.shape == (6,) and a.dtype == np.float32 and np.all(np.isfinite(a))
    env.close()


SMOKE_PEFT = "outputs/train/lora_smoke16/checkpoints/last/pretrained_model"


@pytest.mark.skipif(
    not __import__("os").path.exists(f"{SMOKE_PEFT}/adapter_config.json"),
    reason="needs the LoRA smoke checkpoint (bash train/sft_lora.sh lora_smoke16 50 16)",
)
def test_adapter_loads_peft_checkpoint_merged():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    from policies.smolvla_policy import SmolVLAPolicyAdapter

    env = BallSortEnv(render=True)
    obs, _ = env.reset(seed=0)
    policy = SmolVLAPolicyAdapter(SMOKE_PEFT)
    assert type(policy.policy) is SmolVLAPolicy  # LoRA merged, no PEFT wrapper left
    assert not any("lora_" in n for n, _ in policy.policy.named_parameters())
    policy.reset(env)
    a = policy.act(obs)
    assert a.shape == (6,) and a.dtype == np.float32 and np.all(np.isfinite(a))
    env.close()
