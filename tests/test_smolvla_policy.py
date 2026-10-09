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
