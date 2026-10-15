import os

import pytest
import torch

from train import sft_aux

CKPT = "outputs/train/sft_v3/checkpoints/040000/pretrained_model"


def test_xy_normalization_round_trip():
    xy = torch.tensor([[0.10, -0.12], [0.30, 0.22], [0.2, 0.05]])
    z = sft_aux.normalize_xy(xy)
    assert torch.allclose(z[0], torch.tensor([-1.0, -1.0])) and torch.allclose(z[1], torch.tensor([1.0, 1.0]))
    assert torch.allclose(sft_aux.unnormalize_xy(z), xy, atol=1e-6)


def test_preprocessor_patch_keeps_target_xy(monkeypatch):
    from lerobot.processor import converters

    monkeypatch.setattr(converters, "_extract_complementary_data", sft_aux._extract_with_aux)
    batch = {"observation.state": torch.zeros(2, 6), "task": ["a", "b"], "target_xy": torch.ones(2, 2)}
    out = converters.transition_to_batch(converters.batch_to_transition(batch))
    assert torch.equal(out["target_xy"], torch.ones(2, 2))


def test_early_indices_only_first_frames_of_each_episode():
    from types import SimpleNamespace

    ds = SimpleNamespace(
        episodes=None,
        meta=SimpleNamespace(episodes={"dataset_from_index": [0, 5, 30], "dataset_to_index": [5, 30, 42]}),
    )
    idx = sft_aux.early_indices(ds, max_frame=10)
    assert idx == list(range(0, 5)) + list(range(5, 15)) + list(range(30, 40))


V8 = "data/so101_ballsort_sim_v8"


def test_early_loader_yields_only_early_frames():
    if not os.path.exists(V8):
        pytest.skip("needs the v8 dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/so101_ballsort_sim_v8", root=V8)
    loader = sft_aux.early_loader(ds, batch_size=4, num_workers=0)
    assert len(loader.dataset) == sum(
        min(10, t - f) for f, t in zip(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"])
    )
    b = next(iter(loader))
    assert b["frame_index"].shape == (4,) and torch.all(b["frame_index"] < sft_aux.AUX_MAX_FRAME)
    assert "target_xy" in b


@pytest.fixture(scope="module")
def policy():
    if not torch.cuda.is_available() or not os.path.exists(CKPT):
        pytest.skip("needs CUDA and the sft_v3 checkpoint")
    p = sft_aux.SmolVLAAuxPolicy.from_pretrained(CKPT)
    p.train()
    return p


def _batch(policy, b=2):
    dev = "cuda"
    batch = {
        "observation.images.camera1": torch.rand(b, 3, 256, 256, device=dev),
        "observation.images.camera2": torch.rand(b, 3, 256, 256, device=dev),
        "observation.state": torch.randn(b, 6, device=dev),
        "action": torch.randn(b, policy.config.chunk_size, 6, device=dev),
        "observation.language.tokens": torch.randint(0, 1000, (b, 48), device=dev),
        "observation.language.attention_mask": torch.ones(b, 48, dtype=torch.bool, device=dev),
        "target_xy": torch.tensor([[0.15, 0.0], [0.25, 0.1]], device=dev)[:b],
    }
    return batch


def test_aux_head_shape_and_time_is_one(policy, monkeypatch):
    times = []
    orig = policy.model.embed_suffix

    def record(x_t, timestep):
        times.append(timestep.detach().clone())
        return orig(x_t, timestep)

    monkeypatch.setattr(policy.model, "embed_suffix", record)
    with torch.no_grad():
        pred = policy.aux_predict(_batch(policy))
    assert pred.shape == (2, 2)
    assert len(times) == 1 and torch.all(times[0] == 1.0)


def test_aux_forward_zeroes_state(policy, monkeypatch):
    states = []
    orig = policy.model.embed_prefix

    def record(*a, state=None, **k):
        states.append(state.detach().clone())
        return orig(*a, state=state, **k)

    monkeypatch.setattr(policy.model, "embed_prefix", record)
    with torch.no_grad():
        policy.aux_predict(_batch(policy))
    assert len(states) == 1 and torch.all(states[0] == 0)


def test_forward_aux_uses_early_batch_not_training_batch(policy, monkeypatch):
    early = _batch(policy)
    early["target_xy"] = torch.tensor([[0.12, -0.1], [0.28, 0.2]], device="cuda")
    seen = []
    orig = policy.aux_predict
    monkeypatch.setattr(policy, "aux_predict", lambda b: seen.append(b) or orig(b))
    monkeypatch.setattr(policy, "aux_batches", iter([early]))
    with torch.no_grad():
        policy.forward(_batch(policy))
    assert len(seen) == 1 and seen[0] is early


def test_forward_without_early_sampler_is_plain_flow_loss(policy):
    with torch.no_grad():
        _, d = policy.forward(_batch(policy))
    assert "aux_loss" not in d


def test_forward_adds_aux_loss_and_grads_reach_head_and_expert(policy, monkeypatch):
    monkeypatch.setattr(policy, "aux_batches", iter([_batch(policy)]))
    policy.zero_grad()
    loss, d = policy.forward(_batch(policy))
    assert {"flow_loss", "aux_loss", "aux_err_cm"} <= d.keys()
    assert abs(d["loss"] - (d["flow_loss"] + policy.aux_lambda * d["aux_loss"])) < 1e-4
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in policy.aux_head.parameters())
    expert = policy.model.vlm_with_expert.lm_expert
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in expert.parameters())
    policy.zero_grad()

