import numpy as np
import pytest

from sim.commands import COLOR_WORDS, HELDOUT_TEMPLATES, TRAIN_TEMPLATES, normalize, sample_command
from sim.env import COLORS, BallSortEnv, task_text


def test_templates_disjoint_and_have_color_slot():
    assert len(TRAIN_TEMPLATES) >= 8 and len(HELDOUT_TEMPLATES) >= 3
    assert not set(TRAIN_TEMPLATES) & set(HELDOUT_TEMPLATES)
    for t in TRAIN_TEMPLATES + HELDOUT_TEMPLATES:
        assert t.count("{color}") == 1


def test_sky_blue_ball_is_called_blue():
    assert COLOR_WORDS["sky blue"] == ["blue", "blue-colored"]


def test_sample_command_mentions_only_the_commanded_color():
    rng = np.random.default_rng(0)
    for color in COLORS:
        for _ in range(50):
            words = normalize(sample_command(color, rng)).replace("-colored", "").split()
            assert any(w in words for w in [w.replace("-colored", "") for w in COLOR_WORDS[color]])
            others = {w.replace("-colored", "") for c in COLORS if c != color for w in COLOR_WORDS[c]}
            assert not others & set(words)


def test_sample_command_is_diverse_and_respects_split():
    rng = np.random.default_rng(1)
    train = {normalize(sample_command("red", rng, split="train")) for _ in range(300)}
    held = {normalize(sample_command("red", rng, split="heldout")) for _ in range(100)}
    assert len(train) >= 10 and not train & held
    raw = {sample_command("red", rng) for _ in range(300)}
    assert any(c[0].isupper() for c in raw) and any(c.endswith(".") for c in raw)
    assert sample_command("red", rng, split="canonical") == task_text("red")
    with pytest.raises(ValueError):
        sample_command("red", rng, split="bogus")


def test_env_reset_accepts_command_override():
    env = BallSortEnv(render=False)
    _, info = env.reset(seed=0, options={"target": "black", "command": "Take the black-colored ball."})
    assert env.task == info["task"] == "Take the black-colored ball."
    _, info = env.reset(seed=0, options={"target": "black"})
    assert env.task == task_text("black")


def test_basic_tier_is_fixed_grab_sentence():
    from sim.commands import command_for_tier

    rng = np.random.default_rng(2)
    assert command_for_tier("sky blue", "basic", rng) == "Grab the blue ball."
    assert command_for_tier("red", "basic", rng) == "Grab the red ball."


def test_pick_tier_follows_remaining_quota_exactly():
    from sim.commands import TIERS, pick_tier

    rng = np.random.default_rng(3)
    remaining = {"basic": 300, "similar": 150, "rare": 50}
    got = {t: 0 for t in TIERS}
    for _ in range(500):
        t = pick_tier(remaining, rng)
        got[t] += 1
        remaining[t] -= 1
    assert got == {"basic": 300, "similar": 150, "rare": 50}


def test_rare_and_similar_and_heldout_are_separate():
    from sim.commands import RARE_TEMPLATES, SIMILAR_TEMPLATES

    assert not set(RARE_TEMPLATES) & set(SIMILAR_TEMPLATES)
    assert not set(HELDOUT_TEMPLATES) & (set(RARE_TEMPLATES) | set(SIMILAR_TEMPLATES))
