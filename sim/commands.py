"""명령문 다양화: 표현 유형(기본/유사/드문) × 색 표현 × 표기(대소문자, 마침표).

사용자 지정(2026-10-14):
- 하늘색 공은 "blue"로 부른다.
- 짧은 명령("Grab the blue ball.")도 전체 작업(잡아서 통에 넣고 제자리 복귀)을 뜻한다.
- 명령 색별 배분: 기본 표현 300 / 유사 표현 150 / 드문 표현 50 (TIER_SHARE).
- 평가 전용 표현(HELDOUT_TEMPLATES)은 학습에 쓰지 않는다.
"""

from sim.env import task_text

BASIC_TEMPLATE = "Grab the {color} ball."

SIMILAR_TEMPLATES = [
    "pick up the {color} ball",
    "take the {color} ball",
    "pick up the {color} one",
    "grab the {color} ball and put it in the box",
    "pick up the {color} ball and put it in the box",
    "take the {color} ball and place it in the box",
]

RARE_TEMPLATES = [
    "put the {color} ball in the box",
    "move the {color} ball into the box",
    "the {color} one, please",
]

HELDOUT_TEMPLATES = [
    "get the {color} ball",
    "lift the {color} ball and drop it in the box",
    "bring the {color} one to the box",
]

# IRL 전용 확장 표현(Plan 9): 학습 표현에 더해 IRL 시범·바꿔치기 음성 표본·정책 학습 명령에 쓴다.
# 평가 전용 표현과 겹치지 않고, 그 동사(get/lift/bring/drop)도 쓰지 않는다.
IRL_EXTRA_TEMPLATES = [
    "grasp the {color} ball",
    "pick the {color} ball",
    "please grab the {color} ball",
    "can you pick up the {color} ball",
    "grab that {color} ball",
    "take hold of the {color} ball",
    "collect the {color} ball",
    "go for the {color} ball",
    "choose the {color} one",
    "now the {color} ball",
    "place the {color} ball into the bin",
    "put away the {color} ball",
    "store the {color} ball in the container",
    "I want the {color} ball in the box",
    "the {color} ball goes in the box",
    "{color} ball into the box",
    "select the {color} ball and put it away",
    "pick out the {color} ball and place it in the box",
]

TRAIN_TEMPLATES = [BASIC_TEMPLATE.rstrip(".").lower()] + SIMILAR_TEMPLATES + RARE_TEMPLATES

BASE_WORD = {"red": "red", "black": "black", "sky blue": "blue"}
COLOR_WORDS = {
    "red": ["red", "red-colored"],
    "black": ["black", "black-colored"],
    "sky blue": ["blue", "blue-colored"],
}

TIERS = ("basic", "similar", "rare")
TIER_SHARE = {"basic": 300 / 500, "similar": 150 / 500, "rare": 50 / 500}
SPLITS = ("train", "heldout", "canonical", "irl")


def normalize(command):
    """표기 차이(대소문자, 마침표)를 없앤 형태. 테스트와 집계용."""
    return command.strip().rstrip(".").lower()


def _vary(text, rng):
    if rng.random() < 0.5:
        text = text[0].upper() + text[1:]
    if rng.random() < 0.5:
        text += "."
    return text


def command_for_tier(color, tier, rng):
    """표현 유형(tier)에 맞는 명령문 하나. basic은 항상 "Grab the {기본 색 이름} ball." 그대로다."""
    if tier == "basic":
        return BASIC_TEMPLATE.format(color=BASE_WORD[color])
    templates = {"similar": SIMILAR_TEMPLATES, "rare": RARE_TEMPLATES}[tier]
    words = COLOR_WORDS[color]
    text = templates[rng.integers(len(templates))].format(color=words[rng.integers(len(words))])
    return _vary(text, rng)


def pick_tier(remaining, rng):
    """남은 배정 수(remaining: tier -> 개수)에 비례해 tier를 고른다. 모두 0이면 비율(TIER_SHARE)대로."""
    counts = [max(remaining.get(t, 0), 0) for t in TIERS]
    weights = counts if sum(counts) > 0 else [TIER_SHARE[t] for t in TIERS]
    total = float(sum(weights))
    return TIERS[rng.choice(len(TIERS), p=[w / total for w in weights])]


def sample_command(color, rng, split="train"):
    """split: train(배분 비율로 tier를 뽑음), heldout(평가 전용 틀), canonical(기존 고정 문장),
    irl(학습 표현 + IRL 확장 표현에서 고르게 뽑음)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "canonical":
        return task_text(color)
    if split in ("heldout", "irl"):
        templates = HELDOUT_TEMPLATES if split == "heldout" else TRAIN_TEMPLATES + IRL_EXTRA_TEMPLATES
        words = COLOR_WORDS[color]
        text = templates[rng.integers(len(templates))].format(color=words[rng.integers(len(words))])
        return _vary(text, rng)
    return command_for_tier(color, pick_tier({}, rng), rng)
