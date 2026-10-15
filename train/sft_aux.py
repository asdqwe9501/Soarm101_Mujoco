"""lerobot-train을 감싸 SmolVLA에 보조 과제(명령한 색 공의 위치 예측)를 더한다. Plan 6 Task 2.

- 보조 출력 층(aux_head): expert가 행동 토큰에 대해 낸 은닉 상태(action_out_proj의 입력)의 평균 → target_xy(정규화).
- 누출 방지: 보조 예측은 t = 1(순수 노이즈 행동)로 따로 forward 해서 구한다. 흐름 손실은 원래 forward 그대로다.
- 지름길 방지(sft_aux1의 교훈): sft_aux1은 학습 배치에서 오차 0.9 cm였지만 첫 프레임에서는 8 cm였고 명령을 바꿔도 예측이
  0.7 cm밖에 안 바뀌었다. 팔이 이미 목표로 움직인 뒤의 관절 상태/영상에서 목표를 읽었기 때문이다. 그래서
  (1) 보조 forward에는 고유수용 상태 대신 0을 넣고, (2) 보조 손실은 팔이 아직 움직이지 않은 초기 프레임
  (frame_index < AUX_MAX_FRAME, 기본 10)에서만 계산한다. 무작위 학습 배치에는 그런 프레임이 ~2%뿐이므로 초기 프레임만
  뽑는 별도 DataLoader(AUX_BS, 기본 16)를 학습 전에 만들어 학습과 같은 전처리기를 거친 배치를 매 스텝 하나씩 쓴다.
- loss = flow_loss + λ·MSE(정규화 좌표). λ는 환경변수 AUX_LAMBDA(기본 0.1: smolvla_base에서 시작할 때 스모크로 정함. sft_v3처럼 흐름 손실이 이미 작은 체크포인트에서 시작하면 0.01).
  메모리 때문에 λ·MSE의 backward는 forward 안에서 먼저 하고, 돌려주는 손실의 보조 항은 detach 돼 있다.
- target_xy는 학습 손실에만 쓴다. 저장된 체크포인트는 일반 SmolVLAPolicy로도 읽힌다(from_pretrained는 strict=False라
  aux_head.* 키를 무시한다).

사용: python -m train.sft_aux <lerobot-train 인자...>  (train/sft_aux.sh 참고)
"""

import logging
import os

import torch
from lerobot.policies.smolvla import modeling_smolvla
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.processor import converters
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from torch import nn

AUX_KEY = "target_xy"
AUX_MAX_FRAME = int(os.environ.get("AUX_MAX_FRAME", "10"))
AUX_BS = int(os.environ.get("AUX_BS", "16"))
# 작업 공간 (x, y) 범위(m, 로봇 좌표). 이 상자를 [-1, 1]로 정규화한다.
XY_LO = torch.tensor([0.10, -0.12])
XY_HI = torch.tensor([0.30, 0.22])


def normalize_xy(xy):
    lo, hi = XY_LO.to(xy), XY_HI.to(xy)
    return 2 * (xy - lo) / (hi - lo) - 1


def unnormalize_xy(z):
    lo, hi = XY_LO.to(z), XY_HI.to(z)
    return (z + 1) / 2 * (hi - lo) + lo


def expert_hidden(model, images, img_masks, lang_tokens, lang_masks, state, x_t, time):
    """VLAFlowMatching.forward와 같은 계산으로 행동 토큰의 expert 은닉 상태 (B, chunk, H)를 float32로 돌려준다."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_out), _ = model.vlm_with_expert.forward(
        attention_mask=att_2d_masks,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
    )
    return suffix_out[:, -model.config.chunk_size :].to(dtype=torch.float32)


class SmolVLAAuxPolicy(SmolVLAPolicy):
    """SmolVLAPolicy + 보조 출력 층. 추론(select_action)은 부모와 같고 aux_head를 쓰지 않는다."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        hidden = self.model.vlm_with_expert.expert_hidden_size
        self.aux_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Linear(256, 2))
        self.aux_lambda = float(os.environ.get("AUX_LAMBDA", "0.1"))
        self.aux_log_freq = int(os.environ.get("AUX_LOG_FREQ", "100"))
        self._aux_sums, self._aux_n = [0.0, 0.0, 0.0], 0
        self.aux_batches = None  # 전처리된 초기 프레임 배치의 무한 반복자. 없으면 보조 손실 없이 흐름 손실만.

    def aux_predict(self, batch):
        """t = 1(순수 노이즈)로 forward 해서 정규화된 목표 위치 (B, 2)를 예측한다.

        상태는 0으로 바꾼다: 관절 상태로 목표를 읽는 지름길을 막아 명령(언어)과 영상만으로 맞히게 한다.
        """
        images, img_masks = self.prepare_images(batch)
        state = torch.zeros_like(self.prepare_state(batch))
        actions = self.prepare_action(batch)
        time = torch.ones(actions.shape[0], dtype=torch.float32, device=actions.device)
        x_t = self.model.sample_noise(actions.shape, actions.device)  # t=1이면 x_t = noise
        h = expert_hidden(
            self.model, images, img_masks,
            batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK],
            state, x_t, time,
        )
        return self.aux_head(h.mean(dim=1))

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        assert reduction == "mean", "sft_aux supports reduction='mean' only"
        if self.aux_batches is None or not self.training:
            return super().forward(batch, noise=noise, time=time)
        # 보조 손실은 일반 배치가 아니라 초기 프레임 배치(aux_batches)에서 계산한다. 일반 배치는 흐름 손실에만 쓴다.
        early = next(self.aux_batches)
        # 보조 forward를 먼저 하고, 학습 중이면 여기서 바로 backward 해서 그래프를 푼다. 두 forward의 그래프를
        # 동시에 들고 있으면 bs16에서 8 GB GPU를 넘는다(공유 메모리로 넘쳐 매우 느려짐). 기울기는 .grad에 쌓이고,
        # lerobot이 돌려받은 손실로 흐름 손실의 backward를 더한 뒤 옵티마이저를 한 번 밟는다.
        pred = self.aux_predict(early)
        target = early[AUX_KEY].to(device=pred.device, dtype=torch.float32)
        aux_loss = nn.functional.mse_loss(pred, normalize_xy(target))
        with torch.no_grad():
            err_cm = (unnormalize_xy(pred) - target).norm(dim=-1).mean().item() * 100
        if aux_loss.requires_grad:
            (self.aux_lambda * aux_loss).backward()
            aux_loss = aux_loss.detach()
        flow_loss, loss_dict = super().forward(batch, noise=noise, time=time)
        loss = flow_loss + self.aux_lambda * aux_loss
        loss_dict.update(flow_loss=flow_loss.item(), aux_loss=aux_loss.item(), aux_err_cm=err_cm, loss=loss.item())
        self._log_aux(loss_dict)
        return loss, loss_dict

    def _log_aux(self, d):
        for i, k in enumerate(("flow_loss", "aux_loss", "aux_err_cm")):
            self._aux_sums[i] += d[k]
        self._aux_n += 1
        if self._aux_n >= self.aux_log_freq:
            f, a, e = (s / self._aux_n for s in self._aux_sums)
            logging.info(
                f"aux: flow_loss {f:.4f} aux_loss {a:.4f} aux_err_cm {e:.2f} "
                f"(aux on early frames frame_index<{AUX_MAX_FRAME}, state zeroed; lambda {self.aux_lambda})"
            )
            self._aux_sums, self._aux_n = [0.0, 0.0, 0.0], 0


_orig_extract = converters._extract_complementary_data


def _extract_with_aux(batch):
    """전처리기가 target_xy를 버리지 않도록 complementary_data로 넘긴다."""
    data = _orig_extract(batch)
    if AUX_KEY in batch:
        data[AUX_KEY] = batch[AUX_KEY]
    return data


def early_indices(dataset, max_frame=AUX_MAX_FRAME):
    """에피소드마다 처음 max_frame 프레임(frame_index < max_frame)의 데이터셋 인덱스."""
    assert dataset.episodes is None, "early-frame sampler assumes the full dataset (no --dataset.episodes)"
    ep = dataset.meta.episodes
    return [
        i
        for f, t in zip(ep["dataset_from_index"], ep["dataset_to_index"], strict=True)
        for i in range(f, min(f + max_frame, t))
    ]


def early_loader(dataset, batch_size=AUX_BS, num_workers=1):
    """학습 데이터셋(같은 영상/delta 설정) 중 초기 프레임만 섞어 뽑는 DataLoader."""
    sub = torch.utils.data.Subset(dataset, early_indices(dataset))
    return torch.utils.data.DataLoader(
        sub, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
    )


def infinite_batches(loader, preprocessor):
    while True:  # 에폭마다 다시 섞는다(itertools.cycle은 첫 에폭을 캐시해 같은 순서를 반복함)
        for b in loader:
            yield preprocessor(b)


def install():
    """lerobot의 make_policy가 SmolVLAAuxPolicy를 만들고, 전처리기가 target_xy를 보존하도록 패치한다."""
    modeling_smolvla.SmolVLAPolicy = SmolVLAAuxPolicy  # factory.get_policy_class가 호출 시점에 이 이름을 읽는다
    converters._extract_complementary_data = _extract_with_aux


def install_early_sampler(train_module):
    """lerobot_train이 만드는 데이터셋/정책/전처리기를 가로채, 학습 시작 전에 정책에 초기 프레임 배치 반복자를 붙인다."""
    made = {}
    orig_ds, orig_pol, orig_pre = (
        train_module.make_dataset, train_module.make_policy, train_module.make_pre_post_processors,
    )

    def make_dataset(cfg):
        made["dataset"] = orig_ds(cfg)
        return made["dataset"]

    def make_policy(*a, **k):
        made["policy"] = orig_pol(*a, **k)
        return made["policy"]

    def make_pre_post_processors(*a, **k):
        pre, post = orig_pre(*a, **k)
        loader = early_loader(made["dataset"])
        logging.info(f"aux: early-frame sampler {len(loader.dataset)} frames (frame_index<{AUX_MAX_FRAME}), bs {AUX_BS}")
        made["policy"].aux_batches = infinite_batches(loader, pre)
        return pre, post

    train_module.make_dataset = make_dataset
    train_module.make_policy = make_policy
    train_module.make_pre_post_processors = make_pre_post_processors


def main():
    install()
    from lerobot.scripts import lerobot_train

    install_early_sampler(lerobot_train)
    lerobot_train.main()


if __name__ == "__main__":
    main()
