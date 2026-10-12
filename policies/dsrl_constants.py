"""DSRL 상수 (torch 없이 가져올 수 있도록 분리)."""

CHUNK = 50  # SmolVLA chunk_size
ACT_DIM = 6  # 실제 관절 수
MAX_DIM = 32  # SmolVLA max_action_dim (노이즈 마지막 차원)
LATENT_DIM = CHUNK * ACT_DIM  # 노이즈 정책 출력 = 300
EXEC_STEPS = 50  # RL 한 스텝에 실행하는 행동 수 (chunk 전체, 잠재 1개당 50스텝)
LATENT_SCALE = 2.0  # 잠재 행동 범위 [-LATENT_SCALE, +LATENT_SCALE]
STEER_DIM = 12  # SAC 행동 차원: 관절 6개 × 시간 DCT 성분 2개(k=0 상수, k=1 완만한 기울기)
