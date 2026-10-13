# LoRA smoke test (SmolVLA VLM attention + expert)

Command: `bash train/sft_lora.sh <job> <steps> <bs> [extra]` (wraps `train/sft.sh`;
INIT = sft_v3 040000 = `last`, DATASET = so101_ballsort_sim_v5).

## Targets (regex, PEFT fullmatch on module names)
```
(model\.vlm_with_expert\.vlm\.model\.text_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj
|model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj
|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))
```
- VLM text model: 16 layers x q,k,v,o = 64 Linear (vision encoder 12 layers + connector NOT targeted, frozen)
- Expert: 16 layers x q,v = 32 Linear; plus 5 projections (SmolVLA defaults)
- r=16, lora_alpha=8 (PEFT default; lerobot's PeftConfig has no alpha field -> scale 0.5), dropout 0,
  full_training_modules=[] (no fully trained modules)
- Trainable params: 2,381,056 / 452,427,232 (0.53%)

## Gradient flow (1 backward, bs 2, sft_v3 + LoRA, train_expert_only=True)
SmolVLA's forward has no no_grad/detach on the VLM path; train_expert_only only sets
requires_grad=False on base weights + vlm.eval() (no dropout effect: lora_dropout=0). So the default
`train_expert_only=true` is kept. lora_B grads nonzero: VLM attn 62/64, expert 32/32, projections 5/5.
The 2 zeros are text layer 15 q_proj/o_proj: the last VLM layer's attention output is unused
(the expert only reads its K/V), so they are dead but harmless. (lora_A grads are 0 at step 0 since B is zero-initialised.)

## Memory / speed (RTX 5070 Laptop 8 GB, 50 steps)
| batch | peak GPU mem (nvidia-smi) | update s/step | data s/step |
|------:|--------------------------:|--------------:|------------:|
| 16    | 6.88 GB                   | 0.77          | 0.01        |

Chosen batch: 16 (same as sft_v3). Smoke loss 0.010-0.018.
15000 steps ~ 3.3 h.

## Eval
`policies/smolvla_policy.py` `load_smolvla` detects `adapter_config.json`, loads the base from
`base_model_name_or_path` (repo-root fallback for relative paths), applies the adapter and
`merge_and_unload()`. Processors come from the PEFT checkpoint folder (lerobot saves them there,
with the v5 dataset normalization stats).
