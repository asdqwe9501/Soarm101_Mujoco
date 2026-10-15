"""팔이 제자리(첫 프레임)일 때 보조 출력 층이 명령한 색 공의 위치를 맞히는지, 명령에 따라 예측이 바뀌는지 본다.

상태는 SmolVLAAuxPolicy.aux_predict 안에서 학습 때와 똑같이 0으로 바뀐다.
판정(스모크 관문): 같은 장면에서 명령 3개의 예측 간 평균 거리 ≥ 5 cm, 명령 공에 가장 가까운 비율 ≥ 60%(우연 33%).
사용: python tools/aux_frame0_diag.py --checkpoint <pretrained_model> [--layouts 15] [--commands basic|heldout] [--out x.json]
"""
import sys, os, argparse, json; sys.path.insert(0, '.')
import numpy as np, torch
from lerobot.policies.factory import make_pre_post_processors
from train.sft_aux import SmolVLAAuxPolicy, unnormalize_xy
from policies.smolvla_policy import RENAME_MAP
from sim.env import BallSortEnv, COLORS
from sim.commands import command_for_tier, sample_command


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--layouts', type=int, default=15)
    ap.add_argument('--commands', choices=['basic', 'heldout'], default='basic')
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    ck = args.checkpoint
    pol = SmolVLAAuxPolicy.from_pretrained(ck).to('cuda').eval()
    pre, _ = make_pre_post_processors(pol.config, pretrained_path=ck, preprocessor_overrides={
        'device_processor': {'device': 'cuda'}, 'rename_observations_processor': {'rename_map': RENAME_MAP}})
    env = BallSortEnv(render=True); rng = np.random.default_rng(args.seed + 7)
    errs, spread, nearest_true = [], [], 0
    for L in range(args.layouts):
        preds = {}
        for c in COLORS:
            cmd = command_for_tier(c, 'basic', rng) if args.commands == 'basic' else sample_command(c, rng, split='heldout')
            obs, _ = env.reset(seed=args.seed + L, options={'target': c, 'command': cmd})
            b = {'observation.state': torch.from_numpy(obs['observation.state'])[None], 'task': [env.task]}
            for k in ('observation.images.top', 'observation.images.front'):
                b[k] = torch.from_numpy(obs[k]).permute(2, 0, 1).float().div(255)[None]
            b = pre(b)
            b['action'] = torch.zeros(1, pol.config.chunk_size, 6, device='cuda')
            with torch.no_grad():
                p = unnormalize_xy(pol.aux_predict(b))[0].cpu().numpy()
            preds[c] = p
            errs.append(float(np.linalg.norm(p - env.ball_pos(c)[:2]) * 100))
            d = {cc: np.linalg.norm(p - env.ball_pos(cc)[:2]) for cc in COLORS}
            nearest_true += min(d, key=d.get) == c
        spread.append(float(np.mean([np.linalg.norm(preds[a] - preds[b]) for a in COLORS for b in COLORS if a < b]) * 100))
    n = len(errs)
    s = {'checkpoint': ck, 'layouts': args.layouts, 'commands': args.commands, 'n': n,
         'err_cm_mean': float(np.mean(errs)), 'err_cm_median': float(np.median(errs)),
         'nearest_correct': int(nearest_true), 'nearest_correct_rate': nearest_true / n,
         'spread_cm_mean': float(np.mean(spread)),
         'gate_pass': bool(np.mean(spread) >= 5 and nearest_true / n >= 0.6)}
    print(f"frame0 aux error: mean {s['err_cm_mean']:.1f} cm (median {s['err_cm_median']:.1f})")
    print(f"prediction nearest to the commanded ball: {nearest_true}/{n}")
    print(f"mean distance between predictions for the 3 commands in the same scene: {s['spread_cm_mean']:.1f} cm")
    print(f"gate (spread>=5cm and nearest>=60%): {'PASS' if s['gate_pass'] else 'FAIL'}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(s, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
