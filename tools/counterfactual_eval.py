"""같은 배치 × 명령 3종: 처음 들어 올린 공이 명령에 따라 바뀌는가."""
import sys, os, argparse; sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from sim.env import BallSortEnv, MAX_STEPS, COLORS
ap = argparse.ArgumentParser()
ap.add_argument('--policy', choices=['smolvla', 'dsrl'], default='smolvla')
ap.add_argument('--checkpoint', default=os.environ.get('CK'))
ap.add_argument('--actor', default=None, help='--policy dsrl일 때 SAC 체크포인트(.pt)')
ap.add_argument('--n', type=int, default=int(os.environ.get('N', 12)))
args = ap.parse_args()
if args.policy == 'dsrl':
    from policies.dsrl import DSRLPolicyAdapter
    pol = DSRLPolicyAdapter(args.checkpoint, args.actor)
else:
    from policies.smolvla_policy import SmolVLAPolicyAdapter
    pol = SmolVLAPolicyAdapter(args.checkpoint)
env = BallSortEnv(render=True); sid = env.model.site('gripperframe').id
N = args.n
conf = Counter(); nearest_wrong = 0; wrong = 0; layouts_all_right = 0
for L in range(N):
    picks = {}
    for color in COLORS:
        obs, _ = env.reset(seed=20000 + L, options={'target': color}); pol.reset(env)
        tip0 = env.data.site_xpos[sid][:2].copy()
        dist = {c: np.linalg.norm(env.ball_pos(c)[:2] - tip0) for c in COLORS}
        first = None
        for t in range(400):  # 집어 올릴 때까지만
            obs, *_ = env.step(pol.act(obs))
            for c in COLORS:
                if env.ball_pos(c)[2] > 0.05: first = c
            if first: break
        picks[color] = first
        conf[(color, first)] += 1
        if first and first != color:
            wrong += 1; nearest_wrong += first == min(dist, key=dist.get)
    layouts_all_right += all(picks[c] == c for c in COLORS)
    print(L, picks, flush=True)
print('confusion (command -> first lifted):')
for c in COLORS: print(' ', c, {str(k[1]): v for k, v in conf.items() if k[0] == c})
print('wrong picks', wrong, 'of which nearest-to-gripper ball', nearest_wrong)
print('layouts where all 3 commands picked correctly', layouts_all_right, '/', N)
