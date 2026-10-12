"""같은 배치 × 명령 3종: 처음 들어 올린 공이 명령에 따라 바뀌는가."""
import sys, os; sys.path.insert(0, '.')
import numpy as np
from collections import Counter
from policies.smolvla_policy import SmolVLAPolicyAdapter
from sim.env import BallSortEnv, MAX_STEPS, COLORS
pol = SmolVLAPolicyAdapter(os.environ['CK'])
env = BallSortEnv(render=True); sid = env.model.site('gripperframe').id
N = int(os.environ.get('N', 12))
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
