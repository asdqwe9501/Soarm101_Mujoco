"""같은 배치 × 명령 3종: 첫 '잡기 시도'가 명령 색의 공인가 (StagedReward 판정).

에피소드는 첫 시도가 판정되거나, 종료/시간 초과(StagedReward 또는 env)일 때 끝난다.
--policy smolvla: NoiseSteeredSmolVLA + N(0,1) 잠재 행동(±LATENT_SCALE로 clip), 잠재 1개당 EXEC_STEPS 스텝
                  (DSRL과 같은 실행 단위의 '조종 없음' 기준선).
--policy dsrl:    DSRLPolicyAdapter.
"""
import sys, os, argparse, json; sys.path.insert(0, '.')
import numpy as np
from collections import Counter


def run_attempt(env, sr, act_fn, obs):
    """첫 시도(또는 종료)까지 실행. 반환: (outcome, end_reason). outcome ∈ 색 | 'none' | 'timeout' | 'wrong_ball'."""
    info = {}
    while True:
        a = act_fn(obs)
        obs, _, term, trunc, _ = env.step(a)
        _, done, info = sr.step(a)
        if info["first_attempt"] is not None:
            fa = info["first_attempt"]
            return (info["attempt_color"] if fa in ("target", "other") else "none"), info["end_reason"]
        if done:  # 시도 없이 다른 공이 들림 등
            return ("timeout" if info["end_reason"] == "attempt_timeout" else info["end_reason"]), info["end_reason"]
        if term or trunc:
            return "timeout", "env_end"


class RandomLatentPolicy:
    """조종 없는 기준선: 50스텝 묶음마다 새 N(0,1) 잠재 행동."""

    def __init__(self, checkpoint, seed=0):
        from policies.dsrl import NoiseSteeredSmolVLA, LATENT_DIM, LATENT_SCALE, EXEC_STEPS
        self.steer = NoiseSteeredSmolVLA(checkpoint)
        self.rng = np.random.default_rng(seed)
        self.dim, self.scale, self.exec = LATENT_DIM, LATENT_SCALE, EXEC_STEPS
        self.queue = []

    def reset(self, env):
        self.steer.reset(env); self.queue = []

    def act(self, obs):
        if not self.queue:
            z = np.clip(self.rng.standard_normal(self.dim), -self.scale, self.scale).astype(np.float32)
            self.queue = list(self.steer.chunk(obs, z)[:self.exec])
        return self.queue.pop(0)


def main():
    from sim.env import BallSortEnv, COLORS
    from rl.staged_reward import StagedReward
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['smolvla', 'dsrl'], default='smolvla')
    ap.add_argument('--checkpoint', default=os.environ.get('CK'))
    ap.add_argument('--actor', default=None)
    ap.add_argument('--layouts', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.policy == 'dsrl':
        from policies.dsrl import DSRLPolicyAdapter
        pol = DSRLPolicyAdapter(args.checkpoint, args.actor)
    else:
        pol = RandomLatentPolicy(args.checkpoint)
    env = BallSortEnv(render=True); sr = StagedReward(env)
    conf = Counter(); lay_ok = 0
    for L in range(args.layouts):
        res = {}
        for color in COLORS:
            obs, _ = env.reset(seed=args.seed + L, options={'target': color}); pol.reset(env); sr.reset()
            res[color], _ = run_attempt(env, sr, pol.act, obs)
            conf[(color, res[color])] += 1
        lay_ok += all(res[c] == c for c in COLORS)
        print(L, res, flush=True)
    outcomes = sorted({k[1] for k in conf} | set(COLORS))
    table = {c: {o: conf[(c, o)] for o in outcomes} for c in COLORS}
    n = args.layouts
    tot = 3 * n
    summary = {
        'policy': args.policy, 'checkpoint': args.checkpoint, 'actor': args.actor,
        'layouts': n, 'seed': args.seed, 'episodes': tot,
        'confusion': table,
        'target_attempt_rate': sum(conf[(c, c)] for c in COLORS) / tot,
        'target_attempt_rate_by_color': {c: conf[(c, c)] / n for c in COLORS},
        'wrong_color_rate': sum(v for (c, o), v in conf.items() if o in COLORS and o != c) / tot,
        'miss_rate': sum(v for (c, o), v in conf.items() if o == 'none') / tot,
        'timeout_rate': sum(v for (c, o), v in conf.items() if o == 'timeout') / tot,
        'other_rate': sum(v for (c, o), v in conf.items() if o not in tuple(COLORS) + ('none', 'timeout')) / tot,
        'layouts_all_3_correct': lay_ok,
    }
    print('confusion (command -> first attempt):')
    for c in COLORS: print(' ', c, table[c])
    print({k: v for k, v in summary.items() if k.endswith('rate')}, 'all3', lay_ok, '/', n)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(summary, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
