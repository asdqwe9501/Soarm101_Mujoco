"""DSRL 상한선: 같은 장면·명령에 N(0,1) 잠재 시퀀스를 K번 시도, 하나라도 첫 시도가 target이면 도달 가능."""
import sys, os, argparse, json, time; sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
from counterfactual_eval import run_attempt, RandomLatentPolicy


def command_for(color, commands, rng):
    """(장면, 명령) 쌍마다 한 번 뽑는 명령문. canonical이면 None(env 기본 문장)."""
    from sim.commands import command_for_tier, sample_command
    if commands == 'canonical':
        return None
    if commands == 'basic':
        return command_for_tier(color, 'basic', rng)
    return sample_command(color, rng, split=commands)


def main():
    from sim.env import BallSortEnv, COLORS
    from rl.staged_reward import StagedReward
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.environ.get('CK', 'outputs/train/sft_v3/checkpoints/last/pretrained_model'))
    ap.add_argument('--layouts', type=int, default=30)
    ap.add_argument('--tries', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--commands', choices=['canonical', 'basic', 'train', 'heldout'], default='canonical')
    args = ap.parse_args()
    import numpy as np
    cmd_rng = np.random.default_rng(args.seed + 7)
    pol = RandomLatentPolicy(args.checkpoint, seed=1)  # rng/gen은 호출 사이에 이어지므로 시도마다 독립
    env = BallSortEnv(render=True); sr = StagedReward(env)
    reach = {c: 0 for c in COLORS}; hits = {c: 0 for c in COLORS}; per = []
    t0 = time.time()
    for L in range(args.layouts):
        for color in COLORS:
            outs = []
            cmd = command_for(color, args.commands, cmd_rng)
            opts = {'target': color} if cmd is None else {'target': color, 'command': cmd}
            for k in range(args.tries):
                obs, _ = env.reset(seed=args.seed + L, options=opts); pol.reset(env); sr.reset()
                o, _ = run_attempt(env, sr, pol.act, obs)
                outs.append(o)
            h = sum(o == color for o in outs)
            hits[color] += h; reach[color] += h > 0
            per.append({'layout': L, 'command': color, 'text': cmd, 'outcomes': outs, 'hits': h})
        print(L, {c: (reach[c], hits[c]) for c in COLORS}, f'{time.time()-t0:.0f}s', flush=True)
    n = args.layouts
    summary = {
        'checkpoint': args.checkpoint, 'layouts': n, 'tries': args.tries, 'seed': args.seed, 'commands': args.commands,
        'reachable_rate': sum(reach.values()) / (3 * n),
        'reachable_rate_by_color': {c: reach[c] / n for c in COLORS},
        'per_try_target_rate': sum(hits.values()) / (3 * n * args.tries),
        'unreached': [(p['layout'], p['command']) for p in per if p['hits'] == 0],
        'per_try_target_rate_by_color': {c: hits[c] / (n * args.tries) for c in COLORS},
        'runtime_s': time.time() - t0, 'detail': per,
    }
    print({k: v for k, v in summary.items() if k != 'detail'})
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(summary, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
