"""같은 배치 × 명령 3종: 첫 '잡기 시도'가 명령 색의 공인가 (StagedReward 판정).

에피소드는 첫 시도가 판정되거나, 종료/시간 초과(StagedReward 또는 env)일 때 끝난다.
--policy smolvla: NoiseSteeredSmolVLA + N(0,1) 잠재 행동(±LATENT_SCALE로 clip), 잠재 1개당 EXEC_STEPS 스텝
                  (DSRL과 같은 실행 단위의 '조종 없음' 기준선).
--policy dsrl:    DSRLPolicyAdapter.
--guard:          정책을 ColorGuard(손목 카메라 색 확인 + 재시도)로 감싼다. 집계는 실제로 실행된 행동에 StagedReward를
                  적용하므로 가드가 막은 닫기는 시도로 세지 않는다. 가드 판별마다 그 순간 노린 공(StagedReward 기준,
                  평가용 정답)을 기록해 거부가 맞았는지 센다.
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
    from sim.commands import command_for_tier, sample_command
    from sim.env import BallSortEnv, COLORS
    from rl.staged_reward import StagedReward
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', choices=['smolvla', 'dsrl'], default='smolvla')
    ap.add_argument('--checkpoint', default=os.environ.get('CK'))
    ap.add_argument('--actor', default=None)
    ap.add_argument('--layouts', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20000)
    ap.add_argument('--out', default=None)
    ap.add_argument('--guard', action='store_true')
    ap.add_argument('--records', action='store_true', help='에피소드별 기록(명령문, 결과, 공·통 위치)을 결과에 넣는다')
    ap.add_argument('--commands', choices=['canonical', 'basic', 'train', 'heldout'], default='canonical',
                    help="명령문: canonical(예전 고정 문장), basic(Grab the X ball.), train(학습 배분대로), heldout(평가 전용 표현)")
    args = ap.parse_args()
    cmd_rng = np.random.default_rng(args.seed + 7)
    if args.policy == 'dsrl':
        from policies.dsrl import DSRLPolicyAdapter
        pol = DSRLPolicyAdapter(args.checkpoint, args.actor)
    else:
        pol = RandomLatentPolicy(args.checkpoint)
    if args.guard:
        from policies.color_guard import ColorGuard
        pol = ColorGuard(pol)
    env = BallSortEnv(render=True); sr = StagedReward(env)
    conf = Counter(); lay_ok = 0; guard_log = []; records = []

    def act_fn(obs):
        a = pol.act(obs)
        if args.guard and len(pol.stats['checks']) > len(guard_log[-1]['checks']):
            chk = dict(pol.stats['checks'][-1], aimed=sr._aimed_ball())  # 평가용 정답: 지금 집게가 노린 공
            guard_log[-1]['checks'].append(chk)
        return a
    for L in range(args.layouts):
        res = {}
        for color in COLORS:
            opts = {'target': color}
            if args.commands != 'canonical':
                opts['command'] = (command_for_tier(color, 'basic', cmd_rng) if args.commands == 'basic'
                                   else sample_command(color, cmd_rng, split=args.commands))
            obs, _ = env.reset(seed=args.seed + L, options=opts); pol.reset(env); sr.reset()
            rec = {'layout': L, 'seed': args.seed + L, 'color': color, 'command': env.task,
                   'balls': {c: env.ball_pos(c)[:2].round(4).tolist() for c in COLORS},
                   'bin': env.bin_pos()[:2].round(4).tolist()}
            guard_log.append({'layout': L, 'command': color, 'checks': []})
            res[color], _ = run_attempt(env, sr, act_fn, obs)
            guard_log[-1]['outcome'] = res[color]
            conf[(color, res[color])] += 1
            records.append(dict(rec, outcome=res[color]))
        lay_ok += all(res[c] == c for c in COLORS)
        print(L, res, flush=True)
    outcomes = sorted({k[1] for k in conf} | set(COLORS))
    table = {c: {o: conf[(c, o)] for o in outcomes} for c in COLORS}
    n = args.layouts
    tot = 3 * n
    summary = {
        'policy': args.policy, 'checkpoint': args.checkpoint, 'actor': args.actor,
        'layouts': n, 'seed': args.seed, 'episodes': tot, 'commands': args.commands,
        'confusion': table,
        'target_attempt_rate': sum(conf[(c, c)] for c in COLORS) / tot,
        'target_attempt_rate_by_color': {c: conf[(c, c)] / n for c in COLORS},
        'wrong_color_rate': sum(v for (c, o), v in conf.items() if o in COLORS and o != c) / tot,
        'miss_rate': sum(v for (c, o), v in conf.items() if o == 'none') / tot,
        'timeout_rate': sum(v for (c, o), v in conf.items() if o == 'timeout') / tot,
        'other_rate': sum(v for (c, o), v in conf.items() if o not in tuple(COLORS) + ('none', 'timeout')) / tot,
        'layouts_all_3_correct': lay_ok,
    }
    if args.records:
        summary['records'] = records
    if args.guard:
        checks = [c for e in guard_log for c in e['checks']]
        rej = [c for c in checks if c['rejected']]
        passed = [c for c in checks if not c['rejected']]
        summary['guard'] = {
            'checks': len(checks),
            'rejections': len(rej),
            'episodes_with_rejection': sum(any(c['rejected'] for c in e['checks']) for e in guard_log),
            'rejections_correct': sum(c['aimed'] is not None and c['aimed'] != c['commanded'] for c in rej),
            'rejections_of_target': sum(c['aimed'] == c['commanded'] for c in rej),
            'rejections_no_ball_aimed': sum(c['aimed'] is None for c in rej),
            'passed_wrong_ball': sum(c['aimed'] is not None and c['aimed'] != c['commanded'] for c in passed),
            'passed_wrong_ball_seen': Counter(str(c['seen']) for c in passed
                                              if c['aimed'] is not None and c['aimed'] != c['commanded']),
            'seen_vs_aimed': Counter(f"{c['seen']}|{c['aimed']}" for c in checks),
            'episodes': guard_log,
        }
        print({k: v for k, v in summary['guard'].items() if k != 'episodes'})
    print('confusion (command -> first attempt):')
    for c in COLORS: print(' ', c, table[c])
    print({k: v for k, v in summary.items() if k.endswith('rate')}, 'all3', lay_ok, '/', n)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(summary, open(args.out, 'w'), indent=2)


if __name__ == '__main__':
    main()
