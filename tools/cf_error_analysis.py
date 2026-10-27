"""반사실 시험 오류 원인 분석: docs/results/erran/의 기록(cf_*.json)과 재현 시험(repeat_*.json)을 집계한다.

python tools/cf_error_analysis.py --dir docs/results/erran --out docs/results/erran/analysis.md

- 색 정확도: 공을 노린 시도 중 명령한 색의 비율 (공 없는 곳 시도는 제외)
- 오류 쌍(명령 → 노린 색), 명령 표현별 오류율
- 장면 기하: 명령한 공과 다른 공 사이 거리, 로봇 기준 거리·방향
- 모델 간 공통 오류: 같은 장면 + 같은 명령에서 여러 모델이 틀렸는가 (명령문은 seed가 같아 모델 간 동일)
- 재현 시험: 오류 사례를 노이즈만 바꿔 반복했을 때 맞는 비율 (비교: 정답 사례)
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

COLORS = ["red", "black", "sky blue"]
COLOR_WORD = re.compile(r"\b(red|black|blue)(-colored)?\b")


def template(cmd):
    return COLOR_WORD.sub("{c}", cmd.strip().rstrip(".").lower())


def geom(r, aimed=None):
    b = {c: np.array(v) for c, v in r["balls"].items()}
    t = b[r["color"]]
    others = [c for c in COLORS if c != r["color"]]
    d_other = {c: float(np.linalg.norm(b[c] - t)) for c in others}
    g = {"r_target": float(np.linalg.norm(t)), "ang_target": float(np.degrees(np.arctan2(t[1], t[0]))),
         "min_gap": min(d_other.values())}
    if aimed in d_other:
        g["gap_to_aimed"] = d_other[aimed]
        g["aimed_is_nearest_other"] = d_other[aimed] == g["min_gap"]
        g["aimed_closer_to_robot"] = float(np.linalg.norm(b[aimed])) < g["r_target"]
    return g


def is_wrong(r):
    return r["outcome"] in COLORS and r["outcome"] != r["color"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/results/erran")
    ap.add_argument("--out", default="docs/results/erran/analysis.md")
    args = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(f"{args.dir}/cf_*.json")):
        name = Path(f).stem[3:]  # 예: aux2_heldout
        runs[name] = json.load(open(f))["records"]
    L = ["# 반사실 시험 오류 원인 분석", ""]

    L += ["## 1. 모델별 요약 (배치 100 × 명령 3 = 300회)", "",
          "| 실행 | 맞는 공 | 다른 색 | 공 없음/시간초과 | 색 정확도 (공을 노린 시도 중) | 오류 쌍 (명령→노린 색) |", "|---|---|---|---|---|---|"]
    for name, recs in runs.items():
        n = len(recs)
        ok = sum(r["outcome"] == r["color"] for r in recs)
        wr = [r for r in recs if is_wrong(r)]
        none = n - ok - len(wr)
        pairs = Counter(f"{r['color']}→{r['outcome']}" for r in wr)
        L.append(f"| {name} | {ok} ({ok / n:.1%}) | {len(wr)} ({len(wr) / n:.1%}) | {none} | "
                 f"{ok / max(ok + len(wr), 1):.1%} | {dict(pairs)} |")

    L += ["", "## 2. 명령 표현별 오류율", "", "| 실행 | 표현 | 시도 | 다른 색 | 오류율 |", "|---|---|---|---|---|"]
    for name, recs in runs.items():
        by = defaultdict(list)
        for r in recs:
            by[template(r["command"])].append(is_wrong(r))
        for t, v in sorted(by.items()):
            L.append(f"| {name} | `{t}` | {len(v)} | {sum(v)} | {np.mean(v):.1%} |")

    L += ["", "## 3. 장면 기하 (오류 vs 정답)", "",
          "| 실행 | 그룹 | 개수 | 명령한 공과 가장 가까운 다른 공 거리 (cm, 평균) | 로봇에서 명령한 공까지 거리 (cm) | 노린 공이 가장 가까운 다른 공 | 노린 공이 로봇에 더 가까움 |",
          "|---|---|---|---|---|---|---|"]
    for name, recs in runs.items():
        for grp, rs in (("오류", [r for r in recs if is_wrong(r)]), ("정답", [r for r in recs if r["outcome"] == r["color"]])):
            if not rs:
                continue
            gs = [geom(r, r["outcome"] if grp == "오류" else None) for r in rs]
            near = [g["aimed_is_nearest_other"] for g in gs if "aimed_is_nearest_other" in g]
            closer = [g["aimed_closer_to_robot"] for g in gs if "aimed_closer_to_robot" in g]
            L.append(f"| {name} | {grp} | {len(rs)} | {100 * np.mean([g['min_gap'] for g in gs]):.1f} | "
                     f"{100 * np.mean([g['r_target'] for g in gs]):.1f} | "
                     f"{f'{sum(near)}/{len(near)}' if near else '-'} | {f'{sum(closer)}/{len(closer)}' if closer else '-'} |")

    L += ["", "## 4. 모델 간 공통 오류 (같은 장면 + 같은 명령)", ""]
    for cmds in ("basic", "heldout"):
        sets = {n.split("_")[0]: {(r["layout"], r["color"]) for r in recs if is_wrong(r)}
                for n, recs in runs.items() if n.endswith(cmds)}
        if not sets:
            continue
        cnt = Counter(k for s in sets.values() for k in s)
        L.append(f"- {cmds}: 오류 사례 {len(cnt)}개 중 2개 이상 모델이 함께 틀린 사례 {sum(v >= 2 for v in cnt.values())}개, "
                 f"3개 모두 {sum(v >= 3 for v in cnt.values())}개 (모델별 오류 수: { {k: len(v) for k, v in sets.items()} })")

    L += ["", "## 5. 재현 시험 (같은 장면 + 같은 명령, 노이즈만 바꿔 10회)", "",
          "| 실행 | 오류 사례 수 | 오류 사례 반복 시 맞는 비율 | 같은 오류 반복 비율 | 정답 사례 반복 시 맞는 비율 | 오류 사례 중 반복의 절반 이상 틀림 |",
          "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{args.dir}/repeat_*.json")):
        d = json.load(open(f))
        err = [c for c in d["cases"] if c["group"] == "error"]
        ctl = [c for c in d["cases"] if c["group"] == "control"]
        if not err:
            L.append(f"| {Path(f).stem[7:]} | 0 | - | - | - | - |")
            continue
        L.append(f"| {Path(f).stem[7:]} | {len(err)} | {np.mean([c['repeat_correct'] for c in err]):.1%} | "
                 f"{np.mean([c['repeat_same_wrong'] for c in err]):.1%} | "
                 f"{np.mean([c['repeat_correct'] for c in ctl]):.1%} | {sum(c['repeat_correct'] < 0.5 for c in err)} |")
    L += ["", "### 반복의 절반 이상 틀린 오류 사례 (체계적 오류 후보)", ""]
    for f in sorted(glob.glob(f"{args.dir}/repeat_*.json")):
        for c in json.load(open(f))["cases"]:
            if c["group"] == "error" and c["repeat_correct"] < 0.5:
                g = geom(c, c["outcome"])
                L.append(f"- {Path(f).stem[7:]} 배치 {c['layout']} `{c['command']}` → {c['outcome']}, 반복 {Counter(c['repeats'])}, "
                         f"공 간격 {100 * g['min_gap']:.1f}cm, 노린 공까지 {100 * g.get('gap_to_aimed', float('nan')):.1f}cm")

    text = "\n".join(L) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
