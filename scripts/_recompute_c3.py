"""
只重算 C3，用论文 [1] 完整的 q 网格（q = 0, 0.25, ..., 5，共 21 个点）。

为什么可以只重算这一部分
------------------------
C3（H(q)、α(q=0)、谱宽）只依赖**事件流**，不依赖模拟结果：
    事件流 -> ISI -> MFDFA -> H(q)          <-- 本脚本
    事件流 -> 目的地 -> 消息 -> 排队 -> 延迟  <-- 不需要
而生成器是确定性的（同一 seed 给同一流），dataset.py 用的是
`seed = run_id * 1000 + src`，所以可以精确复现。

脚本会断言重生成的流长度 == 原表里的 `n_events_actual`；
若不一致就报错退出（说明 seed 或族参数变了），而不是静默合并。

用法：
  env\\python.exe scripts\\_recompute_c3.py
  env\\python.exe scripts\\_recompute_c3.py --csv results/dataset_n10.csv --out results/dataset_n10_v2.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import CONFIG  # noqa: E402
from generators_multi import generate  # noqa: E402
from generators_poisson import generate_poisson  # noqa: E402
from mfdfa import mfdfa, spectrum_width  # noqa: E402

FAMILY_PARAMS = {
    "poisson": {},
    "renewal": {"shape": 0.5},
    "hawkes": {"branching": 0.7},
    "lrd": {"hurst": 0.6},
}

Q = np.arange(0.0, 5.0 + 1e-9, 0.25)                      # 21 个
QCOLS = [f"Hq_{int(round(q * 100)):03d}" for q in Q]      # Hq_000 .. Hq_500
OLD_HQ = [f"Hq_{i}" for i in range(1, 6)]                  # 旧的 5 点列，将被替换


def gen_stream(family, lam, n_events, seed):
    if family == "poisson":
        return generate_poisson(lam, n_events, seed=seed)
    return generate(family, lam, n_events, seed=seed, **FAMILY_PARAMS[family])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("results", "dataset_n10.csv"))
    ap.add_argument("--out", default=os.path.join("results", "dataset_n10_v2.csv"))
    ap.add_argument("--lam", type=float, default=None)
    args = ap.parse_args()

    lam = CONFIG["lambda_s"] if args.lam is None else args.lam
    n_events = CONFIG["events_per_source"]

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    print(f"{args.csv}: {len(rows)} rows")
    print(f"q grid = {len(Q)} points: {Q[0]} .. {Q[-1]} step {Q[1]-Q[0]}")
    print(f"n_events = {n_events}, lambda = {lam}\n")

    t0 = time.perf_counter()
    mismatched, degenerate, invalid = [], 0, 0

    for i, r in enumerate(rows):
        fam, run_id, src = r["family"], int(r["run"]), int(r["src"])
        expected_n = int(r["n_events_actual"])
        ts = gen_stream(fam, lam, n_events, seed=run_id * 1000 + src)
        if len(ts) != expected_n:
            mismatched.append((fam, run_id, src, expected_n, len(ts)))

        res = mfdfa(np.diff(ts), q=Q)
        if not res["valid"]:
            invalid += 1
            for c in QCOLS:
                r[c] = ""
            r["alpha_q0"] = ""
            r["spec_width"] = ""
            r["c3_degenerate"] = ""
            continue

        for c, v in zip(QCOLS, res["Hq"]):
            r[c] = float(v)
        qm = Q[:-1]
        r["alpha_q0"] = float(res["hq"][int(np.argmin(np.abs(qm - 0.0)))])
        r["spec_width"] = spectrum_width(res)
        r["c3_degenerate"] = bool(res["degenerate"])
        if res["degenerate"]:
            degenerate += 1

        if (i + 1) % 250 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{len(rows)}  ({el:.0f}s, "
                  f"eta {el/(i+1)*(len(rows)-i-1):.0f}s)", flush=True)

    if mismatched:
        print(f"\n[FATAL] {len(mismatched)} stream-length mismatches — "
              f"seed or family params changed; refusing to merge.")
        for m in mismatched[:5]:
            print("   ", m)
        sys.exit(1)

    # 列顺序：原列去掉旧的 5 点 Hq，再追加 21 个新列
    base_cols = [c for c in rows[0].keys() if c not in OLD_HQ]
    # 保证 C3 列紧跟在 spec_width 之后
    cols = base_cols + QCOLS
    # 去重（QCOLS 可能已在 base_cols 里，重复运行需要幂等）
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    cols = ordered

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    el = time.perf_counter() - t0
    meta = {
        "source_csv": args.csv, "out_csv": args.out,
        "n_rows": len(rows), "q_grid": [float(x) for x in Q],
        "q_cols": QCOLS, "n_features_total": len(cols),
        "invalid": invalid, "degenerate": degenerate,
        "wall_s": el,
    }
    meta_path = os.path.splitext(args.out)[0] + "_c3meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {args.out}  ({len(cols)} columns, {el:.0f}s)")
    print(f"  valid={len(rows)-invalid}  degenerate={degenerate}  invalid={invalid}")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
