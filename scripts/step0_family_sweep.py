"""
扫描三个族的结构参数，量出各自的统计特性，用来挑一组"结构强度可比"的参数。

为什么需要这一步
----------------
三个族的参数现在只存在于函数签名默认值里，彼此没有标定过：
    renewal shape=0.5  -> CV ~= 2.24
    hawkes  branching=0.5
    lrd     hurst=0.75
"结构强度"是否相当完全未知。若某族太弱（接近 Poisson），族间差异会被
C1/C2 而非 C3 解释；若某族太强，又会把队列打满、目标量被丢弃截断。

本脚本只用生成器 + 最终的特征函数（features.py），不接模拟器。

用法（在 snn-proposal/ 目录下）：
  env\\python.exe scripts\\step0_family_sweep.py
  env\\python.exe scripts\\step0_family_sweep.py --n-events 10000 --n-streams 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import CONFIG  # noqa: E402
from features import autocorr_lag1, burst_poisson, cv, fano, mean_rate  # noqa: E402
from generators_multi import generate  # noqa: E402
from generators_poisson import generate_poisson  # noqa: E402
from timescale import TimeScale  # noqa: E402

# 每个族要扫的参数网格
GRID = {
    "renewal": ("shape", [0.3, 0.5, 0.7, 0.85]),
    "hawkes": ("branching", [0.3, 0.5, 0.7, 0.9]),
    "lrd": ("hurst", [0.6, 0.7, 0.8, 0.9]),
}


def summarize(ts, sc):
    """对一条流算全部 C1 统计量。"""
    b = burst_poisson(ts, sc)
    return {
        "rate": mean_rate(ts),
        "cv": cv(ts),
        "fano": fano(ts, sc),
        "burst_per_event": b["burst_per_event"],
        "mean_surprise": b["mean_surprise"],
        "acf_lag1": autocorr_lag1(ts, sc),
        "n_burst": b["n_burst"],
    }


def scan(family, param, value, lam, n_events, n_streams, seed0=0):
    """对一组参数跑 n_streams 条流，返回统计量的均值/标准差与生成耗时。"""
    rows = []
    gen_ms = []
    for s in range(n_streams):
        t0 = time.perf_counter()
        if family == "poisson":
            ts = generate_poisson(lam, n_events, seed=seed0 + s)
        else:
            ts = generate(family, lam, n_events, seed=seed0 + s, **{param: value})
        gen_ms.append((time.perf_counter() - t0) * 1000.0)

        sc = TimeScale.from_events(ts)
        if not sc.valid:
            continue
        rows.append(summarize(ts, sc))

    keys = ("rate", "cv", "fano", "burst_per_event", "mean_surprise", "acf_lag1", "n_burst")
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        agg[k] = float(vals.mean()) if vals.size else float("nan")
        agg[k + "_sd"] = float(vals.std()) if vals.size else float("nan")
    agg["gen_ms"] = float(np.mean(gen_ms))
    agg["n_ok"] = len(rows)
    agg["family"] = family
    agg["param"] = param
    agg["value"] = value
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=10_000)
    ap.add_argument("--n-streams", type=int, default=20)
    ap.add_argument("--lambda-s", type=float, default=0.035,
                    help="目标事件率（config.lambda_s_targets 的中间档）")
    ap.add_argument("--out", default=os.path.join("results", "step0_family_sweep.json"))
    args = ap.parse_args()

    lam = args.lambda_s
    print(f"lambda_s = {lam} (mean ISI = {1/lam:.1f} cycles), "
          f"{args.n_events} events x {args.n_streams} streams\n")

    results = [scan("poisson", "-", 0.0, lam, args.n_events, args.n_streams)]

    for fam, (param, values) in GRID.items():
        for v in values:
            print(f"  scanning {fam} {param}={v} ...", flush=True)
            results.append(scan(fam, param, v, lam, args.n_events, args.n_streams))

    # ---- 表格 ----
    hdr = f"{'family':9s} {'param':10s} {'value':>6s} {'rate':>8s} {'cv':>7s} " \
          f"{'fano':>7s} {'burst/ev':>9s} {'acf1':>8s} {'gen_ms':>8s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        pname = r["param"] if r["param"] != "-" else ""
        print(f"{r['family']:9s} {pname:10s} {r['value']:6.2f} "
              f"{r['rate']:8.5f} {r['cv']:7.3f} {r['fano']:7.3f} "
              f"{r['burst_per_event']:9.5f} {r['acf_lag1']:8.4f} {r['gen_ms']:8.1f}")

    # ---- 与 Poisson 的距离（用于挑"结构强度相当"的组合） ----
    base = results[0]
    print("\n=== 相对 Poisson 的结构强度（越大越'非 Poisson'）===")
    hdr2 = f"{'family':9s} {'value':>6s} {'dCV':>7s} {'dFano':>7s} {'burstx':>8s} {'|acf1|':>8s} {'score':>8s}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in results[1:]:
        d_cv = abs(r["cv"] - base["cv"])
        d_fano = abs(r["fano"] - base["fano"])
        burst_x = r["burst_per_event"] / max(base["burst_per_event"], 1e-9)
        acf = abs(r["acf_lag1"])
        score = d_cv + d_fano + np.log10(max(burst_x, 1.0)) + acf
        print(f"{r['family']:9s} {r['value']:6.2f} {d_cv:7.3f} {d_fano:7.3f} "
              f"{burst_x:8.2f} {acf:8.4f} {score:8.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"lambda_s": lam, "n_events": args.n_events,
                   "n_streams": args.n_streams, "rows": results},
                  f, indent=2, ensure_ascii=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
