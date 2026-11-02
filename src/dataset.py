"""
dataset.py —— 组装 G1 的样本表：**一行 = (family, run, src)**。

每行包含
--------
特征（嵌套的 C0 ⊂ C1 ⊂ C2 ⊂ C3）
    C0: spike_count, mean_rate
    C1: cv, fano, burst_per_event, autocorr_lag1
    C2: fanout_mean, hop_mean, hop_std, hop_p95, cut_weight, locality
    C3: Hq_1..Hq_5, alpha_q0, spec_width      ← 只用 q>=0（论文 [1] 的做法）

目标量
    p95_delay       该源发出的消息的 95 分位延迟（cycles）
    p50_delay       中位数（诊断用）
    peak_occupancy  该源消息路径经过的端口的最大峰值占用（flits）

诊断/控制量
    n_events_actual 实际事件数（**LRD 会少于 n_events**，约 -1.7%）
    n_messages      该源产生的消息数
    n_dropped       该源被丢弃的消息数
    rho_avg         平均链路利用率 = total_hops / (n_dir_links × T)
                    —— 用于验证"四族负载确实相同"

为什么按源切
------------
64 个源共享同一批端口队列，目标量在源之间是耦合的；**独立单位是"运行"**，
所以训练时必须用 GroupKFold(group=run)。本文件只负责把每个源的特征与
目标量算出来，不负责划分。

用法
----
  env\\python.exe src\\dataset.py --runs 2 --families poisson --n-sources 8
  env\\python.exe src\\dataset.py --runs 40 --out results/dataset.csv
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import CONFIG  # noqa: E402
from destinations import destinations_for_stream, sigma_from_config  # noqa: E402
from features import (  # noqa: E402
    autocorr_lag1, burst_poisson, c2_features, cv, fano, mean_rate, spike_count,
)
from generators_multi import generate  # noqa: E402
from generators_poisson import generate_poisson  # noqa: E402
from mfdfa import mfdfa, spectrum_width  # noqa: E402
from multicast import build_messages  # noqa: E402
from noc_sim import simulate_network  # noqa: E402
from timescale import TimeScale  # noqa: E402
from topology import Mesh  # noqa: E402

# 三族的参数（step0_family_sweep 的 CV 对齐结果）
FAMILY_PARAMS = {
    "poisson": {},
    "renewal": {"shape": 0.5},
    "hawkes": {"branching": 0.7},
    "lrd": {"hurst": 0.6},
}

# C3：只用 q >= 0（论文 [1]："negative q ... due to limited data"）
C3_Q = np.arange(0.0, 5.0 + 1e-9, 0.25)      # 21 个
C3_HQ = (1.0, 2.0, 3.0, 4.0, 5.0)
C3_NAMES = [f"Hq_{int(q)}" for q in C3_HQ] + ["alpha_q0", "spec_width"]


def gen_stream(family, lam, n_events, seed):
    if family == "poisson":
        return generate_poisson(lam, n_events, seed=seed)
    return generate(family, lam, n_events, seed=seed, **FAMILY_PARAMS[family])


def c3_of(ts):
    """从 ISI 序列算 7 维 C3（Hq_1..5 + α(q=0) + 谱宽）。"""
    res = mfdfa(ts, q=C3_Q)
    if not res["valid"]:
        return {k: np.nan for k in C3_NAMES}, False
    q, Hq, hq = res["q"], res["Hq"], res["hq"]
    out = {}
    for qv in C3_HQ:
        i = int(np.argmin(np.abs(q - qv)))
        out[f"Hq_{int(qv)}"] = float(Hq[i])
    i0 = int(np.argmin(np.abs(q[:-1] - 0.0)))     # hq 对应 q[:-1]
    out["alpha_q0"] = float(hq[i0])
    out["spec_width"] = spectrum_width(res)
    return out, bool(res["degenerate"])


def run_one(family, run_id, topo, table, lam, n_events, n_sources, with_c3=True):
    """跑一次完整仿真，返回该次所有源的样本行。"""
    rng = np.random.default_rng(run_id)
    msgs, ranges, streams = [], [], []

    for src in range(n_sources):
        ts = gen_stream(family, lam, n_events, seed=run_id * 1000 + src)
        dsts_list = destinations_for_stream(src, len(ts), topo, rng=rng)
        streams.append((src, ts, dsts_list))
        start = len(msgs)
        for t, dsts in zip(ts, dsts_list):
            msgs.extend(build_messages(float(t), src, dsts, topo,
                                       mode=CONFIG["multicast_mode"],
                                       path_table=table))
        ranges.append((start, len(msgs), src))

    sim = simulate_network(
        msgs, topo.n_nodes, topo,
        n_ports=CONFIG["n_output_ports"],
        port_capacity=CONFIG["port_capacity"],
        hops_cycles=CONFIG["hops_cycles"],
        record_occupancy=True,
    )
    occ = sim.get("occupancy") or {}

    # 平均链路利用率（验证四族负载相同）
    n_links = 4 * topo.n_nodes
    T = n_events / lam
    rho_avg = sim["total_hops"] / (n_links * T) if T > 0 else np.nan

    per_msg = sim["per_message"]
    rows = []
    for (start, end, src), (_, ts, dsts_list) in zip(ranges, streams):
        src_msgs = msgs[start:end]
        pm = per_msg[start:end]
        ok = [m for m in pm if not m["dropped"]]
        delays = np.array([m["delay"] for m in ok], dtype=float) if ok else np.empty(0)

        row = {
            "family": family,
            "run": run_id,
            "src": src,
            "n_events_actual": int(len(ts)),
            "n_messages": len(pm),
            "n_dropped": sum(1 for m in pm if m["dropped"]),
            "p95_delay": float(np.percentile(delays, 95)) if delays.size else np.nan,
            "p50_delay": float(np.percentile(delays, 50)) if delays.size else np.nan,
            "rho_avg": rho_avg,
        }

        # ---- C0 ----
        row["spike_count"] = spike_count(ts)
        row["mean_rate"] = mean_rate(ts)

        # ---- C1 ----
        sc = TimeScale.from_events(ts)
        if sc.valid:
            b = burst_poisson(ts, sc)
            row["cv"] = cv(ts)
            row["fano"] = fano(ts, sc)
            row["burst_per_event"] = b["burst_per_event"]
            row["autocorr_lag1"] = autocorr_lag1(ts, sc)
        else:
            for k in ("cv", "fano", "burst_per_event", "autocorr_lag1"):
                row[k] = np.nan

        # ---- C2 ----
        row.update(c2_features([(src, d) for d in dsts_list], topo,
                               path_table=table))

        # ---- C3 ----
        # 注意：MFDFA 的输入是 **ISI 序列**（论文 [1] 分析的是 interspike intervals），
        # 不是累积事件时刻。传 ts 会让 profile 成为二次型、h(q)≈2（已被本文件首轮输出抓到）。
        if with_c3:
            c3, degen = c3_of(np.diff(ts))
            row.update(c3)
            row["c3_degenerate"] = degen

        # ---- 该源路径上的峰值占用 ----
        # occupancy 是 list-of-list: occupancy[node][port]
        ports = set()
        for m in src_msgs:
            path = m["path"]
            for u, v in zip(path, path[1:]):
                ports.add((int(u), topo.out_port(u, v)))
        if ports and occ:
            row["peak_occupancy"] = float(max(occ[u][p] for (u, p) in ports))
        else:
            row["peak_occupancy"] = np.nan

        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*",
                    default=["poisson", "renewal", "hawkes", "lrd"])
    ap.add_argument("--runs", type=int, default=1, help="每族运行次数")
    ap.add_argument("--n-events", type=int, default=CONFIG["events_per_source"])
    ap.add_argument("--n-sources", type=int, default=64)
    ap.add_argument("--lambda-s", type=float, default=CONFIG["lambda_s"])
    ap.add_argument("--no-c3", action="store_true", help="跳过 MFDFA（快速试跑）")
    ap.add_argument("--out", default=os.path.join("results", "dataset.csv"))
    args = ap.parse_args()

    topo = Mesh(CONFIG["mesh_n"])
    print(f"mesh {topo.n}×{topo.n}, sigma={sigma_from_config():.2f}, "
          f"lambda_s={args.lambda_s}, N={args.n_events}, "
          f"sources={args.n_sources}, runs={args.runs}")
    print("building path table ...", flush=True)
    table = topo.path_table()

    all_rows = []
    t_start = time.perf_counter()
    for fam in args.families:
        for run in range(args.runs):
            t0 = time.perf_counter()
            rows = run_one(fam, run, topo, table, args.lambda_s,
                           args.n_events, args.n_sources,
                           with_c3=not args.no_c3)
            all_rows.extend(rows)
            dt = time.perf_counter() - t0
            print(f"  [{fam} run={run}] {len(rows)} rows in {dt:.1f}s", flush=True)

    if not all_rows:
        print("no rows produced")
        return

    # 统一列顺序
    cols = list(all_rows[0].keys())
    extra = sorted({k for r in all_rows for k in r} - set(cols))
    cols += extra

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in cols})

    meta = {
        "config": {
            "lambda_s": args.lambda_s, "n_events": args.n_events,
            "n_sources": args.n_sources, "runs": args.runs,
            "families": args.families, "family_params": FAMILY_PARAMS,
            "sigma": sigma_from_config(), "port_capacity": CONFIG["port_capacity"],
            "fanout_distribution": CONFIG["fanout_distribution"],
            "fanout_alpha": CONFIG["fanout_alpha"],
            "with_c3": not args.no_c3,
        },
        "n_rows": len(all_rows),
        "wall_s": time.perf_counter() - t_start,
        "nan_fraction": {
            k: float(np.mean([not np.isfinite(r.get(k, np.nan))
                              for r in all_rows])) for k in cols
            if k not in ("family",)
        },
    }
    meta_path = os.path.splitext(args.out)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {args.out}  ({len(all_rows)} rows, "
          f"{time.perf_counter()-t_start:.0f}s total)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
